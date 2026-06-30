#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/28
# @author:aiden
# autonomous driving
import os
import cv2
import math
import time
import queue
import rclpy
import threading
import numpy as np
import sdk.pid as pid
import sdk.fps as fps
from rclpy.node import Node
import sdk.common as common

# from app.common import Heart
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from interfaces.msg import ObjectsInfo
from std_srvs.srv import SetBool, Trigger
from sdk.common import colors, plot_one_box
from example.self_driving import lane_detect
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from ros_robot_controller_msgs.msg import BuzzerState, SetPWMServoState, PWMServoState

import socket


class SelfDrivingNode(Node):
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name
        self.is_running = True
        self.pid = pid.PID(0.4, 0.0, 0.15)  # defualt : 0.4 ,0.0, 0.05
        self.param_init()

        self.fps = fps.FPS()
        self.image_queue = queue.Queue(maxsize=2)
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)
        self.machine_type = os.environ.get("MACHINE_TYPE")
        self.lane_detect = lane_detect.LaneDetector("yellow")

        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)

        self.create_service(
            Trigger, "~/enter", self.enter_srv_callback
        )  # enter the game
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)  # exit the game
        self.create_service(SetBool, "~/set_running", self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))
        timer_cb_group = ReentrantCallbackGroup()
        self.client = self.create_client(Trigger, "/yolov5_ros2/init_finish")
        self.client.wait_for_service()
        self.start_yolov5_client = self.create_client(
            Trigger, "/yolov5/start", callback_group=timer_cb_group
        )
        self.start_yolov5_client.wait_for_service()
        self.stop_yolov5_client = self.create_client(
            Trigger, "/yolov5/stop", callback_group=timer_cb_group
        )
        self.stop_yolov5_client.wait_for_service()

        self.timer = self.create_timer(
            0.0, self.init_process, callback_group=timer_cb_group
        )

    def init_process(self):
        self.timer.cancel()

        self.mecanum_pub.publish(Twist())
        if not self.get_parameter("only_line_follow").value:
            self.send_request(self.start_yolov5_client, Trigger.Request())
        time.sleep(1)

        if 1:  # self.get_parameter('start').value:
            self.display = True
            self.enter_srv_callback(Trigger.Request(), Trigger.Response())
            request = SetBool.Request()
            request.data = True
            self.set_running_srv_callback(request, SetBool.Response())

        # self.park_action()
        threading.Thread(target=self.main, daemon=True).start()
        self.create_service(Trigger, "~/init_finish", self.get_node_state)
        self.get_logger().info("\033[1;32m%s\033[0m" % "start")

    def param_init(self):
        self.start = False
        self.enter = False
        self.right = True

        self.have_turn_right = False
        self.detect_turn_right = False
        self.detect_far_lane = False
        self.park_x = -1  # obtain the x-pixel coordinate of a parking sign

        self.start_turn_time_stamp = 0
        self.count_turn = 0
        self.start_turn = False  # start to turn

        self.count_right = 0
        self.count_right_miss = 0
        self.turn_right = False  # right turning sign

        self.last_park_detect = False
        self.count_park = 0
        self.stop = False  # stopping sign
        self.start_park = False  # start parking sign

        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot
        ###### 변수추가 ######
        self.ignore_crosswalk = False  # 횡단보도 통과 후 일정 시간 동안 재검출 무시
        self.ignore_start_time = 0  # 횡단보도 무시 시작 시각
        self.crosswalk_count = 0  # 현재 프레임에서 검출된 횡단보도 개수
        self.force_stop_crosswalk = False  # 첫 번째 횡단보도에서 정지 여부
        self.crosswalk_stop_start = 0  # 횡단보도 정지 시작 시각
        self.prev_crosswalk_count = 0  # 이전 프레임의 횡단보도 개수

        self.crosswalk_detected = False  # 횡단보도 최초 검출 여부
        self.crosswalk_detect_time = 0  # 최초 검출 시간
        self.crosswalk_stop_time = 0  # 계산된 직진 시간
        ######
        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.3  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []

        #### UDP 통신 변수 ####
        self.host_ip = "127.0.0.1"
        self.host_port = 5005
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.last_status = None
        self.last_send_time = 0.0
        self.send_interval = 0.2
        #####################

    def get_node_state(self, request, response):
        response.success = True
        return response

    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()

    def enter_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving enter")
        with self.lock:
            self.start = False
            camera = "depth_cam"  # self.get_parameter('depth_camera_name').value
            self.create_subscription(
                Image, "/ascamera/camera_publisher/rgb0/image", self.image_callback, 1
            )
            self.create_subscription(
                ObjectsInfo, "/yolov5_ros2/object_detect", self.get_object_callback, 1
            )
            self.mecanum_pub.publish(Twist())
            self.enter = True
        response.success = True
        response.message = "enter"
        return response

    def exit_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "self driving exit")
        with self.lock:
            try:
                if self.image_sub is not None:
                    self.image_sub.unregister()
                if self.object_sub is not None:
                    self.object_sub.unregister()
            except Exception as e:
                self.get_logger().info("\033[1;32m%s\033[0m" % str(e))
            self.mecanum_pub.publish(Twist())
        self.param_init()
        response.success = True
        response.message = "exit"
        return response

    def set_running_srv_callback(self, request, response):
        self.get_logger().info("\033[1;32m%s\033[0m" % "set_running")
        with self.lock:
            self.start = request.data
            if not self.start:
                self.mecanum_pub.publish(Twist())
        response.success = True
        response.message = "set_running"
        return response

    def shutdown(self, signum, frame):  # press 'ctrl+c' to close the program
        self.is_running = False

    def image_callback(self, ros_image):  # callback target checking
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)

    # parking processing
    def park_action(self):
        if self.machine_type == "MentorPi_Mecanum":
            twist = Twist()
            twist.linear.y = -0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.38 / 0.2)
        elif self.machine_type == "MentorPi_Acker":
            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(3)

            twist = Twist()
            twist.linear.x = 0.15
            twist.angular.z = -twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(2)

            twist = Twist()
            twist.linear.x = -0.15
            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)

        else:
            twist = Twist()
            twist.angular.z = -1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.linear.x = 0.2
            self.mecanum_pub.publish(twist)
            time.sleep(0.65 / 0.2)
            self.mecanum_pub.publish(Twist())
            twist = Twist()
            twist.angular.z = 1
            self.mecanum_pub.publish(twist)
            time.sleep(1.5)
        self.mecanum_pub.publish(Twist())

        # UDP 통신

    def send_status(self, status):
        now = time.monotonic()

        if (
            status != self.last_status
            or now - self.last_send_time >= self.send_interval
        ):
            self.sock.sendto(status.encode(), (self.host_ip, self.host_port))
            self.get_logger().info(f"Send UDP status: {status}")

            self.last_status = status
            self.last_send_time = now

    def send_drive_status(self):
        if self.stop:
            status = "stopping"
        else:
            status = "working"

        self.send_status(status)

    def main(self):
        while self.is_running:
            time_start = time.time()
            try:
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane
                binary_image = self.lane_detect.get_binary(image)

                twist = Twist()
                twist.linear.x = self.normal_speed

                # if detecting the zebra crossing, start to slow down
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                ######
                # 횡단보도 5초간 무시
                if self.ignore_crosswalk:
                    if time.time() - self.ignore_start_time > 5:
                        self.ignore_crosswalk = False
                # 횡단보도가 일정거리, 속도 늦추지않고, 횡단보도 무시 하지않음 상태.
                if (
                    70 < self.crosswalk_distance and not self.ignore_crosswalk
                ):  # The robot starts to slow down only when it is close enough to the zebra crossing

                    self.count_crosswalk += 1
                    if (
                        self.count_crosswalk == 3
                    ):  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        if not self.crosswalk_detected:
                            self.crosswalk_detected = True
                            self.crosswalk_detect_time = time.time()

                            d = self.crosswalk_distance

                            # 반비례식
                            self.crosswalk_stop_time = 120.0 / max(d, 1)
                            self.crosswalk_stop_time = max(
                                0.5, min(self.crosswalk_stop_time, 2.0)
                            )
                            self.get_logger().info(
                                f"first={d}, stop_after={self.crosswalk_stop_time:.2f}s"
                            )
                        self.force_stop_crosswalk = True
                else:  # need to detect continuously, otherwise reset
                    self.count_crosswalk = 0
                ######
                # 첫 번째 횡단보도에 도착하면 5초간 정지
                if self.crosswalk_detected and not self.ignore_crosswalk:
                    if (
                        time.time() - self.crosswalk_detect_time
                        >= self.crosswalk_stop_time
                    ):
                        self.get_logger().info(
                            f"################# crosswalk stop ################## distance={self.crosswalk_distance}"
                        )
                        self.mecanum_pub.publish(Twist())
                        self.get_logger().info("########crosswalk_stop####")
                        if self.crosswalk_stop_start == 0:
                            self.crosswalk_stop_start = time.time()
                            self.stop = True
                        # 5초 동안 정지 상태 유지
                        if time.time() - self.crosswalk_stop_start < 5:
                            # self.get_logger().info(
                            #     "#################crosswalk stop##################33"
                            # )
                            self.send_drive_status()  # UDP 추가
                            self.mecanum_pub.publish(Twist())
                            time.sleep(0.03)
                            continue
                        # 횡단보도 정지 완료 후 상태 초기화
                        self.stop = False
                        self.force_stop_crosswalk = False
                        self.crosswalk_stop_start = 0
                        twist.linear.x = self.normal_speed
                        self.mecanum_pub.publish(twist)
                        ##
                        self.crosswalk_detected = False
                        self.crosswalk_detect_time = 0
                        self.crosswalk_stop_time = 0
                        # ignore 변수 초기화
                        self.ignore_crosswalk = True
                        self.ignore_start_time = time.time()
                        self.get_logger().info("#######################go###########")

                # UDP 통신 LED 제어 #
                self.send_drive_status()

                # 횡단보도 진입시 스탑 여부 확인
                # if self.stop:
                #     if self.traffic_signs_status is not None:
                #         area = abs(
                #             self.traffic_signs_status.box[0]
                #             - self.traffic_signs_status.box[2]
                #         ) * abs(
                #             self.traffic_signs_status.box[1]
                #             - self.traffic_signs_status.box[3]
                #         )
                #         if (
                #             self.traffic_signs_status.class_name == "red"
                #             and area < 1000
                #         ):  # If the robot detects a red traffic light, it will stop
                #             self.mecanum_pub.publish(Twist())
                #             self.stop = True
                #         elif (
                #             self.traffic_signs_status.class_name == "green"
                #             # 초록불 인지 범위 설정
                #             and area < 1000
                #         ):  # If the traffic light is green, the robot will slow down and pass through
                #             twist.linear.x = self.slow_down_speed
                #             self.stop = False
                #             self.ignore_crosswalk = True # 횡단보도 검출 무시
                #             self.ignore_start_time = time.time() # 횡단보도 진입 시각 확인
                # else:
                #   twist.linear.x = self.normal_speed  # go straight with normal speed

                # # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition
                # if 0 < self.park_x and 135 < self.crosswalk_distance:
                #     twist.linear.x = self.slow_down_speed
                #     if not self.start_park and 180 < self.crosswalk_distance:  # When the robot is close enough to the crosswalk, it will start parking
                #         self.count_park += 1
                #         if self.count_park >= 15:
                #             self.mecanum_pub.publish(Twist())
                #             self.start_park = True
                #             self.stop = True
                #             threading.Thread(target=self.park_action).start()
                #     else:
                #         self.count_park = 0

                # line following processing
                result_image, lane_angle, lane_x = self.lane_detect(
                    binary_image, image.copy()
                )  # return 변수 추가, the coordinate of the line while the robot is in the middle of the lane

                if lane_x >= 0 and not self.stop:
                    if lane_x > 220:  # lane_x 대신 centers로 회전 감지, default : 150
                        self.count_turn += 1
                        if self.count_turn > 5 and not self.start_turn:
                            self.start_turn = True
                            self.count_turn = 0
                            self.start_turn_time_stamp = time.time()
                        if self.machine_type != "MentorPi_Acker":
                            twist.linear.x = self.slow_down_speed
                            twist.angular.z = -0.7  # turning speed defualt : -0.45
                        else:
                            twist.angular.z = twist.linear.x * math.tan(-0.5061) / 0.145
                    else:  # use PID algorithm to correct turns on a straight road
                        self.count_turn = 0
                        if (
                            time.time() - self.start_turn_time_stamp > 2
                            and self.start_turn
                        ):
                            self.start_turn = False
                        if not self.start_turn:
                            # defualt : 130
                            twist.linear.x = self.normal_speed
                            self.pid.SetPoint = 200  # the coordinate of the line while the robot is in the middle of the lane
                            self.pid.update(lane_x)
                            if self.machine_type != "MentorPi_Acker":
                                twist.angular.z = common.set_range(
                                    self.pid.output, -0.3, 0.3
                                )

                            else:
                                twist.angular.z = (
                                    twist.linear.x
                                    * math.tan(
                                        common.set_range(self.pid.output, -0.1, 0.1)
                                    )
                                    / 0.145
                                )
                        else:
                            if self.machine_type == "MentorPi_Acker":
                                twist.angular.z = 0.15 * math.tan(-0.5061) / 0.145
                    self.mecanum_pub.publish(twist)
                else:
                    self.pid.clear()

                if self.objects_info:
                    for i in self.objects_info:
                        box = i.box
                        class_name = i.class_name
                        cls_conf = i.score
                        cls_id = self.classes.index(class_name)
                        color = colors(cls_id, True)
                        plot_one_box(
                            box,
                            result_image,
                            color=color,
                            label="{}:{:.2f}".format(class_name, cls_conf),
                        )

            else:
                time.sleep(0.01)

            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)

            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))

            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    #############################
    # 작은 박스 무시
    # Obtain the target detection result
    def get_object_callback(self, msg):
        now = time.time()
        if hasattr(self, "last_yolo_time"):
            self.get_logger().info(f"YOLO FPS = {1/(now-self.last_yolo_time):.1f}")

        self.last_yolo_time = now
        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:
            min_distance = 0

            for i in self.objects_info:
                class_name = i.class_name
                center = (
                    int((i.box[0] + i.box[2]) / 2),
                    int((i.box[1] + i.box[3]) / 2),
                )
                if class_name == "crosswalk":
                    if (
                        i.box[3] > min_distance
                    ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = i.box[3]
                elif class_name == "right":  # obtain the right turning sign
                    self.count_right += 1
                    self.count_right_miss = 0
                    if (
                        self.count_right >= 5
                    ):  # If it is detected multiple times, take the right turning sign to true
                        self.turn_right = True
                        self.count_right = 0
                elif (
                    class_name == "park"
                ):  # obtain the center coordinate of the parking sign
                    self.park_x = center[0]
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    self.traffic_signs_status = i

            self.get_logger().info("\033[1;32m%s\033[0m" % class_name)
            # 가장 가까운 횡단보도의 위치 저장
            self.crosswalk_distance = min_distance


def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
