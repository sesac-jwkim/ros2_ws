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
from ros_robot_controller_msgs.msg import (
    BuzzerState,
    SetPWMServoState,
    PWMServoState,
    RGBStates,
    RGBState,
)

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

        # STM32 LED 제어
        self.stm32LED_publisher = self.create_publisher(
            RGBStates, "/ros_robot_controller/set_rgb", 1
        )
        self.led_index = 1  # 안 되면 0으로 바꿔서 테스트
        self.led_mode = "off"
        self.yellow_led_on = False

        self.yellow_blink_timer = self.create_timer(
            0.5, self.yellow_blink_callback  # 0.5초마다 ON/OFF 전환
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
        self.park_y = -1  # obtain the x-pixel coordinate of a parking sign

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
        self.park_y = -1  # -1: 표지판 미감지 / 0 이상: 감지됨
        self.target_crosswalk_x = -1  # -1: 아직 탐색 중 / 0 이상: 매칭 완료, 고정됨
        self.crosswalk_aligned = False  # 주차 전 직진 정렬 완료 여부
        self._current_crosswalk_x = None
        self.count_crosswalk = 0
        self.crosswalk_distance = 0  # distance to the zebra crossing
        self.crosswalk_length = 0.1 + 0.3  # the length of zebra crossing and the robot
        self.crosswalk_stop_start = 0
        self.ignore_crosswalk = False
        self.ignore_start = 0
        self.stop_red = False  # 정지 중 빨간 불 여부

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

        self.turn_right_sign = False

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

    # Turn Right 추가

    def right_action(self):
        self.turn_right_sign = True
        turn_duration = 2.2
        start_time = time.time()
        # 속도 줄인 후 90도 회전
        turn_twist = Twist()
        turn_twist.linear.x = 0.3
        turn_twist.angular.z = -0.7

        while time.time() - start_time < turn_duration and rclpy.ok():
            self.mecanum_pub.publish(turn_twist)
            time.sleep(0.05)

        self.mecanum_pub.publish(Twist())

        self.turn_right = False
        self.turn_right_sign = False
        self.pid.clear()

        self.get_logger().info("\nRight Action ...")

    # GPIO UDP 통신
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
            self.set_led_mode("red")
        else:
            if self.turn_right_sign:
                status = "right"
                self.set_led_mode("yellow_blink")

            else:
                status = "working"
                self.set_led_mode("green")

        self.send_status(status)

    # STM32 LED 제어
    def publish_stm32_rgb(self, red, green, blue):
        msg = RGBStates()

        state = RGBState()
        state.index = self.led_index
        state.red = red
        state.green = green
        state.blue = blue

        msg.states = [state]
        self.stm32LED_publisher.publish(msg)

    def yellow_blink_callback(self):
        if self.led_mode != "yellow_blink":
            return

        self.yellow_led_on = not self.yellow_led_on

        if self.yellow_led_on:
            # Yellow = Red + Green
            self.publish_stm32_rgb(255, 255, 0)
        else:
            self.publish_stm32_rgb(0, 0, 0)

    def set_led_mode(self, mode):
        self.led_mode = mode

        if mode == "red":
            self.publish_stm32_rgb(255, 0, 0)

        elif mode == "green":
            self.publish_stm32_rgb(0, 255, 0)

        elif mode == "yellow_blink":
            self.yellow_led_on = False
            # 실제 점멸은 timer callback에서 수행

        elif mode == "off":
            self.publish_stm32_rgb(0, 0, 0)

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
                # if detecting the zebra crossing, start to slow down
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                if self.ignore_crosswalk:
                    self.get_logger().info("횡단보도 무시중 무시무시")
                if self.ignore_crosswalk and time.time() - self.ignore_start > 5:
                    self.get_logger().info("무시 끝")
                    self.ignore_crosswalk = False
                    self.ignore_start = 0

                if (
                    not self.ignore_crosswalk
                    and 280 < self.crosswalk_distance
                    and not self.stop
                ):  # The robot starts to slow down only when it is close enough to the zebra crossing
                    self.count_crosswalk += 1
                    self.get_logger().info("횡단보도다ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ")
                    if (
                        self.count_crosswalk == 3
                    ):  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        self.stop = True
                        self.crosswalk_stop_start = time.time()
                        self.mecanum_pub.publish(Twist())
                # else:  # need to detect continuously, otherwise reset
                #     self.count_crosswalk = 0

                if (
                    self.crosswalk_stop_start != 0
                    and time.time() - self.crosswalk_stop_start > 3
                    and not self.stop_red
                ):

                    # 정지 종료 후, self.turn_right 가 True 이면 우회전
                    if self.turn_right:
                        self.right_action()

                    self.stop = False
                    self.crosswalk_stop_start = 0
                    self.ignore_start = time.time()
                    self.ignore_crosswalk = True
                    self.mecanum_pub.publish(twist)

                # deceleration processing
                # 주행 상태 확인
                if self.stop:
                    # 정지 중 신호등 검출
                    if self.traffic_signs_status is not None:
                        area = abs(
                            self.traffic_signs_status.box[0]
                            - self.traffic_signs_status.box[2]
                        ) * abs(
                            self.traffic_signs_status.box[1]
                            - self.traffic_signs_status.box[3]
                        )

                        if (
                            self.traffic_signs_status.class_name == "red"
                            and area < 1000
                        ):  # If the robot detects a red traffic light, it will stop
                            self.mecanum_pub.publish(Twist())
                            self.stop = True
                            self.stop_red = True
                        # 초록불
                        elif (
                            self.traffic_signs_status.class_name == "green"
                        ):  # If the traffic light is green, the robot will slow down and pass through
                            twist.linear.x = self.normal_speed
                            self.mecanum_pub.publish(twist)
                            self.stop = False
                            # 횡단보도 앞 멈춤이었을 때
                            if 280 < self.crosswalk_distance:
                                self.ignore_crosswalk = True  # 횡단보도 검출 무시
                                self.ignore_start_time = (
                                    time.time()
                                )  # 횡단보도 진입 시각 확인
                            self.stop_red = False
                else:
                    # 주행 중 신호등 검출
                    if self.traffic_signs_status is not None:
                        area = abs(
                            self.traffic_signs_status.box[0]
                            - self.traffic_signs_status.box[2]
                        ) * abs(
                            self.traffic_signs_status.box[1]
                            - self.traffic_signs_status.box[3]
                        )
                        # 빨간불 정지
                        # 첫번째 신호등 / 횡단보도에서 빨간 불
                        if (
                            self.traffic_signs_status.class_name == "red"
                        ):  # If the robot detects a red traffic light, it will stop
                            self.mecanum_pub.publish(Twist())
                            self.stop = True

                # 3. 주차 구역 처리
                # 주차 표지판이 보이고(0 < park_y), 횡단보도 같은 구분선이 보이면 안정성을 위해 감속
                if 0 <= self.park_y and 135 < self.crosswalk_distance:
                    twist.linear.x = self.slow_down_speed
                    # 정지선(횡단보도선)에 바짝 다가갔을 때(180)
                    if not self.start_park and 250 < self.crosswalk_distance:
                        self.count_park += 1
                        if (
                            self.count_park >= 15
                        ):  # 15프레임 연속 확인 후 주차 시퀀스 돌입
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True
                            threading.Thread(
                                target=self.park_action
                            ).start()  # 백그라운드에서 주차 매크로 실행
                    else:
                        self.count_park = 0

                # 4. 주행 처리: target_crosswalk_x가 확정됐으면 그 방향으로, 아니면 기존 차선 추종
                if self.target_crosswalk_x >= 0:
                    ALIGN_THRESHOLD = 15  # 픽셀, 튜닝 필요
                    # 주차 표지판 옆 횡단보도를 향해 전진 (차선 인식 완전히 무시)
                    if not self.crosswalk_aligned:
                        current_x = self._current_crosswalk_x
                        if current_x is not None:
                            error = current_x - (w / 2)
                            if abs(error) < ALIGN_THRESHOLD:
                                # 정렬 완료 -> 이후로는 PID 사용 안 함
                                self.crosswalk_aligned = True
                                self.pid.clear()
                                twist.angular.z = 0.0
                                self.get_logger().info(
                                    "\033[1;36m[정렬 완료] 이후 직진만 수행\033[0m"
                                )
                            else:
                                self.pid.SetPoint = w / 2
                                self.pid.update(current_x)
                                if self.machine_type != "MentorPi_Acker":
                                    twist.angular.z = common.set_range(
                                        self.pid.output, -0.1, 0.1
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
                            # 이번 프레임에 매칭될 crosswalk가 안 보이면 조향 유지 안 함
                            twist.angular.z = 0.0
                    else:
                        # 정렬 완료 이후: PID 완전히 무시, 직진만
                        twist.angular.z = 0.0

                    if not self.stop:
                        self.mecanum_pub.publish(twist)

                else:
                    # 4. 차선 유지(PID) 및 커브 주행 처리
                    # lane_detect.py 로부터 타겟 차선의 X좌표(무게중심)를 얻어옴
                    result_image, lane_angle, lane_x = self.lane_detect(
                        binary_image, image.copy()
                    )  # return 변수 추가, the coordinate of the line while the robot is in the middle of the lane
                    twist.linear.x = self.normal_speed
                    if lane_x >= 0 and not self.stop:
                        if (
                            lane_x > 220
                        ):  # lane_x 대신 centers로 회전 감지, default : 150
                            self.count_turn += 1
                            if self.count_turn > 4 and not self.start_turn:
                                self.start_turn = True
                                self.turn_right_sign = True
                                self.count_turn = 0
                                self.start_turn_time_stamp = time.time()
                            if self.machine_type != "MentorPi_Acker":
                                twist.linear.x = self.slow_down_speed
                                twist.angular.z = -0.7  # turning speed defualt : -0.45
                            else:
                                twist.angular.z = (
                                    twist.linear.x * math.tan(-0.5061) / 0.145
                                )
                        else:  # use PID algorithm to correct turns on a straight road
                            self.count_turn = 0
                            if (
                                time.time() - self.start_turn_time_stamp > 2
                                and self.start_turn
                            ):
                                self.start_turn = False
                                self.turn_right_sign = False
                            if not self.start_turn:
                                # defualt : 130
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
                    # self.turn_right_sign = False

                # UDP 통신 LED 제어 #
                self.send_drive_status()

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
        self.objects_info = msg.objects
        Y_MATCH_THRESHOLD = 50  # park_y와 crosswalk_y 차이 허용 범위 (튜닝 필요)
        self._current_crosswalk_x = None

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
                        center[1] > min_distance
                    ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
                    # park_y가 확정됐고, target_crosswalk_x 미확정이며,
                    # 이 crosswalk의 y가 park_y와 충분히 가까우면 x좌표를 바로 확정
                    if self.park_y > 0 and self.target_crosswalk_x < 0:
                        if abs(center[1] - self.park_y) < Y_MATCH_THRESHOLD:
                            self.target_crosswalk_x = center[0]  # 딱 한 번만 저장
                            self.get_logger().info(
                                "\033[1;35m[매칭 완료] target_crosswalk_x=%d 고정\033[0m"
                                % center[0]
                            )

                    # 정렬 단계(target 확정 O, 정렬 완료 X)일 때만
                    # target_crosswalk_x와 가장 가까운 crosswalk의 현재 위치를 갱신 (추가)
                    if self.target_crosswalk_x >= 0 and not self.crosswalk_aligned:
                        if self._current_crosswalk_x is None or abs(
                            center[0] - self.target_crosswalk_x
                        ) < abs(self._current_crosswalk_x - self.target_crosswalk_x):
                            self._current_crosswalk_x = center[0]
                elif class_name == "right":  # obtain the right turning sign
                    # self.count_right += 1
                    # self.count_right_miss = 0
                    # if self.count_right >= 5:  # If it is detected multiple times, take the right turning sign to true
                    #     self.turn_right = True
                    #     self.count_right = 0
                    if i.score >= 0.5:
                        self.count_right += 1
                        if self.count_right >= 5:
                            self.turn_right = True
                            self.count_right = 0
                elif (
                    class_name == "park"
                ):  # obtain the center coordinate of the parking sign
                    self.park_y = center[0]
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light
                    self.traffic_signs_status = i

            self.get_logger().info("\033[1;32m%s\033[0m" % class_name)
            self.crosswalk_distance = min_distance


def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
