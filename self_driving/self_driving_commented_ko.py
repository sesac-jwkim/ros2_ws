#!/usr/bin/env python3
# encoding: utf-8
# ================================================================
# 비전공자용 상세 주석 버전
# ================================================================
# 원본 코드의 동작은 바꾸지 않고, 이해를 돕기 위한 설명 주석을 추가한 파일입니다.
#
# 전체 목적:
#   1) 카메라 이미지를 받는다.
#   2) 차선을 인식한다.
#   3) YOLO가 인식한 표지판/신호등/횡단보도 정보를 받는다.
#   4) 차선 중심을 화면 중앙에 맞추도록 PID로 회전값을 계산한다.
#   5) Twist 메시지로 로봇에게 전진/회전 명령을 보낸다.
#
# 주의:
#   - 이 파일은 학습/수정용 주석 파일입니다.
#   - 실제 실행 전에 원본 파일을 백업해 두세요.
#   - 속도, PID, 회전 제한값은 한 번에 크게 바꾸지 말고 조금씩 바꾸세요.
# ================================================================
# @data:2023/03/28
# @author:aiden
# autonomous driving

# ================================================================
# [1] 필요한 라이브러리 불러오기
# ================================================================
# import는 '다른 사람이 만들어 둔 기능을 가져와서 쓰겠다'는 뜻입니다.
# 이 코드는 ROS2, OpenCV, NumPy, 로봇 제어 메시지, 차선 인식 모듈을 사용합니다.
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




# ================================================================
# [2] SelfDrivingNode 클래스
# ================================================================
# ROS2에서 Node는 '하나의 실행 프로그램 단위'입니다.
# 이 클래스 하나가 카메라 입력, YOLO 객체 인식 결과 입력, 차선 인식, 모터 명령 출력을 모두 담당합니다.
# 쉽게 말해 이 파일의 핵심 본체입니다.
class SelfDrivingNode(Node):

    # ------------------------------------------------------------
    # __init__ : 노드가 처음 만들어질 때 딱 한 번 실행되는 초기 설정 함수
    # ------------------------------------------------------------
    # 여기서 PID, 카메라 큐, YOLO 클래스 목록, Publisher/Service/Client 등을 준비합니다.
    def __init__(self, name):
        rclpy.init()
        super().__init__(
            name,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True,
        )
        self.name = name
        self.is_running = True

        # PID 제어기 생성
        # PID는 '현재 차선 중심'과 '목표 화면 중앙'의 차이를 보고 회전값을 계산합니다.
        # 0.4 = P값: 오차에 얼마나 강하게 반응할지
        # 0.0 = I값: 누적 오차를 얼마나 반영할지, 현재는 사용 안 함
        # 0.05 = D값: 갑자기 변하는 오차를 얼마나 잡아줄지
        # 로봇이 좌우로 심하게 흔들리면 P/D를 낮추고, 반응이 너무 느리면 조금 올립니다.
        self.pid = pid.PID(0.4, 0.0, 0.05)
        self.param_init()

        self.fps = fps.FPS()

        # 카메라 이미지를 잠깐 담아두는 대기줄입니다.
        # maxsize=2인 이유: 오래된 화면을 보고 움직이면 위험하므로 최신 이미지 위주로 처리하기 위해서입니다.
        self.image_queue = queue.Queue(maxsize=2)

        # YOLO가 구분할 객체 이름 목록입니다.
        # go/right/park/red/green/crosswalk 같은 이름이 들어오면 이 코드가 의미를 해석합니다.
        self.classes = ["go", "right", "park", "red", "green", "crosswalk"]
        self.display = True

        # ROS 카메라 이미지 메시지를 OpenCV 이미지로 바꾸는 변환기입니다.
        # ROS의 Image 타입은 cv2가 바로 처리할 수 없어서 CvBridge가 필요합니다.
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        self.colors = common.Colors()
        # signal.signal(signal.SIGINT, self.shutdown)

        # 로봇 종류를 환경변수에서 읽어옵니다.
        # 로봇 종류에 따라 주차 동작이나 조향 방식이 달라질 수 있습니다.
        self.machine_type = os.environ.get("MACHINE_TYPE")

        # 차선 인식기 생성
        # 현재는 'yellow', 즉 노란색 차선을 따라가도록 설정되어 있습니다.
        # 흰색 차선을 쓰려면 lane_detect.py가 white를 지원하는지 확인한 뒤 "white"로 바꿔야 합니다.
        self.lane_detect = lane_detect.LaneDetector("yellow")


        # 로봇 바퀴에 이동 명령을 보내는 Publisher입니다.
        # /controller/cmd_vel 토픽으로 Twist 메시지를 보내면 로봇이 움직입니다.
        # Twist.linear.x  : 전진/후진 속도
        # Twist.linear.y  : 좌우 이동 속도, 메카넘 바퀴일 때 사용 가능
        # Twist.angular.z : 회전 속도
        self.mecanum_pub = self.create_publisher(Twist, "/controller/cmd_vel", 1)
        self.servo_state_pub = self.create_publisher(
            SetPWMServoState, "ros_robot_controller/pwm_servo/set_state", 1
        )

        # 처리 결과 이미지를 밖으로 내보내는 Publisher입니다.
        # 차선 표시, YOLO 박스, FPS가 그려진 이미지를 확인할 때 사용합니다.
        self.result_publisher = self.create_publisher(Image, "~/image_result", 1)


        # Service는 외부에서 '명령'을 호출할 수 있게 하는 통로입니다.
        # 여기서는 enter/exit/set_running 같은 명령을 받을 수 있게 준비합니다.
        self.create_service(
            Trigger, "~/enter", self.enter_srv_callback
        )  # enter the game
        self.create_service(Trigger, "~/exit", self.exit_srv_callback)  # exit the game
        self.create_service(SetBool, "~/set_running", self.set_running_srv_callback)
        # self.heart = Heart(self.name + '/heartbeat', 5, lambda _: self.exit_srv_callback(None))

        # 콜백 그룹 설정입니다.
        # ROS2에서 여러 콜백이 동시에 돌 수 있게 할 때 사용합니다.
        # 비전공자 입장에서는 '서비스/타이머/클라이언트 실행 충돌을 줄이는 설정' 정도로 보면 됩니다.
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


    # ------------------------------------------------------------
    # init_process : 노드 시작 후 실제 실행 준비를 하는 함수
    # ------------------------------------------------------------
    # 타이머로 한 번 호출된 뒤, YOLO 시작 요청을 보내고 자율주행을 켭니다.
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


    # ------------------------------------------------------------
    # param_init : 로봇 상태값을 초기화하는 함수
    # ------------------------------------------------------------
    # 정지/시작/주차/횡단보도/신호등 등 주행 상태를 처음 상태로 되돌립니다.
    # exit할 때도 다시 호출되므로 '리셋 버튼' 같은 역할입니다.
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

        self.start_slow_down = False  # slowing down sign
        self.normal_speed = 0.1  # normal driving speed
        self.slow_down_speed = 0.1  # slowing down speed

        self.traffic_signs_status = None  # record the state of the traffic lights
        self.red_loss_count = 0

        self.object_sub = None
        self.image_sub = None
        self.objects_info = []


    # ------------------------------------------------------------
    # get_node_state : 노드가 준비됐는지 알려주는 서비스 콜백
    # ------------------------------------------------------------
    def get_node_state(self, request, response):
        response.success = True
        return response


    # ------------------------------------------------------------
    # send_request : 다른 ROS2 서비스에 요청을 보내고 응답을 기다리는 함수
    # ------------------------------------------------------------
    # 여기서는 YOLO 시작/초기화 같은 서비스를 호출할 때 사용합니다.
    def send_request(self, client, msg):
        future = client.call_async(msg)
        while rclpy.ok():
            if future.done() and future.result():
                return future.result()


    # ------------------------------------------------------------
    # enter_srv_callback : 자율주행 모드에 진입할 때 실행되는 함수
    # ------------------------------------------------------------
    # 카메라 토픽과 YOLO 객체 인식 토픽을 구독하기 시작합니다.
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


    # ------------------------------------------------------------
    # exit_srv_callback : 자율주행 모드에서 나갈 때 실행되는 함수
    # ------------------------------------------------------------
    # 구독을 해제하고, 로봇을 정지시키고, 상태값을 초기화합니다.
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


    # ------------------------------------------------------------
    # set_running_srv_callback : 실제 주행 시작/정지를 바꾸는 함수
    # ------------------------------------------------------------
    # request.data가 True면 주행 시작, False면 정지입니다.
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

        # ROS 이미지 메시지를 OpenCV에서 처리 가능한 이미지 배열로 변환합니다.
        # "rgb8"은 빨강/초록/파랑 3채널 8비트 이미지라는 뜻입니다.
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, "rgb8")
        rgb_image = np.array(cv_image, dtype=np.uint8)

        # 이미지 큐가 가득 찼다면 오래된 이미지를 하나 버립니다.
        # 로봇은 최신 화면을 보고 판단해야 하므로 오래된 프레임은 필요 없습니다.
        if self.image_queue.full():
            # if the queue is full, remove the oldest image
            self.image_queue.get()
        # put the image into the queue
        self.image_queue.put(rgb_image)

    # parking processing

    # ------------------------------------------------------------
    # park_action : 주차 표지판/조건이 만족됐을 때 실행되는 주차 동작
    # ------------------------------------------------------------
    # 로봇 종류에 따라 조금씩 다른 주차 움직임을 합니다.
    # time.sleep()은 '이 속도로 몇 초 동안 움직여라'라는 의미로 사용됩니다.
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


    # ------------------------------------------------------------
    # main : 실제 자율주행이 계속 반복되는 핵심 루프
    # ------------------------------------------------------------
    # 카메라 이미지를 가져와서 차선/횡단보도/신호등/주차 조건을 판단하고
    # 최종적으로 Twist 메시지를 publish하여 로봇을 움직입니다.
    def main(self):
        while self.is_running:

            # 이번 반복이 시작된 시간을 기록합니다.
            # 아래에서 처리 시간이 너무 짧으면 sleep을 걸어 전체 루프 속도를 맞춥니다.
            time_start = time.time()
            try:

                # 카메라 이미지 큐에서 이미지 한 장을 꺼냅니다.
                # timeout=1은 최대 1초만 기다리겠다는 뜻입니다.
                image = self.image_queue.get(block=True, timeout=1)
            except queue.Empty:
                if not self.is_running:
                    break
                else:
                    continue

            result_image = image.copy()

            # self.start가 True일 때만 자율주행 판단을 수행합니다.
            # False면 이미지를 받아도 로봇은 움직이지 않습니다.
            if self.start:
                h, w = image.shape[:2]

                # obtain the binary image of the lane

                # 원본 이미지에서 차선만 잘 보이도록 이진화 이미지를 만듭니다.
                # 예: 노란 차선은 흰색, 나머지는 검은색처럼 바꾸는 과정입니다.
                binary_image = self.lane_detect.get_binary(image)


                # 이번 루프에서 로봇에게 보낼 이동 명령 객체를 새로 만듭니다.
                # 여기에 전진 속도와 회전 속도를 채운 뒤 publish합니다.
                twist = Twist()

                # if detecting the zebra crossing, start to slow down
                self.get_logger().info("\033[1;33m%s\033[0m" % self.crosswalk_distance)
                if (
                    70 < self.crosswalk_distance and not self.start_slow_down
                ):  # The robot starts to slow down only when it is close enough to the zebra crossing

                    # 횡단보도가 감지됐다고 바로 믿지 않고 카운트를 올립니다.
                    # 몇 번 연속 확인해야 진짜라고 판단하기 위함입니다.
                    self.count_crosswalk += 1
                    if (
                        self.count_crosswalk == 3
                    ):  # judge multiple times to prevent false detection
                        self.count_crosswalk = 0
                        self.start_slow_down = True  # sign for slowing down
                        self.count_slow_down = (
                            time.time()
                        )  # fixing time for slowing down
                else:  # need to detect continuously, otherwise reset
                    self.count_crosswalk = 0

                # deceleration processing

                # 감속 모드일 때의 처리입니다.
                # 횡단보도 근처에서는 신호등 상태를 보고 정지하거나 천천히 지나갑니다.
                if self.start_slow_down:
                    if self.traffic_signs_status is not None:

                        # YOLO가 잡은 신호등 박스의 면적을 계산합니다.
                        # 보통 가까울수록 박스 면적이 커집니다.
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

                            # 빈 Twist()를 보내면 속도와 회전이 모두 0이므로 로봇이 멈춥니다.
                            self.mecanum_pub.publish(Twist())
                            self.stop = True
                        elif (
                            self.traffic_signs_status.class_name == "green"
                        ):  # If the traffic light is green, the robot will slow down and pass through

                            # 초록불이면 멈추지 않고 감속 속도로 전진합니다.
                            twist.linear.x = self.slow_down_speed
                            self.stop = False
                    if (
                        not self.stop
                    ):  # In other cases where the robot is not stopped, slow down the speed and calculate the time needed to pass through the crosswalk. The time needed is equal to the length of the crosswalk divided by the driving speed
                        twist.linear.x = self.slow_down_speed
                        if (
                            time.time() - self.count_slow_down
                            > self.crosswalk_length / twist.linear.x
                        ):
                            self.start_slow_down = False
                else:
                    twist.linear.x = self.normal_speed  # go straight with normal speed

                # If the robot detects a stop sign and a crosswalk, it will slow down to ensure stable recognition

                # 주차 표지판이 보이고, 횡단보도도 어느 정도 가까워졌을 때 주차 준비 상태로 봅니다.
                # park_x는 주차 표지판 중심의 x좌표입니다. 0보다 크면 표지판을 본 적이 있다는 뜻입니다.
                if 0 < self.park_x and 135 < self.crosswalk_distance:

                        # 정지 상태가 아니라면 횡단보도를 천천히 통과합니다.
                    twist.linear.x = self.slow_down_speed
                    if (
                        not self.start_park and 180 < self.crosswalk_distance
                    ):  # When the robot is close enough to the crosswalk, it will start parking

                        # 주차 조건도 한 번만 보고 바로 실행하지 않고 여러 번 확인합니다.
                        self.count_park += 1
                        if self.count_park >= 15:

                            # 빈 Twist()를 보내면 속도와 회전이 모두 0이므로 로봇이 멈춥니다.
                            self.mecanum_pub.publish(Twist())
                            self.start_park = True
                            self.stop = True

                            # 주차 동작은 별도 스레드에서 실행합니다.
                            # 메인 루프가 완전히 멈추지 않게 하려는 구조입니다.
                            threading.Thread(target=self.park_action).start()
                    else:
                        self.count_park = 0

                ##################################################################################
                # line following processing
                # lane_angle: , lane_x : lanes center
                # line following processing

                # 차선 인식 실행
                # lane_center_x는 화면에서 차선 중심의 x좌표입니다.
                # 이 값이 화면 중앙에 오도록 로봇을 좌우로 회전시킵니다.
                result_image, lane_angle, lane_center_x = self.lane_detect(
                    binary_image, image.copy()
                )


                # lane_center_x가 0 이상이면 차선을 찾았다는 뜻입니다.
                # self.stop이 False일 때만 주행 제어를 계속합니다.
                if lane_center_x >= 0 and not self.stop:
                    # 차선 중앙을 화면 중앙에 맞추는 PID
                    self.pid.SetPoint = w / 2.0  # 640 기준이면 320

                    # 현재 차선 중심 x좌표를 PID에 넣어서 회전 보정값을 계산합니다.
                    self.pid.update(lane_center_x)

                    if self.machine_type != "MentorPi_Acker":
                        # PID 출력값을 회전 속도로 사용

                        # PID 출력값을 로봇 회전 속도로 사용합니다.
                        # -0.25~0.25로 제한해서 너무 급격하게 돌지 않게 합니다.
                        twist.angular.z = common.set_range(self.pid.output, -0.25, 0.25)
                    else:
                        # Ackermann 타입이면 조향각처럼 사용

                        # Ackermann 방식 차량은 바퀴를 직접 회전시키는 느낌이라 조향각 범위로 제한합니다.
                        steer_angle = common.set_range(self.pid.output, -0.35, 0.35)
                        twist.angular.z = twist.linear.x * math.tan(steer_angle) / 0.145


                    # 최종 계산된 전진 속도와 회전 속도를 로봇에게 보냅니다.
                    # 이 줄이 실제로 로봇을 움직이게 하는 핵심입니다.
                    self.mecanum_pub.publish(twist)

                else:

                    # 차선을 못 찾았거나 정지 상태이면 PID 내부 값을 초기화합니다.
                    # 안전하게 하려면 여기에 self.mecanum_pub.publish(Twist())를 추가해 정지시킬 수도 있습니다.
                    self.pid.clear()

                #################################################################################

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


            # OpenCV 표시/출력용으로 RGB 이미지를 BGR 이미지로 바꿉니다.
            # OpenCV는 보통 BGR 순서를 사용합니다.
            bgr_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            if self.display:
                self.fps.update()
                bgr_image = self.fps.show_fps(bgr_image)


            # 처리 결과 이미지를 다시 ROS Image 메시지로 바꿔서 publish합니다.
            # RViz나 rqt_image_view 같은 도구에서 결과 화면을 확인할 수 있습니다.
            self.result_publisher.publish(self.bridge.cv2_to_imgmsg(bgr_image, "bgr8"))


            # 루프 주기를 대략 0.03초, 즉 약 30FPS 수준으로 맞추려는 코드입니다.
            # 이번 반복이 빨리 끝났으면 남은 시간만큼 잠깐 쉽니다.
            time_d = 0.03 - (time.time() - time_start)
            if time_d > 0:
                time.sleep(time_d)
        self.mecanum_pub.publish(Twist())
        rclpy.shutdown()

    # Obtain the target detection result

    # ------------------------------------------------------------
    # get_object_callback : YOLO 객체 인식 결과가 들어올 때마다 실행되는 함수
    # ------------------------------------------------------------
    # 인식된 객체 목록을 보고 횡단보도, 우회전 표지판, 주차 표지판, 신호등 상태를 저장합니다.
    def get_object_callback(self, msg):

        # YOLO가 현재 프레임에서 찾은 객체 목록입니다.
        # 각 객체에는 class_name, box, score 같은 정보가 들어 있습니다.
        self.objects_info = msg.objects
        if self.objects_info == []:  # If it is not recognized, reset the variable
            self.traffic_signs_status = None
            self.crosswalk_distance = 0
        else:

            # 횡단보도 중심 y좌표 중 가장 큰 값을 저장하기 위한 변수입니다.
            # 화면에서는 y좌표가 클수록 아래쪽이고, 보통 아래쪽일수록 로봇과 더 가깝습니다.
            min_distance = 0
            for i in self.objects_info:
                class_name = i.class_name

                # 객체 박스의 중심 좌표를 계산합니다.
                # center[0]은 x좌표, center[1]은 y좌표입니다.
                center = (
                    int((i.box[0] + i.box[2]) / 2),
                    int((i.box[1] + i.box[3]) / 2),
                )


                # 횡단보도를 찾은 경우입니다.
                # 가장 가까운 횡단보도를 판단하기 위해 y좌표가 가장 큰 값을 저장합니다.
                if class_name == "crosswalk":
                    if (
                        center[1] > min_distance
                    ):  # Obtain recent y-axis pixel coordinate of the crosswalk
                        min_distance = center[1]
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

                    # 주차 표지판 중심의 x좌표를 저장합니다.
                    # 이 값이 0보다 크면 주차 표지판을 본 상태로 판단할 수 있습니다.
                    self.park_x = center[0]
                elif (
                    class_name == "red" or class_name == "green"
                ):  # obtain the status of the traffic light

                    # 빨간불/초록불 객체 정보를 저장합니다.
                    # 메인 루프에서 이 값을 보고 정지 또는 감속 통과를 결정합니다.
                    self.traffic_signs_status = i

            self.get_logger().info("\033[1;32m%s\033[0m" % class_name)

            # 가장 가까운 횡단보도의 화면 y좌표를 저장합니다.
            # 이름은 distance지만 실제 거리가 아니라 화면 위치값에 가깝습니다.
            self.crosswalk_distance = min_distance




# ================================================================
# [3] 프로그램 시작점
# ================================================================
# 파이썬 파일을 직접 실행하면 아래 main() 함수가 실행됩니다.
# 여기서 SelfDrivingNode를 만들고 ROS2 executor에 등록한 뒤 계속 spin합니다.
def main():
    node = SelfDrivingNode("self_driving")
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # spin은 ROS2 노드를 계속 살아있게 하면서 콜백을 처리하는 함수입니다.
    # 카메라 이미지, YOLO 결과, 서비스 요청 등이 들어오면 해당 콜백이 실행됩니다.
    executor.spin()
    node.destroy_node()


if __name__ == "__main__":
    main()
