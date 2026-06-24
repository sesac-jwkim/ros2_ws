#!/usr/bin/env python3
# encoding: utf-8
# @data:2023/03/11
# @author:aiden
# lane detection for autonomous driving

import os
import cv2
import math
import queue
import threading
import numpy as np
import sdk.common as common
from cv_bridge import CvBridge

bridge = CvBridge()

# 외부 yaml 파일에서 환경 설정값(LAB 색상 공간의 최소/최대 임계값 등)을 불러옵니다.
lab_data = common.get_yaml_data("/home/ubuntu/software/lab_tool/lab_config.yaml")


class LaneDetector(object):
    def __init__(self, color):
        # 목표로 하는 차선의 색상 (예: 'yellow')
        self.target_color = color

        # 차선 인식을 위한 관심 영역(ROI, Region of Interest) 설정
        # (시작 y, 끝 y, 시작 x, 끝 x, 가중치) 형태로 화면을 가로로 3분할 합니다.
        if os.environ["DEPTH_CAMERA_TYPE"] == "ascamera":
            self.rois = (
                (338, 360, 0, 640, 0.7),
                (292, 315, 0, 640, 0.2),
                (248, 270, 0, 640, 0.1),
            )
        else:
            self.rois = (
                (450, 480, 0, 640, 0.7),
                (390, 480, 0, 640, 0.2),
                (330, 480, 0, 640, 0.1),
            )
        self.weight_sum = 1.0  # 가중치 총합 (0.7 + 0.2 + 0.1 = 1.0)

    def set_roi(self, roi):
        self.rois = roi

    @staticmethod
    def get_area_max_contour(contours, threshold=100):
        """
        여러 윤곽선(contours) 중에서 넓이가 임계치(threshold) 이상이면서 가장 큰 윤곽선을 찾습니다.
        잡음(노이즈)을 걸러내고 실제 차선 덩어리만 찾기 위한 함수입니다.
        """
        contour_area = zip(
            contours, tuple(map(lambda c: math.fabs(cv2.contourArea(c)), contours))
        )
        contour_area = tuple(filter(lambda c_a: c_a[1] > threshold, contour_area))
        if len(contour_area) > 0:
            max_c_a = max(contour_area, key=lambda c_a: c_a[1])
            return max_c_a
        return None

    def add_horizontal_line(self, image):
        """
        화면 우측 하단을 분석하여 차선이 급격히 꺾이거나 끊기는 지점(수평 가이드라인)을 찾습니다.
        """
        #   |____  --->   |————   ---> ——
        h, w = image.shape[:2]
        roi_w_min = int(w / 2)
        roi_w_max = w
        roi_h_min = 0
        roi_h_max = h
        roi = image[
            roi_h_min:roi_h_max, roi_w_min:roi_w_max
        ]  # 이미지의 오른쪽 절반만 자름
        flip_binary = cv2.flip(roi, 0)  # 상하 반전 (가장 위쪽의 픽셀을 쉽게 찾기 위해)
        max_y = cv2.minMaxLoc(flip_binary)[-1][
            1
        ]  # 값이 255(흰색)인 가장 첫 번째 픽셀의 Y좌표 추출

        return h - max_y  # 원래 이미지 기준의 Y좌표로 복원하여 반환

    def add_vertical_line_far(self, image):
        """
        먼 거리(화면 위쪽)에서 차선이 꺾이는 수직 지점을 찾아 가이드라인 직선을 생성하는 함수입니다.
        """
        h, w = image.shape[:2]
        roi_w_min = int(w / 8)
        roi_w_max = int(w / 2)
        roi_h_min = 0
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]
        flip_binary = cv2.flip(
            roi, -1
        )  # 상하좌우 반전시켜 탐색 시작점을 우측 하단으로 맞춤

        # 첫 번째 기준점 찾기
        x_0, y_0 = cv2.minMaxLoc(flip_binary)[-1]

        # 두 번째 기준점 찾기 (첫 번째 지점에서 55픽셀 아래)
        y_center = y_0 + 55
        roi = flip_binary[y_center:, :]
        x_1, y_1 = cv2.minMaxLoc(roi)[-1]
        down_p = (roi_w_max - x_1, roi_h_max - (y_1 + y_center))

        # 세 번째 기준점 찾기 (첫 번째 지점에서 65픽셀 아래)
        y_center = y_0 + 65
        roi = flip_binary[y_center:, :]
        x_2, y_2 = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x_2, roi_h_max - (y_2 + y_center))

        up_point = (0, 0)
        down_point = (0, 0)

        # 찾은 점들을 이용해 직선의 방정식을 구하고, 화면 맨 위와 아래에 닿는 최종 좌표 계산
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (
                int(
                    -down_p[1] / ((up_p[1] - down_p[1]) / (up_p[0] - down_p[0]))
                    + down_p[0]
                ),
                0,
            )
            down_point = (
                int(
                    (h - down_p[1]) / ((up_p[1] - down_p[1]) / (up_p[0] - down_p[0]))
                    + down_p[0]
                ),
                h,
            )

        return up_point, down_point

    def add_vertical_line_near(self, image):
        """
        가까운 거리(화면 아래쪽)에서 차선이 꺾이는 수직 지점을 찾아 가이드라인 직선을 생성하는 함수입니다.
        """
        # ——|         |——        |
        #   |   --->  |     --->
        h, w = image.shape[:2]
        roi_w_min = 0
        roi_w_max = int(w / 2)
        roi_h_min = int(h / 2)
        roi_h_max = h
        roi = image[roi_h_min:roi_h_max, roi_w_min:roi_w_max]  # 좌측 하단 영역만 잘라냄
        flip_binary = cv2.flip(roi, -1)  # 상하좌우 반전

        # 첫 번째 끝점
        x_0, y_0 = cv2.minMaxLoc(flip_binary)[-1]
        down_p = (roi_w_max - x_0, roi_h_max - y_0)

        # 두 번째 끝점
        x_1, y_1 = cv2.minMaxLoc(roi)[-1]
        y_center = int((roi_h_max - roi_h_min - y_1 + y_0) / 2)
        roi = flip_binary[y_center:, :]
        x, y = cv2.minMaxLoc(roi)[-1]
        up_p = (roi_w_max - x, roi_h_max - (y + y_center))

        up_point = (0, 0)
        down_point = (0, 0)

        # 직선 방정식 계산을 통한 가이드라인 연장
        if up_p[1] - down_p[1] != 0 and up_p[0] - down_p[0] != 0:
            up_point = (
                int(
                    -down_p[1] / ((up_p[1] - down_p[1]) / (up_p[0] - down_p[0]))
                    + down_p[0]
                ),
                0,
            )
            down_point = down_p

        return up_point, down_point, y_center

    def get_binary(self, image):
        """
        RGB 이미지를 LAB 색상 공간으로 변환하여 노란색 부분만 흰색(255)으로 이진화합니다.
        """
        img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)  # RGB -> LAB 변환
        img_blur = cv2.GaussianBlur(img_lab, (3, 3), 3)  # 가우시안 블러로 노이즈 제거

        # 설정된 노란색 범위로 마스크 생성
        mask = cv2.inRange(
            img_blur,
            tuple(lab_data["lab"]["Stereo"][self.target_color]["min"]),
            tuple(lab_data["lab"]["Stereo"][self.target_color]["max"]),
        )

        # 침식(erode)과 팽창(dilate)을 통해 자잘한 점 노이즈 제거
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        return dilated

    def __call__(self, image, result_image):
        """
        이진화된 이미지를 바탕으로 최종 주행 각도(Angle)를 계산합니다.
        """
        left_centroid_sum = 0
        right_centroid_sum = 0
        left_weight_sum = 0
        right_weight_sum = 0

        h, w = image.shape[:2]
        max_center_x = -1
        left_center_x = []
        right_center_x = []
        # 카메라로 보이는 트랙 너비
        track_w_pix = 500

        for roi in self.rois:
            blob = image[roi[0] : roi[1], roi[2] : roi[3]]  # ROI 구역 자르기

            # 좌우 roi 나눠서 각 영역에서 차선 검출
            left_blob = blob[:, : w // 2]
            right_blob = blob[:, w // 2 :]

            left_contours = cv2.findContours(
                left_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )[-2]
            right_contours = cv2.findContours(
                right_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
            )[-2]
            left_max_contour_area = self.get_area_max_contour(left_contours, 30)
            right_max_contour_area = self.get_area_max_contour(right_contours, 30)

            # 차선 검출 후 각 차선의 중심점 찾기
            if left_max_contour_area is not None:
                rect = cv2.minAreaRect(left_max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                for j in range(4):
                    box[j, 1] = box[j, 1] + roi[0]
                cv2.drawContours(
                    result_image, [box], -1, (255, 255, 0), 2
                )  # 화면에 사각형 박스 그리기

                # 사각형 대각선 꼭짓점을 통해 중심점 X좌표 구하기
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2, (pt1_y + pt3_y) / 2

                cv2.circle(
                    result_image,
                    (int(line_center_x), int(line_center_y)),
                    5,
                    (0, 0, 255),
                    -1,
                )
                left_center_x.append(line_center_x)
            else:
                left_center_x.append(-1)

            if right_max_contour_area is not None:
                rect = cv2.minAreaRect(right_max_contour_area[0])
                box = np.intp(cv2.boxPoints(rect))
                for j in range(4):
                    box[j, 1] = box[j, 1] + roi[0]
                cv2.drawContours(
                    result_image, [box], -1, (255, 255, 0), 2
                )  # 화면에 사각형 박스 그리기

                # 사각형 대각선 꼭짓점을 통해 중심점 X좌표 구하기
                pt1_x, pt1_y = box[0, 0], box[0, 1]
                pt3_x, pt3_y = box[2, 0], box[2, 1]
                line_center_x, line_center_y = (pt1_x + pt3_x) / 2 + w // 2, (
                    pt1_y + pt3_y
                ) / 2

                cv2.circle(
                    result_image,
                    (int(line_center_x), int(line_center_y)),
                    5,
                    (0, 0, 255),
                    -1,
                )
                right_center_x.append(line_center_x)
            else:
                right_center_x.append(-1)

        for i in range(len(left_center_x)):
            if left_center_x[i] != -1:
                weight = self.rois[i][-1]
                if left_center_x[i] > max_center_x:
                    max_center_x = left_center_x[i]
                left_centroid_sum += (
                    left_center_x[i] * weight
                )  # 구역별 가중치를 곱해 더함
                left_weight_sum += weight

        for i in range(len(right_center_x)):
            if right_center_x[i] != -1:
                weight = self.rois[i][-1]
                if right_center_x[i] > max_center_x:
                    max_center_x = right_center_x[i]
                right_centroid_sum += (
                    right_center_x[i] * weight
                )  # 구역별 가중치를 곱해 더함
                right_weight_sum += weight
        if left_weight_sum > 0:
            left_centroid_sum /= left_weight_sum

        if right_weight_sum > 0:
            right_centroid_sum /= right_weight_sum

        print("left :", left_center_x)
        print("right:", right_center_x)

        if left_centroid_sum != 0 and right_centroid_sum != 0:
            centroid_sum = (left_centroid_sum + right_centroid_sum) // 2
            self.last_center_pos = centroid_sum  # 정상 검출되었으므로 위치 기억
        elif left_centroid_sum != 0:
            centroid_sum = left_centroid_sum + track_w_pix // 2
            self.last_center_pos = centroid_sum  # 정상 검출되었으므로 위치 기억
        elif right_centroid_sum != 0:
            centroid_sum = right_centroid_sum - track_w_pix // 2
            self.last_center_pos = centroid_sum  # 정상 검출되었으므로 위치 기억
        else:
            center_pos = self.last_center_pos
            angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / (h / 2.0)))
            return result_image, angle, max_center_x
            # return result_image, None, max_center_x

        center_pos = centroid_sum  # 최종 타겟의 중심 좌표

        # 화면의 중앙점과 계산된 차선 중심점 사이의 거리를 이용해 차량 회전 각도(Angle) 계산
        angle = math.degrees(-math.atan((center_pos - (w / 2.0)) / (h / 2.0)))

        return result_image, angle, max_center_x


image_queue = queue.Queue(2)


def image_callback(ros_image):
    """
    카메라 센서로부터 이미지를 받아 큐(Queue)에 담는 ROS2 콜백 함수
    """
    cv_image = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    bgr_image = np.array(cv_image, dtype=np.uint8)
    if image_queue.full():
        image_queue.get()
    image_queue.put(bgr_image)


def main():
    running = True
    # self.get_logger().info('\033[1;32m%s\033[0m' % (*tuple(lab_data['lab']['Stereo'][self.target_color]['min']), tuple(lab_data['lab']['Stereo'][self.target_color]['max']))) # (코드설명: 디버깅 용도로 터미널에 최소/최대 임계값을 출력하던 코드이나 현재는 주석 처리됨)

    while running:
        try:
            image = image_queue.get(block=True, timeout=1)
        except queue.Empty:
            if not running:
                break
            else:
                continue

        binary_image = lane_detect.get_binary(image)
        cv2.imshow("binary", binary_image)
        img = image.copy()

        y = lane_detect.add_horizontal_line(binary_image)
        roi = [(0, y), (640, y), (640, 0), (0, 0)]
        cv2.fillPoly(
            binary_image, [np.array(roi)], [0, 0, 0]
        )  # y 좌표 윗부분(먼 곳의 노이즈)은 흑백으로 칠해 간섭 차단
        min_x = cv2.minMaxLoc(binary_image)[-1][0]
        cv2.line(
            img, (min_x, y), (640, y), (255, 255, 255), 50
        )  # 조향을 돕기 위한 굵은 가상선(수평선) 렌더링

        result_image, angle, x = lane_detect(binary_image, image.copy())

        """
        # (코드설명: 아래 3줄은 급커브 구간에서 차량 이탈을 막기 위해 '가상의 수직선'을 그리는 코드입니다. 현재 테스트 버전에서는 수평선 기능만 사용 중이며, 수직선 기능은 필요 시 주석을 해제하여 사용하도록 남겨둔 상태입니다.)
        up, down = lane_detect.add_vertical_line_far(binary_image) # (코드설명: 앞서 정의한 함수로 먼 거리 수직선 끝점 2개를 추출합니다.)
        #up, down, center = lane_detect.add_vertical_line_near(binary_image) # (코드설명: 가까운 거리의 수직선 끝점을 찾는 함수입니다. 위 함수와 선택적으로 사용하기 위해 한 번 더 주석 처리되어 있습니다.)
        cv2.line(img, up, down, (255, 255, 255), 10) # (코드설명: 추출한 2개의 끝점을 이어 화면에 두께 10의 하얀색 수직 가이드라인을 그립니다.)
        """

        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == ord("q") or key == 27:  # q 또는 ESC 누르면 종료
            break

    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    import rclpy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("lane_detect")
    lane_detect = LaneDetector("yellow")  # '노란색' 추적 모드로 객체 초기화
    node.create_subscription(
        Image, "/ascamera/camera_publisher/rgb0/image", image_callback, 1
    )
    threading.Thread(target=main, daemon=True).start()
    rclpy.spin(node)
