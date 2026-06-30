import socket
import time
from gpiozero import LED

"""
LED PIN
Green  : 17
Red    : 27
Yellow : 22
"""

led_g = LED(17)
led_r = LED(27)
led_y = LED(22)

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# 메시지가 없어도 while 루프가 계속 돌도록 timeout 설정
sock.settimeout(0.1)

print(f"GPIO UDP Server started on {UDP_IP}:{UDP_PORT}")

current_status = None
prev_status = None

yellow_on = False
last_blink_time = time.monotonic()
blink_interval = 0.5

try:
    while True:
        # 1. UDP 메시지 수신 시도
        try:
            data, addr = sock.recvfrom(1024)
            current_status = data.decode().strip().lower()

            print(f"Received: {current_status} from {addr}")

        except socket.timeout:
            # 새 메시지가 없어도 current_status는 그대로 유지됨
            pass

        # 2. 상태가 바뀌었을 때만 기본 LED 상태 설정
        if current_status != prev_status:
            if current_status == "working":
                led_g.on()
                led_r.off()
                led_y.off()
                yellow_on = False
                print("Vehicle Working")

            elif current_status == "stopping":
                led_g.off()
                led_r.on()
                led_y.off()
                yellow_on = False
                print("Vehicle Stopping")

            elif current_status == "right":
                led_g.on()
                led_r.off()
                led_y.off()
                yellow_on = False
                last_blink_time = time.monotonic()
                print("Right Turn")

            elif current_status is None:
                pass

            else:
                led_g.off()
                led_r.off()
                led_y.off()
                yellow_on = False
                print(f"Unknown status: {current_status}")

            prev_status = current_status

        # 3. 현재 상태가 right이면 yellow LED 점멸 유지
        if current_status == "right":
            now = time.monotonic()

            if now - last_blink_time >= blink_interval:
                yellow_on = not yellow_on

                if yellow_on:
                    led_y.on()
                else:
                    led_y.off()

                last_blink_time = now

        # 4. working/stopping 상태에서는 LED 상태 유지
        elif current_status == "working":
            # green ON 상태를 유지
            pass

        elif current_status == "stopping":
            # red ON 상태를 유지
            pass

except KeyboardInterrupt:
    pass

finally:
    led_g.off()
    led_r.off()
    led_y.off()
    sock.close()
    print("GPIO UDP Server Stopped")
