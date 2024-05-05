# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example of Python controller for Mavic patrolling around the house.
   Open the robot window to see the camera view.
   This demonstrates how to go to specific world coordinates using its GPS, imu and gyroscope.
   The drone reaches a given altitude and patrols from waypoint to waypoint."""

from email.mime import image
from pickletools import uint8
from controller import Robot
import sys
import asyncio
from aiohttp import web
try:
    import numpy as np
except ImportError:
    sys.exit("Warning: 'numpy' module not found.")
import time
import socket
import select
import re
from PIL import Image
import requests
import json
import base64
from urllib.parse import unquote
# import cv2
# from http.server import HTTPServer, BaseHTTPRequestHandler
 
base_url = "http://127.0.0.1:8080"


def clamp(value, value_min, value_max):
    return min(max(value, value_min), value_max)

# class NonBlockiingHTTPServer(HTTPServer):
#     def 

class Mavic (Robot):
    # Constants, empirically found.
    K_VERTICAL_THRUST = 68.5  # with this thrust, the drone lifts.
    # Vertical offset where the robot actually targets to stabilize itself.
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0        # P constant of the vertical PID.
    K_ROLL_P = 50.0           # P constant of the roll PID.
    K_PITCH_P = 30.0          # P constant of the pitch PID.

    MAX_YAW_DISTURBANCE = 0.4
    MAX_PITCH_DISTURBANCE = -1
    # Precision between the target position and the robot position in meters
    target_precision = 0.5

    DOCUMENTS_ROOT = "./html"
 
 
    port = 8888

    def __init__(self):
        Robot.__init__(self)

        self.time_step = int(self.getBasicTimeStep())

        # Get and enable devices.
        self.camera = self.getDevice("camera")
        self.camera.enable(self.time_step)
        self.depth = self.getDevice("depth")
        self.depth.enable(self.time_step)
        self.imu = self.getDevice("inertial unit")
        self.imu.enable(self.time_step)
        self.gps = self.getDevice("gps")
        self.gps.enable(self.time_step)
        self.gyro = self.getDevice("gyro")
        self.gyro.enable(self.time_step)

        self.front_left_motor = self.getDevice("front left propeller")
        self.front_right_motor = self.getDevice("front right propeller")
        self.rear_left_motor = self.getDevice("rear left propeller")
        self.rear_right_motor = self.getDevice("rear right propeller")
        self.camera_pitch_motor = self.getDevice("camera pitch")
        self.camera_pitch_motor.setPosition(0.7)
        # self.depth_pitch_motor = self.getDevice("depth pitch")
        # self.depth_pitch_motor.setPosition(0.7)
        motors = [self.front_left_motor, self.front_right_motor,
                  self.rear_left_motor, self.rear_right_motor]
        for motor in motors:
            motor.setPosition(float('inf'))
            motor.setVelocity(1)

        self.current_pose = 6 * [0]  # X, Y, Z, yaw, pitch, roll
        self.target_position = [0, 0, 0]
        self.target_index = 0
        self.target_altitude = 0

        # Init Api
        print("init api server...")
        # 1. 创建套接字
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 2. 绑定本地信息
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(("", self.port))
        # 3. 变为监听套接字
        self.server_socket.listen(128)
 
        self.server_socket.setblocking(False)
        self.client_socket_list = list()
 
        # self.documents_root = documents_root

    def set_position(self, pos):
        """
        Set the new absolute position of the robot
        Parameters:
            pos (list): [X,Y,Z,yaw,pitch,roll] current absolute position and angles
        """
        self.current_pose = pos

    def move_to_target(self, waypoints, verbose_movement=False, verbose_target=False):
        """
        Move the robot to the given coordinates
        Parameters:
            waypoints (list): list of X,Y coordinates
            verbose_movement (bool): whether to print remaning angle and distance or not
            verbose_target (bool): whether to print targets or not
        Returns:
            yaw_disturbance (float): yaw disturbance (negative value to go on the right)
            pitch_disturbance (float): pitch disturbance (negative value to go forward)
        """

        if self.target_position[0:2] == [0, 0]:  # Initialization
            self.target_position[0:2] = waypoints[0]
            if verbose_target:
                print("First target: ", self.target_position[0:2])

        # if the robot is at the position with a precision of target_precision
        if all([abs(x1 - x2) < self.target_precision for (x1, x2) in zip(self.target_position, self.current_pose[0:2])]):

            self.target_index += 1
            if self.target_index > len(waypoints) - 1:
                self.target_index = 0
            self.target_position[0:2] = waypoints[self.target_index]
            if verbose_target:
                print("Target reached! New target: ",
                      self.target_position[0:2])

        # This will be in ]-pi;pi]
        self.target_position[2] = np.arctan2(
            self.target_position[1] - self.current_pose[1], self.target_position[0] - self.current_pose[0])
        # This is now in ]-2pi;2pi[
        angle_left = self.target_position[2] - self.current_pose[5]
        # Normalize turn angle to ]-pi;pi]
        angle_left = (angle_left + 2 * np.pi) % (2 * np.pi)
        if (angle_left > np.pi):
            angle_left -= 2 * np.pi

        # Turn the robot to the left or to the right according the value and the sign of angle_left
        yaw_disturbance = self.MAX_YAW_DISTURBANCE * angle_left / (2 * np.pi)
        # non proportional and decreasing function
        pitch_disturbance = clamp(
            np.log10(abs(angle_left)), self.MAX_PITCH_DISTURBANCE, 0.1)

        if verbose_movement:
            distance_left = np.sqrt(((self.target_position[0] - self.current_pose[0]) ** 2) + (
                (self.target_position[1] - self.current_pose[1]) ** 2))
            print("remaning angle: {:.4f}, remaning distance: {:.4f}".format(
                angle_left, distance_left))
        return yaw_disturbance, pitch_disturbance

    def run(self):
        t1 = self.getTime()

        # Specify the patrol coordinates
        waypoints = [[-30, 20], [-60, 20], [-60, 10], [-30, 5]]
        # target altitude of the robot in meters
        self.target_altitude = 1
        roll_disturbance = 0
        pitch_disturbance = 0
        yaw_disturbance = 0
        action = 0
        keep = 0
        while self.step(self.time_step) != -1:
            if action == 0:
                roll_disturbance = 0
                pitch_disturbance = 0
                yaw_disturbance = 0
            # Read sensors
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            x_pos, y_pos, altitude = self.gps.getValues()
            roll_acceleration, pitch_acceleration, _ = self.gyro.getValues()
            self.set_position([x_pos, y_pos, altitude, roll, pitch, yaw])

            # if altitude > self.target_altitude - 1:
            #     # as soon as it reach the target altitude, compute the disturbances to go to the given waypoints.
            #     if self.getTime() - t1 > 0.1:
            #         yaw_disturbance, pitch_disturbance = self.move_to_target(
            #             waypoints)
            #         t1 = self.getTime()

            # read api
            try:
                new_socket, new_addr = self.server_socket.accept()
            except Exception as ret:
                pass
                # print("-----1----", ret)  # for test
            else:
                new_socket.setblocking(False)
                self.client_socket_list.append(new_socket)
 
            for client_socket in self.client_socket_list:
                try:
                    request = client_socket.recv(1024).decode('utf-8')
                except Exception as ret:
                    # print("------2----", ret)  # for test
                    pass
                else:
                    if request:
                        (r, p, y) = self.deal_with_request(request, client_socket) #将接受到的请求组发给deal_with_request函数去完成解析和响应
                        if r != 0 or p != 0 or y != 0:
                            roll_disturbance = r
                            pitch_disturbance = p
                            yaw_disturbance = y
                            # front_left_motor_input = self.K_VERTICAL_THRUST - yaw_disturbance - 60
                            # front_right_motor_input = self.K_VERTICAL_THRUST + yaw_disturbance + 60
                            # rear_left_motor_input = self.K_VERTICAL_THRUST + yaw_disturbance  + 60
                            # rear_right_motor_input = self.K_VERTICAL_THRUST - yaw_disturbance - 60

                            # print(front_left_motor_input, front_right_motor_input, rear_left_motor_input, rear_right_motor_input)

                            # self.front_left_motor.setVelocity(front_left_motor_input)
                            # self.front_right_motor.setVelocity(-front_right_motor_input)
                            # self.rear_left_motor.setVelocity(-rear_left_motor_input)
                            # self.rear_right_motor.setVelocity(rear_right_motor_input)
                            
                            action = 20
                            
                    else:
                        client_socket.close()
                        self.client_socket_list.remove(client_socket)
            
            ###
            # roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            # pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            # yaw_input = yaw_disturbance
            # clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            # vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)

            # front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            # front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            # rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            # rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            # self.front_left_motor.setVelocity(front_left_motor_input)
            # self.front_right_motor.setVelocity(-front_right_motor_input)
            # self.rear_left_motor.setVelocity(-rear_left_motor_input)
            # self.rear_right_motor.setVelocity(rear_right_motor_input)
            # print(yaw_disturbance)
            # if yaw_disturbance != 0:
            #     print(yaw_disturbance)
            roll_input = self.K_ROLL_P * clamp(roll, -1, 1) + roll_acceleration + roll_disturbance
            pitch_input = self.K_PITCH_P * clamp(pitch, -1, 1) + pitch_acceleration + pitch_disturbance
            yaw_input = yaw_disturbance
            clamped_difference_altitude = clamp(self.target_altitude - altitude + self.K_VERTICAL_OFFSET, -1, 1)
            # if clamped_difference_altitude < 1:
            #     print(clamped_difference_altitude)
            #     print(altitude)
            vertical_input = self.K_VERTICAL_P * pow(clamped_difference_altitude, 3.0)
            # print(yaw_input)

            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input + pitch_input - roll_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input + pitch_input + roll_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input + yaw_input - pitch_input - roll_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input - yaw_input - pitch_input + roll_input

            # if clamped_difference_altitude < 1:
            #     print(front_left_motor_input)
            #     print(rear_left_motor_input)

            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)
            if action > 0:
                action -= 1

    def deal_with_request(self, request, client_socket):
        """为这个浏览器服务器"""
        # print("debug.")
        if not request:
            return
 
        request_lines = request.splitlines()
        # for i, line in enumerate(request_lines):
        #     print(i, line)
 
        # 提取请求的文件(index.html)
        # GET /a/b/c/d/e/index.html HTTP/1.1
        (roll, pitch, yaw) = 0, 0, 0
        try:
            ins = request_lines[0].split("?ins=")[1].split(" ")[0]
        except Exception as e:
            # print( request_lines[0])
            response_body = "instruction not found, 请输入正确的指令"
            response_header = "HTTP/1.1 404 not found\r\n"
            response_header += "Content-Type: text/html; charset=utf-8\r\n"
            response_header += "Content-Length: %d\r\n" % (len(response_body))
            response_header += "\r\n"
        else:
            # print(ins)
            if ins == "left":
                response_body = "turn left"
                yaw = 1.3
            elif ins == "right":
                response_body = "turn right"
                yaw = -1.3
            elif ins == "forward":
                response_body = "forward"
                pitch = -2.0
            elif ins == "capture":
                response_body = "capture"
                self.camera.saveImage(f"""C:/Users/zhaoyu/Desktop/rgb.png""", quality=100)
                self.depth.saveImage(f"""C:/Users/zhaoyu/Desktop/depth.png""", quality=100)
                # rgb = self.camera.getImage()
                # # if img:
                # #     # display the components of each pixel
                # #     for x in range(0,self.camera.getWidth()):
                # #         for y in range(0,self.camera.getHeight()):
                # #             red   = img[x][y][0]
                # #             green = img[x][y][1]
                # #             blue  = img[x][y][2]
                # rgb = np.asarray(rgb, dtype=np.uint8)
                # # rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
                # # cv2.imwrite(f"""C:/Users/zhaoyu/Desktop/rgb.jpg""", rgb)
                # img = Image.fromarray(np.array(rgb, dtype=np.uint8),  mode="RGB")
                # img.save(f"""C:/Users/zhaoyu/Desktop/rgb.jpg""")
            else:
                ins = unquote(ins)
                # print(ins)
                res = self.request_ai(ins)
                # print("Response:")
                # print(res)
                response_body = res
                if res == "TURN_LEFT":
                    
                    yaw = 1.3
                elif res == "TURN_RIGHT":
                    
                    yaw = -1.3
                elif res == "MOVE_FORWAD":
                    
                    pitch = -2.0
                
            
            # response_body = "content"
            response_header = "HTTP/1.1 200 OK\r\n"
            response_header += "Content-Length: %d\r\n" % (len(response_body))
            response_header += "\r\n"
        # ret = re.match(r"([^/]*)([^ ]+)", request_lines[0])
        # if ret:
        #     print("正则提取数据:", ret.group(1))
        #     print("正则提取数据:", ret.group(2))
        #     file_name = ret.group(2)
        #     if file_name == "/":
        #         file_name = "/index.html"
 
 
        # 读取文件数据
        # try:
        #     f = open(self.documents_root+file_name, "rb")
        # except:
        #     response_body = "file not found, 请输入正确的url"
        #     response_header = "HTTP/1.1 404 not found\r\n"
        #     response_header += "Content-Type: text/html; charset=utf-8\r\n"
        #     response_header += "Content-Length: %d\r\n" % (len(response_body))
        #     response_header += "\r\n"
 
        #     # 将header返回给浏览器
        #     client_socket.send(response_header.encode('utf-8'))
 
        #     # 将body返回给浏览器
        #     client_socket.send(response_body.encode("utf-8"))
        # else:
        #     content = f.read()
        #     f.close()
 
        #     response_body = content
        #     response_header = "HTTP/1.1 200 OK\r\n"
        #     response_header += "Content-Length: %d\r\n" % (len(response_body))
        #     response_header += "\r\n"
 
        # 将header返回给浏览器
        client_socket.send( response_header.encode('utf-8') + response_body.encode('utf-8'))
        
        return roll, pitch, yaw
 
    def request_ai(self, ins):
        self.camera.saveImage(f"""C:/Users/zhaoyu/Desktop/rgb.jpg""", quality=100)
        self.depth.saveImage(f"""C:/Users/zhaoyu/Desktop/depth.jpg""", quality=100)
        img_url_rgb = f"data:image/jpeg;base64,{encode_image('C:/Users/zhaoyu/Desktop/rgb.png')}"
        img_url_depth = f"data:image/jpeg;base64,{encode_image('C:/Users/zhaoyu/Desktop/depth.png')}"
        massages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": ins,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_url_depth
                        },
                    },
                    {
                        "type": "image_url",
                         "image_url": {
                            "url": img_url_rgb
                        },
                    }
                ],
            },
        ]
        data = {
            "model": "VLNCE",
            "messages": massages,
            "stream": False,
            "max_tokens": 190,
        }
        response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=False)

        if response.status_code == 200:
            # for line in response.iter_lines():
            #     if line:
            #         decode_line = line.decode("utf-8")[6:]
            #         try:
            #             response_json = json.loads(decode_line)
            #             content = response_json.get("choices",[{}])[0].get("delta",{}).get("content","")
            #         except Exception as e:
            #             print(e)
            #             print("Special Token:", decode_line)
            decoded_line = response.json()
            # print(decoded_line.get("choices",[{}])[0]["message"])
            content = decoded_line.get("choices",[{}])[0].get("message","").get("content","")
        return content

def encode_image(image_path):

    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode("utf-8")
# To use this controller, the basicTimeStep should be set to 8 and the defaultDamping
# with a linear and angular damping both of 0.5

print("init robot...")
robot = Mavic()
robot.run()
