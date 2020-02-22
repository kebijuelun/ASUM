import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer

import torch
from torch.autograd import Variable
import cv2

from ASUM_GUI_final import *
import argparse
import numpy as np
import serial
import time
from data import BaseTransform, VOC_CLASSES as labelmap

ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.5)  # 使用USB连接串行口

parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
parser.add_argument(
    "--weights",
    default="./models/SSD_sections_det.pth",
    type=str,
    help="Trained state_dict file path",
)
parser.add_argument("--cuda", default=True, type=bool, help="Use cuda in live demo")
args = parser.parse_args()


class MyWindow(QMainWindow, Ui_MainWindowASUM):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        # create a timer
        self.timer = QTimer()
        self.timer1 = QTimer()

        # set timer timeout callback function
        self.timer.timeout.connect(self.original_video_play)

        self.timer1.timeout.connect(self.detection_video_play)

        # set control_bt callback clicked  function
        self.camera_start.clicked.connect(self.controlTimer)
        # self.camera_start.clicked.connect(self.controlTimer)
        self.checkBox.stateChanged.connect(self.controlTimer)
        # self.checkBox.stateChanged.connect(self.state_save)
        self.spinBox_motor_speed.valueChanged["int"].connect(self.motor2wafer)

        self.spinBox_motor_speed.valueChanged["int"].connect(self.serial_motor_control)
        self.check_state = None

        self.COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX

        self.speed = "8v50\n"
        self.motor_speed = 50  # moBaseTransform(tor rotation speed, (2*pi/min)

        # for calculating the speed of motor
        self.r_wafer = 45  # silicon wafer with a radius of about 45mm
        self.reduction_ratio = 256  # reduction ratio of motor reducer
        self.d_rightbaffle = 10  # length of right baffle
        self.speed_gain = 4

        # global first_automated_process
        self.first_automated_process = True

    def serial_motor_control(self, value):
        motor_speed = value
        manual_motor_speed = "8v" + str(motor_speed) + "\n"
        ser.write(manual_motor_speed.encode())

    def motor2wafer(self, value):
        wafer_speed = value / 256
        self.lcdNumber.display(wafer_speed)

    def state_save(self, state):
        if state:
            self.check_state = True
        else:
            self.check_state = False

    def cv2_demo(self, net, transform, input_frame):
        def predict(loop_frame):

            height, width = loop_frame.shape[:2]
            x = torch.from_numpy(transform(loop_frame)[0]).permute(2, 0, 1)
            x = Variable(x.unsqueeze(0))

            global first_slice_flag
            global last_slice_flag
            global speed_start_time
            global speed

            print("motor speed:", self.speed)
            if args.cuda:
                x = x.cuda()

            y = net(x)  # forward pass
            detections = y.data
            # scale each detection back up to the image
            scale = torch.Tensor([width, height, width, height])

            # print("use cuda:",args.cuda)

            """ add control part for automatic slices collection"""
            """
            control statagy:
            1. distinguish the non collected slices between the left baffle and right baffle, 
               start the spin motor when the last slices reach the bottom of right baffle;
            2. distinguish the collected slices which on the right side of the right baffle,
               when all slices are not in the right baffle area, and when the non collected slices 
               don't reach the bottom of the right baffle, stop the motor;
            3. distinguish the slices on the left side of the left baffle, stop the motor;
            """
            ultrathin_slices = []
            left_baffle = []
            right_baffle = []
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    if i == 1:
                        ultrathin_slices.append(pt)
                    elif i == 2:
                        left_baffle.append(pt)
                    elif i == 3:
                        right_baffle.append(pt)
                    cv2.rectangle(
                        loop_frame,
                        (int(pt[0]), int(pt[1])),
                        (int(pt[2]), int(pt[3])),
                        self.COLORS[i % 3],
                        2,
                    )
                    cv2.putText(
                        loop_frame,
                        labelmap[i - 1],
                        (int(pt[0]), int(pt[1]) - 10),
                        self.FONT,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    j += 1
            """ control part"""

            if ultrathin_slices and left_baffle and right_baffle:
                non_collected_slices = []  # 未收取的切片列表，切片中心坐标
                collected_slices = []  # 收取的切片列表，切片中心坐标
                for slice in ultrathin_slices:
                    x_middle, y_middle = (
                        (slice[0] + slice[2]) / 2,
                        (slice[1] + slice[3]) / 2,
                    )
                    if x_middle > left_baffle[0][2] and x_middle < right_baffle[0][0]:
                        non_collected_slices.append([x_middle, y_middle])
                    else:
                        collected_slices.append([x_middle, y_middle])
                non_collected_slices = np.array(non_collected_slices)
                collected_slices = np.array(collected_slices)
                if len(non_collected_slices) > 0:
                    if not first_slice_flag:
                        speed_start_time = time.time()
                        first_slice_flag = True
                    ind_leftside_slices = np.array([])
                    rightbaffle_inside_slices_final = np.array([])
                    if len(collected_slices) > 0:
                        ind_leftside_slices = collected_slices[
                            collected_slices[:, 0] < left_baffle[0][0]
                        ]
                        # 区分在右挡板间的切片，如果有这样的切片代表切片收取正在进行，不能停止电机旋转
                        rightbaffle_inside_slices_temp = collected_slices[
                            collected_slices[:, 0] >= right_baffle[0][0]
                        ]
                        rightbaffle_inside_slices_final = rightbaffle_inside_slices_temp[
                            rightbaffle_inside_slices_temp[:, 0] <= right_baffle[0][2]
                        ]

                    # 未收取的切片排序，找到最下方的切片
                    sorted_noncollected_ind = np.argsort(-non_collected_slices[:, 1])
                    last_noncollected_slices = non_collected_slices[
                        sorted_noncollected_ind[0]
                    ]
                    # calculate the spin speed of motor,(speed-->(2*pi/60s))
                    if (
                        first_slice_flag
                        and not last_slice_flag
                        and (
                            len(rightbaffle_inside_slices_final) > 0
                            or last_noncollected_slices[1] >= right_baffle[0][3]
                        )
                    ):
                        speed_stop_time = time.time()
                        time_through_rightbaffle = speed_stop_time - speed_start_time
                        motor_speed = int(
                            self.d_rightbaffle
                            * self.reduction_ratio
                            * 60
                            / self.r_wafer
                            / (2 * 3.14159)
                            / time_through_rightbaffle
                            * self.speed_gain
                        )
                        self.lcdNumber.display(motor_speed)
                        if motor_speed <= 90 and motor_speed >= 50:
                            speed = "8v" + str(motor_speed) + "\n"
                        else:
                            speed = "8v70\n"
                        last_slice_flag = True

                    """
                    1. stop the motor when more than two slices are on the left side
                    2. start the motor when the last uncollected reach the bottom of right baffle
                    3. stop the motor when the last uncollected slices has not reach the bottom of right baffle
                       and there has not slices inside the rightbaffle area.
                    """
                    if (
                        len(ind_leftside_slices) > 1
                    ):  # number of slices in the leftside more than 2 indicates that collection is done
                        ser.write("8v0\n".encode())  # stop motor
                    else:
                        if (
                            last_noncollected_slices[1] >= right_baffle[0][3]
                            or len(rightbaffle_inside_slices_final) > 0
                        ):
                            ser.write(speed.encode())  # start motor
                        elif (
                            last_noncollected_slices[1] <= right_baffle[0][3]
                            and len(rightbaffle_inside_slices_final) == 0
                        ):
                            # if motor is on then stop motor else continue
                            ser.write("8v0\n".encode())  # stop motor

            return loop_frame

        det_frame = predict(input_frame)
        return det_frame

    def original_video_play(self):
        # read frame from video capture
        ret, frame = self.cap.read()

        # convert frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        # create QImage from RGB frame
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    def detection_video_play(self):
        ret, frame = self.cap.read()

        if self.first_automated_process:
            ser.write("8v0\n".encode())
            self.spinBox_motor_speed.setValue(0)
            self.first_automated_process = False
            self.lcdNumber.display(0.32)
        if args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

        import sys
        from os import path

        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

        from ssd import build_ssd

        global speed_change_flag
        speed_change_flag = False
        global first_slice_flag
        first_slice_flag = False
        global last_slice_flag
        last_slice_flag = False
        global speed_start_time
        global speed

        net = build_ssd("test", 300, 4)  # initialize SSD

        net.load_state_dict(torch.load(args.weights))
        transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

        if args.cuda:
            net = net.cuda()
            # cudnn.benchmark = True

        result_frame = self.cv2_demo(net.eval(), transform, frame)

        frame_qt_det = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        # get frame infos
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame_qt_det.data, width, height, step, QImage.Format_RGB888)
        # show frame in img_label
        self.image_label.setPixmap(QPixmap.fromImage(qImg))

    # start/stop timer
    def controlTimer(self, state):
        # if timer is stopped
        # print(state)
        if not self.timer.isActive() and not self.timer1.isActive():
            if not state:
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.timer.start(20)
                # update control_bt text
                self.camera_start.setText("Stop Camera")
            else:
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.timer1.start(20)
                # update control_bt text
                self.camera_start.setText("Stop Camera")
        # if timer is started
        elif self.timer.isActive() and state:
            self.timer.stop()
            self.timer1.start(20)
        elif self.timer1.isActive() and not state:
            self.timer1.stop()
            self.timer.start(20)
        else:
            # stop timer
            self.timer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.camera_start.setText("Start Camera")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()

    sys.exit(app.exec_())
