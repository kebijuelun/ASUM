from __future__ import print_function
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

import serial


def cv2_demo(net, transform, debug=None):
    def predict(loop_frame, debug=None):
        height, width = loop_frame.shape[:2]
        x = torch.from_numpy(transform(loop_frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        global first_slice_flag
        global last_slice_flag
        global speed_start_time
        global speed

        # print("first_slice_flag:",first_slice_flag)
        print("motor speed:", speed)
        # print("last_slice_flag:",last_slice_flag)
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
                    COLORS[i % 3],
                    2,
                )
                cv2.putText(
                    loop_frame,
                    labelmap[i - 1],
                    (int(pt[0]), int(pt[1]) - 10),
                    FONT,
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
                        d_rightbaffle
                        * reduction_ratio
                        * 60
                        / r_wafer
                        / (2 * 3.14159)
                        / time_through_rightbaffle
                        * speed_gain
                    )
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
                    if not debug:
                        ser.write("8v0\n".encode())  # stop motor
                    else:
                        pass
                else:
                    if (
                        last_noncollected_slices[1] >= right_baffle[0][3]
                        or len(rightbaffle_inside_slices_final) > 0
                    ):
                        if not debug:
                            ser.write(speed.encode())  # start motor
                        else:
                            pass
                    elif (
                        last_noncollected_slices[1] <= right_baffle[0][3]
                        and len(rightbaffle_inside_slices_final) == 0
                    ):
                        # if motor is on then stop motor else continue
                        if not debug:
                            ser.write("8v0\n".encode())  # stop motor
                        else:
                            pass
        return loop_frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream = WebcamVideoStream(src=0).start()  # default camera
    frame_width = int(1024)
    frame_height = int(768)

    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:

        start_time = time.time()

        # grab next frame
        frame = stream.read()

        key = cv2.waitKey(1) & 0xFF

        # update FPS counter
        fps.update()
        # ori_out.write(frame)

        det_frame = predict(frame)

        # det_out.write(det_frame)
        end_time = time.time()
        current_fps = 1 / (end_time - start_time)
        print("[INFO] current. FPS: {:.2f}".format(current_fps))

        # keybindings for display
        if key == ord("p"):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xFF
                cv2.imshow("frame", det_frame)
                if key2 == ord("p"):  # resume
                    break
        cv2.imshow("frame", det_frame)
        if key == 27:  # exit
            stream.stop()
            break


if __name__ == "__main__":
    import sys
    from os import path
    import argparse

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    parser = argparse.ArgumentParser(
        description="SSD pretrained detection model for ASUM"
    )
    parser.add_argument(
        "--weights",
        default="./models/SSD_sections_det.pth",
        type=str,
        help="Trained state_dict file path",
    )
    parser.add_argument(
        "--cuda", default=False, type=bool, help="Use cuda in live demo"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=True,
        help="Debug when do not have the ASUM device",
    )
    args = parser.parse_args()

    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if not args.debug:
        ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.5)  # 使用USB连接串行口

    global speed_change_flag
    speed_change_flag = False
    global first_slice_flag
    first_slice_flag = False
    global last_slice_flag
    last_slice_flag = False
    global speed_start_time
    global speed

    speed = "8v50\n"
    motor_speed = 50  # moBaseTransform(tor rotation speed, (2*pi/min)
    # for calculating the speed of motor

    r_wafer = 45  # silicon wafer with a radius of about 45mm
    reduction_ratio = 256  # reduction ratio of motor reducer
    d_rightbaffle = 10  # length of right baffle
    speed_gain = 4

    net = build_ssd("test", 300, 4)  # initialize SSD
    if args.cuda:
        net.load_state_dict(torch.load(args.weights))
    else:
        net.load_state_dict(torch.load(args.weights, map_location="cpu"))
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    if args.cuda:
        net = net.cuda()
        # cudnn.benchmark = True

    fps = FPS().start()
    cv2_demo(net.eval(), transform, debug=args.debug)
    # stop the timer and display FPS information

    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup

    cv2.destroyAllWindows()
    # stream.stop()
