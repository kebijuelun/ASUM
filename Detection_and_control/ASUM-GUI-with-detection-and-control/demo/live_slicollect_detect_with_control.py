from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
import numpy as np

import serial

ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.5)  # 使用USB连接串行口

parser = argparse.ArgumentParser(description="Single Shot MultiBox Detection")
parser.add_argument(
    "--weights",
    # default='../weights/ssd300_mAP_77.43_v2.pth',
    default="/home/lwz/object_detection/ssd_sliecs.pytorch/weights/originalSSD_slices/SSD_final_VOC_map_0.9696_sliceAP_0.9089.pth",
    type=str,
    help="Trained state_dict file path",
)
parser.add_argument("--cuda", default=True, type=bool, help="Use cuda in live demo")
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")


def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
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
                    frame,
                    (int(pt[0]), int(pt[1])),
                    (int(pt[2]), int(pt[3])),
                    COLORS[i % 3],
                    2,
                )
                cv2.putText(
                    frame,
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

                    speed = "2v" + str(motor_speed) + "\n"
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
                    ser.write("2v0\n".encode())  # stop motor
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
                        ser.write("2v0\n".encode())  # stop motor

        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera

    # read video for detection
    # stream = cv2.VideoCapture("/home/lwz/data/videos/slices7_cut.avi")
    stream = cv2.VideoCapture("/home/lwz/data/videos/slices7_cut.avi")
    frame_width = int(1024)
    frame_height = int(768)

    # save the detected vedio
    # ori_out = cv2.VideoWriter('/home/lwz/object_detection/ssd.pytorch/out_original.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 35, (frame_width, frame_height))
    ori_out = cv2.VideoWriter(
        "/media/lwz/BCF60E29F60DE50C/out_original.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        35,
        (frame_width, frame_height),
    )
    # det_out = cv2.VideoWriter('/home/lwz/object_detection/ssd.pytorch/out_slices7_ssdFPN_newtest.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 35, (frame_width, frame_height))

    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:

        start_time = time.time()

        # grab next frame
        # ret, frame = stream.read()
        ret, frame = stream.read()
        if ret == True:
            # if ret%2 != 0:
            #     continue

            key = cv2.waitKey(1) & 0xFF

            # update FPS counter
            fps.update()
            ori_out.write(frame)

            frame = predict(frame)

            # det_out.write(frame)
            end_time = time.time()
            current_fps = 1 / (end_time - start_time)
            print("[INFO] current. FPS: {:.2f}".format(current_fps))
            # show current FPS on screen

            # cv2.putText(frame, "FPS: {:.2f}".format(current_fps), (10, 30), 1, 2,
            #             (125, 100, 150), 2)

            # keybindings for display
            if key == ord("p"):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xFF
                    cv2.imshow("frame", frame)
                    if key2 == ord("p"):  # resume
                        break
            cv2.imshow("frame", frame)
            if key == 27:  # exit
                stream.stop()
                break
        else:
            break


if __name__ == "__main__":
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    global speed_change_flag
    speed_change_flag = False
    global first_slice_flag
    first_slice_flag = False
    global last_slice_flag
    last_slice_flag = False
    global speed_start_time
    global speed

    speed = "2v50\n"
    motor_speed = 50  # moBaseTransform(tor rotation speed, (2*pi/min)
    # for calculating the speed of motor

    r_wafer = 45  # silicon wafer with a radius of about 45mm
    reduction_ratio = 256  # reduction ratio of motor reducer
    d_rightbaffle = 10  # length of right baffle
    speed_gain = 10
    # time_through_rightbaffle = 30       # time for slices go through the right baffle
    # motor_speed = int(d_rightbaffle*reduction_ratio*60/r_wafer/(2*pi)/time_through_rightbaffle)

    net = build_ssd("test", 300, 4)  # initialize SSD
    # if args.cuda:
    #     net = torch.nn.DaBaseTransform(taParallel(net)
    # cudnn.benchmark = True
    # state_dict = torch.load(args.weights)
    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = 'module.' + k  # remove `module.`
    #     new_state_dict[name] = v
    # net.load_state_dict(new_state_dict)
    # net.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.weights)['state_dict'].items()})
    net.load_state_dict(torch.load(args.weights))
    # net.load_state_dict(torch.load(args.weights,map_location='cuda'))
    transform = BaseTransform(net.size, (104 / 256.0, 117 / 256.0, 123 / 256.0))

    if args.cuda:
        net = net.cuda()
        # cudnn.benchmark = True

    fps = FPS().start()
    cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information

    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup

    cv2.destroyAllWindows()
    # stream.stop()
