from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights',
                    # default='../weights/ssd300_mAP_77.43_v2.pth',
                    default='/home/lwz/object_detection/ssd_sliecs.pytorch/weights/FPN_SSD_slices/Final_FPN_SSD_VOC_SliceAP_0.9089_MAP0.9696.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def cv2_demo(net, transform):
    def predict(frame):
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if args.cuda:
            x = x.cuda()

        y = net(x)  # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])

        # print("use cuda:",args.cuda)

        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              COLORS[i % 3], 2)
                cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])-10),
                            FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera

    # read video for detection
    stream = cv2.VideoCapture("/home/lwz/data/videos/slices7_cut.avi")
    # stream = cv2.VideoCapture(0)
    frame_width = int(1024)

    frame_height = int(768)


    # save the detected vedio
    out = cv2.VideoWriter('/home/lwz/object_detection/ssd_sliecs.pytorch/out_slices7_ssdFPN_test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 35, (frame_width, frame_height))

    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        start_time = time.time()
        ret = True
        # grab next frame
        ret, frame = stream.read()
        if ret == True:
            key = cv2.waitKey(1) & 0xFF

            # update FPS counter
            fps.update()

            frame = predict(frame)
            out.write(frame)
            end_time = time.time()
            current_fps = 1/(end_time - start_time)
            print("[INFO] current. FPS: {:.2f}".format(current_fps))
            # show current FPS on screen

            # cv2.putText(frame, "FPS: {:.2f}".format(current_fps), (10, 30), 1, 2,
            #             (125, 100, 150), 2)

            # keybindings for display
            if key == ord('p'):  # pause
                while True:
                    key2 = cv2.waitKey(1) or 0xff
                    cv2.imshow('frame', frame)
                    if key2 == ord('p'):  # resume
                        break
            cv2.imshow('frame', frame)
            if key == 27:  # exit
                stream.stop()
                break
        else:
            break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd_FPN import build_ssd

    net = build_ssd('test', 300, 4)    # initialize SSD
    # if args.cuda:
    #     net = torch.nn.DataParallel(net)
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
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))


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
