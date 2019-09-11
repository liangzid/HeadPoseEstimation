import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib
import imutils
frame = None #初始化一个全局变量
yawx = None
pitchx = None
rollx = None
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='../hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='../mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',default='headpose_test',type=str)
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = vars(parser.parse_args())
    return args
def test_on_video_dlib_new_init(trigger):
    global frame ,yawx ,pitchx ,rollx
    args = parse_args()
    GESTURE = set(["yaw","pitch","row"])
    PRT_GES = {obj:0 for obj in GESTURE}
    cudnn.enabled = True
    batch_size = 1
    gpu = args["gpu_id"]
    snapshot_path = args["snapshot"]
    out_dir = 'output/video'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not args.get("video_path", False):
        print("[INFO] starting video stream...")
        video = VideoStream(src=0).start()
        time.sleep(1.0)
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        video = cv2.VideoCapture(args["video_path"])

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args["face_model"])

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # video = cv2.VideoCapture(video_path)
    # New cv2
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args["output_string"], 'w')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = None
    frame_num = 1
    while True:
        time.sleep(0.02)
        frame = video.read()
        frame = frame[1] if args.get("video_path", False) else frame
        (height, width) = frame.shape[:2]
        # print(height,width)
         # Define the codec and create VideoWriter object
        if out is None:
            out = cv2.VideoWriter('output/video/output-%s.avi' % args["output_string"], fourcc, args["fps"], (width, height))
        
        if frame is None:
            break
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 0.5:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)
                yaw, pitch, roll = model(img)         

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99


                yawx=yaw_predicted
                pitchx=pitch_predicted
                rollx=roll_predicted


                PRT_GES["yaw"] = int(yaw_predicted)
                PRT_GES["pitch"] = int(pitch_predicted)
                PRT_GES["row"] = int(roll_predicted)

                # 将姿态数据打印出来
                label = ",".join("{} : {}".format(gesture_str,gesture_data) for (gesture_str,gesture_data) in PRT_GES.items())               
                cv2.putText(frame, label,(10, height - 20),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,225,0) , 2)
                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                #Plot expanded bounding box
                # cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
        if out is not None:
            out.write(frame)
        # print('yaw=%f pitch=%f roll=%f\n' %(yawx, pitchx, rollx))
        frame_num += 1
        trigger.emit()
        # key = cv2.waitKey(1) & 0xFF
        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #   break
    # check to see if we need to release the video writer pointer
    if out is not None:
        out.release()
    # if we are not using a video file, stop the camera video stream
    if not args.get("video_path", False):
        video.stop()
    # otherwise, release the video file pointer
    else:
        video.release()
    # close any open windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    GESTURE = set(["yaw","pitch","row"])
    PRT_GES = {obj:0 for obj in GESTURE}
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args["gpu_id"]
    snapshot_path = args["snapshot"]
    out_dir = 'output/video'


    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # if not os.path.exists(args.video_path):
    #     sys.exit('Video does not exist')
    if not args.get("video_path", False):
        print("[INFO] starting video stream...")
        video = VideoStream(src=0).start()
        time.sleep(2.0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        video = cv2.VideoCapture(args["video_path"])

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args["face_model"])

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # video = cv2.VideoCapture(video_path)

    # New cv2
    # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args["output_string"], 'w')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    frame_num = 1
    out = None
    while frame_num <= args["n_frames"]:
        print(frame_num)
        frame = video.read()
        frame = frame[1] if args.get("video_path", False) else frame
        (height, width) = frame.shape[:2]
        print(height,width)
        # Define the codec and create VideoWriter object
        if out is None:
            out = cv2.VideoWriter('output/video/output-%s.avi' % args["output_string"], fourcc, args["fps"], (width, height))
        
        if frame is None:
            break
        # if frame[0] == False:
        #     break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 0.5:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)
                yaw, pitch, roll = model(img)         
                print(yaw)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)


                #QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ
                #print(yaw_predicted.shape) [1,66]
                #print(idx_tensor.shape)     [66]
                #print(roll_predicted.data[0]*idx_tensor)

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                PRT_GES["yaw"] = int(yaw_predicted)
                PRT_GES["pitch"] = int(pitch_predicted)
                PRT_GES["row"] = int(roll_predicted)

                # 将姿态数据打印出来
                label = ",".join("{} : {}".format(gesture_str,gesture_data) for (gesture_str,gesture_data) in PRT_GES.items())               
                cv2.putText(frame, label,(10, height - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,225,0) , 2)
                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                #Plot expanded bounding box
                # cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)

        if out is not None:
            out.write(frame)
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
          break
        frame_num += 1
    # check to see if we need to release the video writer pointer
    if out is not None:
        out.release()
    # if we are not using a video file, stop the camera video stream
    if not args.get("video_path", False):
        video.stop()

    # otherwise, release the video file pointer
    else:
        video.release()

    # close any open windows
    cv2.destroyAllWindows()
