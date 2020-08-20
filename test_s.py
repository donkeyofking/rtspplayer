#base
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
#numpy
import numpy as np
from numpy import random
#opencv
import cv2
#rtsp player
from ffpyplayer.player import MediaPlayer
#pytorch
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
#local python files
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.datasets import letterbox


def show_in_one(images, show_size=(570, 660), blank_size=2, window_name="cam6"):
    small_h, small_w = images[0].shape[:2]
    # print(small_h,small_w)
    column = int(show_size[1] / (small_w + blank_size))
    row = int(show_size[0] / (small_h + blank_size))
    shape = [show_size[0], show_size[1]]
    for i in range(2, len(images[0].shape)):
        shape.append(images[0].shape[i])

    merge_img = np.zeros(tuple(shape), images[0].dtype)

    max_count = len(images)
    count = 0
    for i in range(row):
        if count >= max_count:
            break
        for j in range(column):
            if count < max_count:
                im = images[count]
                t_h_start = i * (small_h + blank_size)
                t_w_start = j * (small_w + blank_size)
                t_h_end = t_h_start + im.shape[0]
                t_w_end = t_w_start + im.shape[1]
                merge_img[t_h_start:t_h_end, t_w_start:t_w_end] = im
                count = count + 1
            else:
                break
    if count < max_count:
        print("total images count： %s" % (max_count - count))
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    # x, y = merge_img.shape[0:2]
    # merge_img = cv2.resize(merge_img, (y*2, x*2))
    cv2.imshow(window_name, merge_img)


def load_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--classes', default=[0 ,1 ,2 ,3], type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    # use cpu or gpu
    device = select_device('')
    # device = select_device('cpu')
    half = device.type == 'cpu' 
    print(device)
    # load yolov5x
    model = torch.load("weights/yolov5s.pt", map_location=device)['model'].float().fuse().eval()
    if half:
        model.half()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    return model, device, half, names, colors ,opt

def img2tensor(image0,half):
    img = letterbox(image0)[0]
    img = np.transpose(img,(2,0,1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    return img

image_hhh = 180
image_www = 320

if __name__ == '__main__':

    model, device, half, names, colors, opt = load_model()
     
     # play rtsp live
    ff_opts = { 'codec':'h264_cuvid', 'fflags':'no_buffer', 'flags':'low_delay', 'strict':'experimental', 'rtsp_transport':'tcp'}
    player = MediaPlayer('rtsp://192.168.8.101/cam6', ff_opts=ff_opts)
    val = ''
    while val != 'eof':
        t0 = time.time()

        frame, val = player.get_frame()
        # print('get frame. (%.3fs)' % (time.time() - t0))

        if val != 'eof' and frame is not None:
            t0 = time.time()

            img, t = frame
            arr = img.to_memoryview()[0]
            # print(type(arr))
            image = np.array(arr)
            # print(image.shape)
            image = np.reshape(image,(image_hhh,image_www*6,3))
            image = image[:,:,[2,1,0]]
            image0 = image[:,0:image_www,:]
            image1 = image[:,image_www:image_www*2,:]
            image2 = image[:,image_www*2:image_www*3,:]
            image3 = image[:,image_www*3:image_www*4,:]
            image4 = image[:,image_www*4:image_www*5,:]
            image5 = image[:,image_www*5:image_www*6,:]
            # cv2.imshow('image0',image0)
            # cv2.imshow('image1',image1)
            # cv2.imshow('image2',image2)
            # cv2.imshow('image3',image3)
            # cv2.imshow('image4',image4)
            # cv2.imshow('image5',image5)
            # # Run inference
            # image0 = cv2.imread("inference/images/bus.jpg")
            # image1 = cv2.imread("inference/images/bus.jpg")
            # image2 = cv2.imread("inference/images/bus.jpg")
            # image3 = cv2.imread("inference/images/bus.jpg")
            # image4 = cv2.imread("inference/images/bus.jpg")
            # image5 = cv2.imread("inference/images/bus.jpg")
            # imagetest = cv2.imread("inference/images/bus.jpg")
            # print(image0.shape)
            # print(image2.shape)
            # print(image3.shape)

            img0_a = np.stack([image0,image1,image2,image3,image4,image5],axis=0)
            print('split. (%.3fs)' % (time.time() - t0))
            t0 = time.time()
            # img0_a = np.stack([image0,image1,image2,image3,image4,image5],axis=0)

            img0 = img2tensor(image0,half)
            img1 = img2tensor(image1,half)
            img2 = img2tensor(image2,half)
            img3 = img2tensor(image3,half)
            img4 = img2tensor(image4,half)
            img5 = img2tensor(image5,half)

            img_a = torch.stack([img0,img1,img2,img3,img4,img5],dim=0)
            
            # print(img_a.shape)
            pred = model(img_a, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Process detections
            images = []
            for i, det in enumerate(pred):  # detections per image
                im0 = img0_a[i].copy()
                img = img_a[i]
                # im0 = image0
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()
                    # Write results
                    for *xyxy, conf, cls in det:
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                images.append(im0)
                # cv2.imwrite( filename = str(i)+'.jpg‘ , img=im0)
            print('pred. (%.3fs)' % (time.time() - t0))
            t0 = time.time()
            show_in_one(images,show_size=(image_hhh* 3 + 6, image_www*2 + 4))
            print('merge. (%.3fs)' % (time.time() - t0))
            t0 = time.time()
            if cv2.waitKey(1) == ord('q'):  # q to quit
                exit()
