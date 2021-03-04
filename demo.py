# -*- coding:utf8 -*-

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
from numpy import random
import cv2

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.yolo import Model


def detect():

    out, source, weights, view_img, save_img, save_txt, img_cfg_sz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')
    print("source: ", source)
    print("out: ", out)
    print("view_img: ", view_img)
    print("save_img: ", save_img)
    print("save_txt: ", save_txt)
    print("weights: ", weights)
    print("img_cfg_sz: ", img_cfg_sz)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize
    set_logging() #打印INFO级别以上的日志
    device = select_device(opt.device)
    print("device: ", device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print("half: ", half)

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    # print("type(model): ", type(model)) #是一个Model类
    # print("model: ", model)
    # print("model.yaml: ", model.yaml)
    # print("model.model: ", model.model)
    # print("model.save: ", model.save)
    
    if half:
        model.half()  # to FP16
        
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]    

    img_cfg_sz = check_img_size(img_cfg_sz, s=model.stride.max())  # check img_size
    
    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_writer = None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_cfg_sz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_cfg_sz)
    # print("dataset: ", dataset)
    # print("dataset[0]: ", dataset[0]) #no __getitem__

    # Run inference
    t0 = time.time()
    
    img = torch.zeros((1, 3, img_cfg_sz, img_cfg_sz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for path, img, img_ori, vid_cap in dataset:
        print("img_ori.shape: ", img_ori.shape)
        print("img.shape: ", img.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        #print("1pred.shape: ", pred.shape)        
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, merge=opt.merge, classes=opt.classes, agnostic=opt.agnostic_nms)
        #print("2pred[0].shape: ", pred[0].shape)        
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, img_ori)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s_log, im_ori = path[i], '%g: ' % i, img_ori[i].copy()
            else:
                p, s_log, im_ori = path, '', img_ori

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '') + '.txt'
            # print("save_path: ", save_path)
            # print("txt_path: ", txt_path)
            
            s_log += '%gx%g ' % img.shape[2:]  # print string

            print("det: ", det)
            if det is not None and len(det):
                # Rescale boxes from img_size to im_ori size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im_ori.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s_log += '%g %ss, ' % (n, names[int(c)])  # add to string #检出种类的个数
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf) #string
                    plot_one_box(xyxy, im_ori, label=label, color=colors[int(cls)])                

                    if save_txt:  # Write to file
                        gn = torch.tensor(im_ori.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

            # Print time (inference + NMS)
            print('s_log:  %sDone. (%.3fs)\n' % (s_log, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im_ori)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im_ori)
                else:
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im_ori)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))
        if platform.system() == 'Darwin' and not opt.update:  # MacOS
            os.system('open ' + save_path)
    
    t3 = time.time()
    print('All Done. (%.3fs)' % (t3 - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--update', action='store_true', help='update all models')

    parser.add_argument('--weights', nargs='+', type=str, default='./weights/seacoast/best_l.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--source', type=str, default='/data/kxlong/dataset/myOwn/seacoast/test/', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    
    parser.add_argument('--augment', action='store_true', default= False, help='augmented inference')
    parser.add_argument('--save-img', action='store_true', default= True, help='save img')
    parser.add_argument('--save-txt', action='store_true', default= False, help='save results to *.txt')
    parser.add_argument('--view-img', action='store_true', help='display results')
    
    parser.add_argument('--merge', action='store_true', help='merge box while do nms') #做nms时是否选择合并框模式
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3') #可以手动选择只输出某一类的检测结果
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS') #class-agnostic nms，同一个bbox会做不同class的nms
    
    opt = parser.parse_args()
    print("opt: ", opt)

    # a = torch.cuda.is_available()
    # b = torch.cuda.device_count()
    # c = torch.cuda.current_device()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

