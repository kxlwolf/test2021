
import os
import time
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadFrame
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized



def init_net_model_config():

    # Initialize
    set_logging() #打印INFO级别以上的日志
    device = select_device('')
    print("device: ", device)
    half = device.type != 'cpu'  # half precision only supported on GPU(volta, turing)
    """pascal is not supported fp16"""
    #half = False
    print("half: ", half)
    weights = "./weights/seacoast/best_l.pt"

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    if half:
        model.half()  # to FP16
        
    # Get names and colors
    cls_names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(cls_names))]    
    
    img_cfg_sz = 416
    img_cfg_sz = check_img_size(img_cfg_sz, s=model.stride.max())  # check img_size

    return model, img_cfg_sz, cls_names, colors, device, half



def detect_frame(frame_read, model, img_cfg_sz, device, half):

    dataset = LoadFrame(frame_read, img_size=img_cfg_sz)
    print("dataset: ", dataset)

    # Run inference
    img = torch.zeros((1, 3, img_cfg_sz, img_cfg_sz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    for img, img_ori in dataset:
        print("img_ori.shape: ", img_ori.shape)
        print("img.shape: ", img.shape)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img, augment=False)[0]
    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    t2 = time_synchronized()

    print('Inference spend (%.3fs)' % (t2 - t1))
    
    return pred, img





def draw_frame(frame_read, img, pred, cls_names, colors, out_img_name, save_img=True):
    
    # Process detections for per image
    for i, det in enumerate(pred):
        print("det: ", det)
        s_log = ''
        if det is not None and len(det):
            # Rescale boxes from img_size to im_ori size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame_read.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s_log += '%g %ss, ' % (n, cls_names[int(c)])  # add to string
                
            # Write results
            for *xyxy, conf, cls in reversed(det):
                # Add bbox to image
                label = '%s %.2f' % (cls_names[int(cls)], conf)
                plot_one_box(xyxy, frame_read, label=label, color=colors[int(cls)])                

            print('s_log:  %s Done' % (s_log))

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(out_img_name, frame_read)


        
def infer_demo():

    model, img_cfg_sz, cls_names, colors, device, half = init_net_model_config()

    in_dir = "/data/kxlong/dataset/myOwn/seacoast/valid/images/"
    out_dir = "./inference/output/"

    img_list = os.listdir(in_dir)

    img_num = len(img_list)

    for img_self_name in img_list:
      if os.path.splitext(img_self_name)[1] in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".bmp"]:
        img_name = os.path.join(in_dir, img_self_name)
        out_img_name = out_dir + img_self_name
        print("\nimg_name: ", img_name)

        a = time.time()
        frame_read = cv2.imread(img_name)
        b = time.time()
        print("frame_read spend: {:.3f}s".format(b-a))

        pred, img = detect_frame(frame_read, model, img_cfg_sz, device, half)
        draw_frame(frame_read, img, pred, cls_names, colors, out_img_name)



if __name__ == '__main__':

    infer_demo()
    
    
    
    
    
    
    
    


