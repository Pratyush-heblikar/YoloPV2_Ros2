#!/usr/bin/env python3
from typing import List
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import torch

from .utils12.utils import select_device, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, letterbox


def show_seg_result(img, result, palette=None, is_demo=False):
    if palette is None:
        palette = np.random.randint(0, 255, size=(3, 3))
    palette[0] = [0, 0, 0]
    palette[1] = [0, 255, 0]
    palette[2] = [255, 0, 0]
    palette = np.array(palette)

    color_seg = np.zeros((result[0].shape[0], result[0].shape[1], 3), dtype=np.uint8)
    color_seg[result[0] == 1] = [0, 255, 0]
    color_seg[result[1] == 1] = [255, 0, 0]

    color_mask = np.mean(color_seg, 2)
    alpha = 0.5  # Adjust the transparency level here (0.0 - fully transparent, 1.0 - fully opaque)

    img[color_mask != 0] = img[color_mask != 0] * (1 - alpha) + color_seg[color_mask != 0] * alpha
    img = img.astype(np.uint8)
    img = cv2.resize(img, (640, 640))

    return img


class YoloPv2(Node):
    def __init__(self, node_name):
        super().__init__('yolopv2_node')
        self.weights = '/home/pratyush/galaxis_ws/src/yolopv2_pkg/yolopv2_pkg/weights/yolopv2.pt'
        self.subscription = self.create_subscription(Image, '/galaxis/camera_raw', self.image_callback, 10)
        self.segmented_image_publisher = self.create_publisher(Image, 'segmented_image_topic', 10)
        self.bridge = CvBridge()
        self.device = select_device('0')
        self.model = None
        self.da_seg_mask = None
        self.ll_seg_mask = None

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.segmented_image = self.detect(cv_image)
        self.Segmented_image_show = show_seg_result(self.img0, (self.da_seg_mask, self.ll_seg_mask))
        try:
            self.segmented_image_msg = self.bridge.cv2_to_imgmsg(self.Segmented_image_show, encoding="bgr8")
            self.segmented_image_publisher.publish(self.segmented_image_msg)
        except CvBridgeError as e:
            print(e)

    def detect(self, cv_image):
        if self.model is None:
            self.model = torch.jit.load(self.weights)
            self.model = self.model.to(self.device)
            self.model.half()
            self.model.eval()

        device = select_device('0') 
        height, width, _ = cv_image.shape   
        roi = cv_image[height // 3:, :] 

        img0_resized = cv2.resize(cv_image, (320,320), interpolation=cv2.INTER_LINEAR)

        self.img0 = img0_resized
        img = torch.from_numpy(img0_resized).to(device)
        im = img.half() 
        im /= 255.0
        im = im.permute(2, 0, 1).unsqueeze(0)  

        with torch.no_grad():
            _ = self.model(torch.zeros(1, 3, 640,640).to(self.device).type_as(next(self.model.parameters())))
            [pred, anchor_grid], seg, ll = self.model(im)

        pred = split_for_trace_model(pred, anchor_grid)
        pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)

        self.da_seg_mask = driving_area_mask(seg)
        self.da_seg_mask = self.da_seg_mask.astype(np.uint8)
        self.da_seg_mask = cv2.resize(self.da_seg_mask, (320,320), interpolation=cv2.INTER_LINEAR)

        self.ll_seg_mask = lane_line_mask(ll)
        self.ll_seg_mask = self.ll_seg_mask.astype(np.uint8)
        self.ll_seg_mask = cv2.resize(self.ll_seg_mask, (320,320), interpolation=cv2.INTER_LINEAR)

        return self.img0


def main(args=None):
    rclpy.init(args=args)
    print("lane detection running")
    yolopv2_node = YoloPv2('yolopv2_node')
    rclpy.spin(yolopv2_node)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
