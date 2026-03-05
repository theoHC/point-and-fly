#!/usr/bin/env python3
"""
ROS2 YOLO crop node

Subscribes:
  - color Image
  - aligned depth Image
  - CameraInfo (intrinsics for the aligned stream)

Synchronizes with ApproximateTimeSynchronizer, runs YOLO on color, crops color+depth,
and publishes cropped images + updated CameraInfo.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data)

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from message_filters import Subscriber, ApproximateTimeSynchronizer

from ultralytics import YOLO


@dataclass
class CropResult:
    x0: int
    y0: int
    w: int
    h: int


def compute_centered_crop_box(W: int, H: int, cx: int, cy: int, crop_w: int, crop_h: int) -> CropResult:
    crop_w = min(crop_w, W)
    crop_h = min(crop_h, H)

    x0 = max(0, cx - crop_w // 2)
    y0 = max(0, cy - crop_h // 2)

    if x0 + crop_w > W:
        x0 = W - crop_w
    if y0 + crop_h > H:
        y0 = H - crop_h

    return CropResult(x0=x0, y0=y0, w=crop_w, h=crop_h)


def crop_array(arr: np.ndarray, box: CropResult) -> np.ndarray:
    # arr can be HxW, HxWxC, etc.
    return arr[box.y0:box.y0 + box.h, box.x0:box.x0 + box.w].copy()


def camera_info_with_crop(src: CameraInfo, box: CropResult) -> CameraInfo:
    """
    Update intrinsics for a crop:
      fx, fy unchanged
      cx' = cx - x0
      cy' = cy - y0
    Also updates width/height and ROI.
    """
    out = CameraInfo()
    out.header = src.header

    out.height = box.h
    out.width = box.w
    out.distortion_model = src.distortion_model
    out.d = list(src.d)

    out.k = list(src.k)
    out.r = list(src.r)
    out.p = list(src.p)

    # K = [fx, 0, cx,
    #      0, fy, cy,
    #      0,  0,  1]
    out.k[2] = float(out.k[2] - box.x0)
    out.k[5] = float(out.k[5] - box.y0)

    # P = [fx, 0, cx, Tx,
    #      0, fy, cy, Ty,
    #      0,  0,  1,  0]
    out.p[2] = float(out.p[2] - box.x0)
    out.p[6] = float(out.p[6] - box.y0)

    # Optional ROI metadata
    out.binning_x = src.binning_x
    out.binning_y = src.binning_y
    out.roi.x_offset = box.x0
    out.roi.y_offset = box.y0
    out.roi.height = box.h
    out.roi.width = box.w
    out.roi.do_rectify = src.roi.do_rectify

    return out


class YoloCropNode(Node):
    def __init__(self):
        super().__init__("yolo_crop_node")

        # ---- Parameters ----
        self.declare_parameter("model_path", "")
        self.declare_parameter("crop_w", 300)
        self.declare_parameter("crop_h", 300)

        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("imgsz", 720)
        self.declare_parameter("device", "")  # e.g. "0" or "cpu"

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            raise RuntimeError(f"Parameter 'model' must be set to a YOLO model path (e.g. *.pt) - currently '{model_path}'")

        self.crop_w = int(self.get_parameter("crop_w").value)
        self.crop_h = int(self.get_parameter("crop_h").value)

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = self.get_parameter("device").get_parameter_value().string_value or None
        # ---- YOLO ----
        self.model = YOLO(model_path)
        self.names = self.model.names if hasattr(self.model, "names") else {}

        # ---- ROS IO ----
        self.bridge = CvBridge()
        self.last_info: Optional[CameraInfo] = None

        # CameraInfo subscriber (not synced; cache latest)
        self.info_sub = self.create_subscription(CameraInfo, "/camera/camera/color/camera_info", self._info_cb, qos_profile_sensor_data)

        # QoS for color (BEST_EFFORT)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Synced color+depth
        self.color_sub = Subscriber(self, Image, "/camera/camera/color/image_raw", qos_profile=image_qos)
        self.depth_sub = Subscriber(self, Image, "/camera/camera/aligned_depth_to_color/image_raw", qos_profile=image_qos)

        self.sync = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size=1, slop=0.1)
        self.sync.registerCallback(self._synced_cb)

        self.pub_color = self.create_publisher(Image, "/cropped/color/image_raw", image_qos)
        self.pub_depth = self.create_publisher(Image, "/cropped/aligned_depth_to_color/image_raw", image_qos)
        self.pub_info = self.create_publisher(CameraInfo, "/cropped/camera_info", image_qos)


        self.cx = 0
        self.cy = 0

    def depth_callback(self, msg: Image):
        self.get_logger().info(f"Depth message timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        pass

    def color_callback(self, msg: Image):
        self.get_logger().info(f"Color message timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}")
        pass

    def _info_cb(self, msg: CameraInfo):
        self.last_info = msg

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self.last_info is None:
            self.get_logger().warn("No CameraInfo received yet; skipping frame.")
            return

        # Convert color
        # Common encodings: "rgb8", "bgr8"
        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        bgr_for_yolo = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out_color_encoding = "rgb8"
        out_color_cv = rgb

        # Convert depth (preserve type/encoding)
        try:
            # "passthrough" keeps dtype (e.g. uint16 for 16UC1, float32 for 32FC1)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
            return

        H, W = out_color_cv.shape[:2]
        if depth.shape[0] != H or depth.shape[1] != W:
            self.get_logger().warn(
                f"Depth not same size as color (depth={depth.shape}, color={(H, W)}). "
                "These topics must be aligned and same resolution."
            )
            return

        # ---- YOLO inference ----
        results = self.model.predict(
            source=bgr_for_yolo,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        r = results[0]

        # Determine crop center
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            best_i = int(np.argmax(confs))
            x1, y1, x2, y2 = map(int, xyxy[best_i])
            self.cx = (x1 + x2) // 2
            self.cy = (y1 + y2) // 2
        else:
            return  # no detections, skip publishing

        box = compute_centered_crop_box(W, H, self.cx, self.cy, self.crop_w, self.crop_h)

        # Crop color+depth
        color_crop = crop_array(out_color_cv, box)
        depth_crop = crop_array(depth, box)

        # Update intrinsics
        info_in = self.last_info
        info_out = camera_info_with_crop(info_in, box)
        info_out.header = color_msg.header  # keep time consistent with published images

        # Publish cropped images
        out_color_msg = self.bridge.cv2_to_imgmsg(color_crop, encoding=out_color_encoding)
        out_color_msg.header = color_msg.header

        out_depth_msg = self.bridge.cv2_to_imgmsg(depth_crop, encoding=depth_msg.encoding)
        out_depth_msg.header = depth_msg.header

        self.pub_color.publish(out_color_msg)
        self.pub_depth.publish(out_depth_msg)
        self.pub_info.publish(info_out)


def main():
    rclpy.init()
    node = YoloCropNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()