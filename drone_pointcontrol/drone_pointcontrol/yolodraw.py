#!/usr/bin/env python3
"""
ROS2 YOLO draw node

Subscribes:
  - color Image
  - CameraInfo (cached, not synced)

Runs YOLO on each color frame, draws bounding boxes, and publishes:
  - ~/annotated_image  (sensor_msgs/Image)
  - ~/camera_info      (sensor_msgs/CameraInfo) - latest intrinsics, re-stamped
"""

from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data)

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Vector3
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloDrawNode(Node):
    def __init__(self):
        super().__init__("yolo_draw")

        self.declare_parameter("model_path", "")
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("imgsz", 720)
        self.declare_parameter("device", "")
        self.declare_parameter("mesh_path", "")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            raise RuntimeError(
                f"Parameter 'model_path' must be set to a YOLO model path (e.g. *.pt)"
            )

        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = self.get_parameter("device").get_parameter_value().string_value or None

        self.mesh_uri = "package://drone_pointcontrol/meshes/tello.stl"
    
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.last_info: Optional[CameraInfo] = None

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.color_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._color_cb, image_qos)

        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._info_cb, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "~/camera_info", qos_profile_sensor_data)

        self.pub_image = self.create_publisher(Image, "~/annotated_image", image_qos)
        self.marker_pub = self.create_publisher(Marker, "~/drone_marker", 10)
        self.create_timer(.1, self._marker_timer_cb)

        self.get_logger().info("YoloDrawNode ready.")

    def _marker_timer_cb(self):
        if not self.mesh_uri:
            return
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "drone_pose"
        marker.ns = "drone_mesh"
        marker.id = 0
        marker.type = Marker.MESH_RESOURCE
        marker.action = Marker.ADD
        marker.mesh_resource = self.mesh_uri
        marker.mesh_use_embedded_materials = True
        marker.scale = Vector3(x=0.003, y=0.003, z=0.003)
        marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        self.marker_pub.publish(marker)

    def _info_cb(self, msg: CameraInfo):
        self.last_info = msg
        self.info_pub.publish(msg)

    def _color_cb(self, color_msg: Image):
        self.get_logger().debug("Received color frame")
        if self.last_info is None:
            self.get_logger().warn("No CameraInfo received yet; skipping frame.")
            return

        bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

        results = self.model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        r = results[0]

        annotated = bgr.copy()
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for i, (box, conf) in enumerate(zip(xyxy, confs)):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = color_msg.header
        self.pub_image.publish(out_msg)


def main():
    rclpy.init()
    node = YoloDrawNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
