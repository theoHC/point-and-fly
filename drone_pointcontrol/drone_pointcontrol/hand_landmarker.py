#!/usr/bin/env python3
"""ROS2 MediaPipe Hand Landmarker node.

Subscribes:
  /camera/camera/color/image_raw                       (sensor_msgs/Image)
  /camera/camera/aligned_depth_to_color/image_raw      (sensor_msgs/Image)
  /camera/camera/color/camera_info                     (sensor_msgs/CameraInfo)

Publishes:
  ~/annotated_image (sensor_msgs/Image) - BGR annotated visualization
"""

from __future__ import annotations

from pathlib import Path
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Int32
from tf2_ros.transform_broadcaster import TransformBroadcaster, TransformStamped
from geometry_msgs.msg import Vector3, Quaternion, Point
from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MARGIN = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

mp_hands_connections = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles


def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands_connections.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

        height, width, _ = annotated_image.shape
        x_coordinates = [lm.x for lm in hand_landmarks]
        y_coordinates = [lm.y for lm in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        label = handedness[0].category_name if handedness and handedness[0] else "Hand"
        cv2.putText(
            annotated_image,
            label,
            (max(text_x, 0), max(text_y, 0)),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def deproject_pixel_to_point(
    depth_msg: Image,
    info: CameraInfo,
    bridge: CvBridge,
    x: float,
    y: float,
) -> tuple[float, float, float] | None:
    """Return the (X, Y, Z) camera-frame position of pixel (x, y).

    Args:
        depth_msg: Aligned depth Image message.
        info:      CameraInfo for the color/depth stream (must be aligned).
        bridge:    CvBridge instance for decoding the depth image.
        x, y:      Pixel coordinates in the color image (floats accepted).

    Returns:
        (X, Y, Z) in metres relative to the camera origin, or None if the
        depth sample at that pixel is zero / out of bounds.
    """
    depth_img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    px, py = int(round(x)), int(round(y))
    h, w = depth_img.shape[:2]
    if not (0 <= px < w and 0 <= py < h):
        return None

    depth_raw = depth_img[py, px]
    if depth_raw == 0:
        return None

    # RealSense depth is in millimetres; convert to metres.
    z = float(depth_raw) / 1000.0

    fx = info.k[0]
    fy = info.k[4]
    cx = info.k[2]
    cy = info.k[5]

    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    return (X, Y, z)


def ensure_model(model_path: Path, url: str = MODEL_URL) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"[info] Model not found at '{model_path}'. Downloading from:\n  {url}")
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    model_path.write_bytes(data)
    print(f"[info] Saved model to: {model_path}")
    return model_path


class HandLandmarkerNode(Node):
    def __init__(self):
        super().__init__('hand_landmarker')

        self.declare_parameter('model_path', 'hand_landmarker.task')
        self.declare_parameter('num_hands', 2)
        self.declare_parameter('min_hand_detection_confidence', 0.5)
        self.declare_parameter('min_hand_presence_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)
        self.declare_parameter('tf_frame_id', 'drone_target')
        self.declare_parameter('distance_multiplier', 3.0)
        self.declare_parameter('finger_point_distance', .07)
        self.declare_parameter('post_hoc_annotation', False)

        model_path = ensure_model(
            Path(self.get_parameter('model_path').get_parameter_value().string_value)
        )

        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.get_parameter('num_hands').get_parameter_value().integer_value,
            min_hand_detection_confidence=self.get_parameter(
                'min_hand_detection_confidence').get_parameter_value().double_value,
            min_hand_presence_confidence=self.get_parameter(
                'min_hand_presence_confidence').get_parameter_value().double_value,
            min_tracking_confidence=self.get_parameter(
                'min_tracking_confidence').get_parameter_value().double_value,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.bridge = CvBridge()
        self.last_info: CameraInfo | None = None
        self.tf_broadcaster = TransformBroadcaster(self)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self._info_cb, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, '~/camera_info', 10)

        self.color_sub = Subscriber(
            self, Image, '/camera/camera/color/image_raw', qos_profile=image_qos)
        self.depth_sub = Subscriber(
            self, Image, '/camera/camera/aligned_depth_to_color/image_raw', qos_profile=image_qos)

        queue_size = 1
        if self.get_parameter('post_hoc_annotation').get_parameter_value().bool_value:
            # If post_hoc_annotation is enabled, we want to synchronize the latest depth and color frames with the latest CameraInfo, even if their timestamps don't match closely. This allows the TF to be published even when the hand detection is too slow to keep up with the camera frame rate.
            queue_size = 10

        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=queue_size, slop=0.1)
        self.sync.registerCallback(self._synced_cb)

        self.pub_annotated = self.create_publisher(Image, '~/annotated_image', image_qos)
        self.pub_markers = self.create_publisher(MarkerArray, '~/hand_markers', 10)

        self.viz_state: int = -1
        self.create_subscription(Int32, '/yolo_draw/viz_state', self._viz_state_cb, 10)

        self.get_logger().info('HandLandmarkerNode ready.')

        self.cur_target_x = 0.0
        self.cur_target_y = 0.0
        self.cur_target_z = 1.0

        self.create_timer(0.1, self._publish_target_tf)

    def _viz_state_cb(self, msg: Int32):
        self.viz_state = msg.data

    def _info_cb(self, msg: CameraInfo):
        self.last_info = msg
        self.info_pub.publish(msg)

    def _synced_cb(self, color_msg: Image, depth_msg: Image):
        if self.last_info is None:
            self.get_logger().warn('No CameraInfo received yet; skipping frame.')
            return

        frame_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Pad to square along the longest dimension (bottom) so MediaPipe receives
        # a square image and the NORM_RECT projection doesn't need IMAGE_DIMENSIONS.
        h, w = frame_rgb.shape[:2]
        pad_size = max(h, w)
        if h != w:
            frame_rgb = np.pad(frame_rgb, ((0, pad_size - h), (0, pad_size - w), (0, 0)))

        stamp = color_msg.header.stamp
        timestamp_ms = int(stamp.sec * 1000 + stamp.nanosec / 1e6)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        markerarr = MarkerArray()
        if len(result.hand_landmarks) > 0:
            first_hand = result.hand_landmarks[0]
            index_mcp = first_hand[5]   # Index as in index finger; mcp as in metacarpophalangeal joint (the "knuckle" where the finger meets the palm)
            index_tip = first_hand[8]   # Tip as in fingertip

            nh, nw, _ = frame_rgb.shape
            mcp_3d = deproject_pixel_to_point(
                depth_msg, self.last_info, self.bridge, index_mcp.x * nw, index_mcp.y * nh)
            tip_3d = deproject_pixel_to_point(
                depth_msg, self.last_info, self.bridge, index_tip.x * nw, index_tip.y * nh)
            

            distance = 0
            #calculate the distance between mcp and tip
            if mcp_3d is not None and tip_3d is not None:
                dx = tip_3d[0] - mcp_3d[0]
                dy = tip_3d[1] - mcp_3d[1]
                dz = tip_3d[2] - mcp_3d[2]
                distance = (dx**2 + dy**2 + dz**2) ** 0.5
            
            if mcp_3d is not None and tip_3d is not None and distance > self.get_parameter('finger_point_distance').get_parameter_value().double_value:
                # Visualize with an arrow
                line_marker = Marker()
                line_marker.header.stamp = self.get_clock().now().to_msg()
                line_marker.header.frame_id = 'camera_color_optical_frame'
                line_marker.ns = 'hand_landmarks'
                line_marker.id = 2
                line_marker.type = Marker.ARROW
                line_marker.action = Marker.ADD
                line_marker.points = [Point(x=mcp_3d[0], y=mcp_3d[1], z=mcp_3d[2]), Point(x=tip_3d[0], y=tip_3d[1], z=tip_3d[2])]
                line_marker.scale = Vector3(x=0.02, y=0.04, z=0.06)  # shaft diameter, head diameter, head length
                line_marker.scale.x = 0.02
                line_marker.color.r = 1.0
                line_marker.color.g = 0.0
                line_marker.color.b = 0.0
                line_marker.color.a = 1.0
                markerarr.markers.append(line_marker)

                # Compute the offset from mcp to tip
                distance_multiplier = self.get_parameter('distance_multiplier').get_parameter_value().double_value
                offset_x = (tip_3d[0] - mcp_3d[0]) * distance_multiplier
                offset_y = (tip_3d[1] - mcp_3d[1]) * distance_multiplier
                offset_z = (tip_3d[2] - mcp_3d[2]) * distance_multiplier

                target_x = mcp_3d[0] + offset_x
                target_y = mcp_3d[1] + offset_y
                target_z = mcp_3d[2] + offset_z

                # Add a small green sphere at the target point
                sphere_marker = Marker()
                sphere_marker.header.stamp = self.get_clock().now().to_msg()
                sphere_marker.header.frame_id = 'camera_color_optical_frame'
                sphere_marker.ns = 'hand_landmarks'
                sphere_marker.id = 3
                sphere_marker.type = Marker.SPHERE
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position = Point(x=target_x, y=target_y, z=target_z)
                sphere_marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
                sphere_marker.color.r = 0.0
                sphere_marker.color.g = 1.0
                sphere_marker.color.b = 0.0
                sphere_marker.color.a = 1.0
                markerarr.markers.append(sphere_marker)

                self.cur_target_x = target_x
                self.cur_target_y = target_y
                self.cur_target_z = target_z
            else:
                line_marker = Marker()
                line_marker.header.stamp = self.get_clock().now().to_msg()
                line_marker.header.frame_id = 'camera_color_optical_frame'
                line_marker.ns = 'hand_landmarks'
                line_marker.id = 2
                line_marker.type = Marker.ARROW
                line_marker.action = Marker.DELETE
                markerarr.markers.append(line_marker)

        else:
            sphere_marker = Marker()
            sphere_marker.header.stamp = self.get_clock().now().to_msg()
            sphere_marker.header.frame_id = 'camera_color_optical_frame'
            sphere_marker.ns = 'hand_landmarks'
            sphere_marker.id = 3
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.pose.position = Point(x=self.cur_target_x, y=self.cur_target_y, z=self.cur_target_z)
            sphere_marker.scale = Vector3(x=0.05, y=0.05, z=0.05)
            sphere_marker.color.r = 0.0
            sphere_marker.color.g = 1.0
            sphere_marker.color.b = 0.0
            sphere_marker.color.a = 1.0
            markerarr.markers.append(sphere_marker)

        if self.viz_state in (-1, 4):
            for marker in markerarr.markers:
                marker.action = Marker.ADD
        else:
            for marker in markerarr.markers:
                marker.action = Marker.DELETE

        self.pub_markers.publish(markerarr)

        annotated_rgb = draw_landmarks_on_image(frame_rgb, result)[:h, :w]
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

        if self.viz_state in (-1, 3):
            out_msg = self.bridge.cv2_to_imgmsg(annotated_bgr, encoding='bgr8')
        else:
            out_msg = color_msg
        out_msg.header = color_msg.header
        out_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_annotated.publish(out_msg)
        self.last_info.header.stamp = self.get_clock().now().to_msg()

    def _publish_target_tf(self):
        if self.cur_target_z < 0 or self.get_parameter('post_hoc_annotation').get_parameter_value().bool_value:
            return
        target_tf = TransformStamped()
        target_tf.header.stamp = self.get_clock().now().to_msg()
        target_tf.header.frame_id = 'camera_color_optical_frame'
        target_tf.child_frame_id = self.get_parameter('tf_frame_id').get_parameter_value().string_value
        target_tf.transform.translation.x = self.cur_target_x
        target_tf.transform.translation.y = self.cur_target_y
        target_tf.transform.translation.z = self.cur_target_z
        self.tf_broadcaster.sendTransform(target_tf)

    def destroy_node(self):
        self.detector.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
