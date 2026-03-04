#!/usr/bin/env python3
"""
tf_diff_on_source_update.py

Compute and print the relative transform between target_frame and source_frame
whenever the source_frame is updated in /tf or /tf_static.

Usage:
  ros2 run <your_pkg> tf_diff_on_source_update --ros-args \
    -p source_frame:=base_link -p target_frame:=map -p timeout_sec:=0.05
"""

from typing import Set

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

import tf2_ros


class TfDiffOnSourceUpdate(Node):
    def __init__(self):
        super().__init__("tf_diff_on_source_update")

        self.declare_parameter("source_frame", "world")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("timeout_sec", 0.1)  # how long to wait for lookup on each update
        self.declare_parameter("use_tf_static", True)

        self.source_frame = self.get_parameter("source_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.use_tf_static = bool(self.get_parameter("use_tf_static").value)

        # tf2 buffer + listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to /tf (dynamic transforms)
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self._on_tf_msg, 100)

        # Optionally subscribe to /tf_static (latched static transforms)
        self.tf_static_sub = None
        if self.use_tf_static:
            self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self._on_tf_msg, 10)

        # Track which frame_ids in each message correspond to the source frame
        self._source_ids: Set[str] = {self.source_frame, self._normalize(self.source_frame)}

        self.get_logger().info(
            f"Listening for updates to source_frame='{self.source_frame}'. "
            f"On update, computing transform: {self.target_frame} -> {self.source_frame}"
        )

    @staticmethod
    def _normalize(frame: str) -> str:
        # tf2 often uses frame ids without leading '/'
        return frame[1:] if frame.startswith("/") else frame

    def _msg_updates_source(self, msg: TFMessage) -> bool:
        for ts in msg.transforms:
            fid = self._normalize(ts.header.frame_id)
            cid = self._normalize(ts.child_frame_id)
            # Update may mention source as parent or child; both indicate the source is involved.
            if fid in self._source_ids or cid in self._source_ids:
                return True
        return False

    def _on_tf_msg(self, msg: TFMessage) -> None:
        if not self._msg_updates_source(msg):
            return

        # Lookup the relative transform target -> source at "latest available"
        try:
            # Passing Time() gives "latest"
            t: TransformStamped = self.tf_buffer.lookup_transform(
                self._normalize(self.target_frame),
                self._normalize(self.source_frame),
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
            )
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TimeoutException) as ex:
            self.get_logger().warn(
                f"TF lookup failed for {self.target_frame} -> {self.source_frame}: {type(ex).__name__}: {ex}"
            )
            return

        tr = t.transform.translation
        rot = t.transform.rotation

        # This is the "difference" transform between frames:
        # position = (x,y,z), orientation quaternion = (x,y,z,w)
        self.get_logger().info(
            f"[{t.header.stamp.sec}.{t.header.stamp.nanosec:09d}] "
            f"{t.header.frame_id} -> {t.child_frame_id} | "
            f"t=({tr.x:.4f}, {tr.y:.4f}, {tr.z:.4f}) "
            f"q=({rot.x:.4f}, {rot.y:.4f}, {rot.z:.4f}, {rot.w:.4f})"
        )


def main():
    rclpy.init()
    node = TfDiffOnSourceUpdate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()