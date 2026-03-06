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

from djitellopy import Tello
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster

import tf2_ros


class TfDiffOnSourceUpdate(Node):
    def __init__(self):
        super().__init__("tf_diff_on_source_update")

        self.declare_parameter("source_frame", "world")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("timeout_sec", 0.1)  # how long to wait for lookup on each update
        self.declare_parameter("use_tf_static", True)
        self.declare_parameter("use_drone", True)  # if True, node will shutdown if no updates received for a while
        self.use_drone = self.get_parameter("use_drone").get_parameter_value().bool_value

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

        self.last_update_time = self.get_clock().now()
        self.drone_acquired = False

        self.tello = Tello()
        if self.get_parameter("use_drone").get_parameter_value().bool_value:
            self.tello.connect()
            self.tello.takeoff()
        
        self.tf_broadcaster = TransformBroadcaster(self)

        self.calibrated = False
        self.x_forward = 1.0
        self.x_sideways = 0.0

        self.timer = self.create_timer(0.1, self.timer_callback)

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

        self.last_update_time = self.get_clock().now()
        if not self.drone_acquired:
            self.drone_acquired = True
            self.get_logger().info("Drone acquired.")

    def check_drone_forward(self):
        #Figure out which of x or z is forward for the drone by moving it and then checking which tf change is greater.
        if not self.use_drone:
            self.get_logger().info("Calibration hit.")
            return

        if not self.drone_acquired:
            self.get_logger().warn("Cannot check drone forward direction because drone not acquired yet.")
            return

        if self.get_parameter("use_drone").get_parameter_value().bool_value:
            t_start: TransformStamped = self.tf_buffer.lookup_transform(
                self._normalize(self.target_frame),
                self._normalize(self.source_frame),
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
            )
            t_start.child_frame_id = "drone_calib_start"
            t_start.header.stamp = self.get_clock().now().to_msg()
            self.tello.set_speed(20)
            self.tello.move_forward(20)

            t_end: TransformStamped = self.tf_buffer.lookup_transform(
                self._normalize(self.target_frame),
                "drone_calib_start",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
            )

            dx = t_end.transform.translation.x
            dz = t_end.transform.translation.z

            if abs(dx) > abs(dz):
                self.x_foward = dx / abs(dx)
                self.x_sideways = 0
            else:
                self.x_forward = 0
                self.x_sideways = dz / abs(dz)
            
            self.get_logger().info(f"Calibrated drone forward direction: x_forward={self.x_forward}, x_sideways={self.x_sideways}")



    def timer_callback(self):
        timeout_duration = 0.5 if self.drone_acquired else 5.0

        if not self.drone_acquired and self.use_drone:
            self.tello.send_rc_control(0, 0, 0, 0)

        if self.drone_acquired and not self.calibrated:
            self.get_logger().info("Checking drone forward direction...")
            self.check_drone_forward()
            self.calibrated = True

        if (self.get_clock().now() - self.last_update_time > rclpy.duration.Duration(seconds=timeout_duration)
            and self.get_parameter("use_drone").get_parameter_value().bool_value):
            self.get_logger().info("No updates received for a while, shutting down.")
            self.shutdown()
            self.destroy_node()
            rclpy.shutdown()
        pass

    def shutdown(self):
        if self.use_drone:
            self.tello.land()


def main():
    rclpy.init()
    node = TfDiffOnSourceUpdate()
    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception in main loop: {e}")
        node.get_logger().error("Stack trace:", exc_info=True)
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()