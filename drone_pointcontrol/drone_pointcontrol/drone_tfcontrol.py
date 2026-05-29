#!/usr/bin/env python3
"""
tf_diff_on_source_update.py

Compute and print the relative transform between target_frame and drone_frame
whenever the drone_frame is updated in /tf or /tf_static.

Usage:
  ros2 run <your_pkg> tf_diff_on_source_update --ros-args \
    -p drone_frame:=base_link -p target_frame:=map -p timeout_sec:=0.05
"""

from typing import Set

from djitellopy import Tello
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from geometry_msgs.msg import TransformStamped, Twist
from tf2_msgs.msg import TFMessage
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Empty

import tf2_ros


import logging
logging.getLogger('djitellopy').setLevel(logging.WARNING)

class TfDiffOnSourceUpdate(Node):
    def __init__(self):
        super().__init__("position_control")

        self.declare_parameter("drone_frame", "drone_pose")
        self.declare_parameter("target_frame", "target_pose")
        self.declare_parameter("timeout_sec", 0.2)  # how long to wait for lookup on each update
        self.declare_parameter("use_tf_static", True)
        self.declare_parameter("use_drone", True)  # if True, node will shutdown if no updates received for a while

        self.declare_parameter("calib_stabilize_time_sec", 2.0)  # how long to wait between acquisition and calibration

        self.declare_parameter("kp_pos", 60.0)
        self.declare_parameter("kp_vert_mul", 2.5)  # Multiplier on vertical error to compensate for typically weaker vertical control response
        self.declare_parameter("kp_yaw", 5.0)

        self.use_drone = self.get_parameter("use_drone").get_parameter_value().bool_value

        self.drone_frame = self.get_parameter("drone_frame").get_parameter_value().string_value
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)
        self.use_tf_static = bool(self.get_parameter("use_tf_static").value)

        self.tfcbgroup = MutuallyExclusiveCallbackGroup()

        self.othercbgroup = MutuallyExclusiveCallbackGroup()

        # tf2 buffer + listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to /tf (dynamic transforms)
        self.tf_sub = self.create_subscription(TFMessage, "/tf", self._on_tf_msg, 100, callback_group=self.tfcbgroup)

        # Optionally subscribe to /tf_static (latched static transforms)
        self.tf_static_sub = None
        if self.use_tf_static:
            self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self._on_tf_msg, 10, callback_group=self.tfcbgroup)


        # Track which frame_ids in each message correspond to the source frame
        self._source_ids: Set[str] = {self.drone_frame, self._normalize(self.drone_frame)}

        self.get_logger().info(
            f"Listening for updates to drone_frame='{self.drone_frame}'. "
            f"On update, computing transform: {self.target_frame} -> {self.drone_frame}"
        )

        self.last_update_time = self.get_clock().now()
        self.drone_acquired = False

        self.tello = Tello()
        if self.use_drone:
            self.tello.connect()
            self.tello.takeoff()
    
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.calibrated = False

        self.declare_parameter("control_yaw_deg", 0.0)
        self.control_frame = self._normalize(self.drone_frame) + "_control"

        self.timer = self.create_timer(0.1, self.timer_callback, callback_group=self.othercbgroup)

        self.create_service(
            Empty,
            '~/reset_calib',
            self.reset_calib_cb)

        self.drone_forward = 0.0
        self.drone_right = 0.0
        self.drone_up = 0.0
        self.drone_yaw = 0.0
        
        self.create_service(
            Empty,
            '/man_acquire',
            self.manual_acquire_cb)
        
        self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10,
            callback_group=self.tfcbgroup)

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

        self.broadcast_control_frame()

        if self.drone_acquired and self.calibrated:
            try:
                t_latest: TransformStamped = self.tf_buffer.lookup_transform(
                    self.control_frame,
                    self._normalize(self.target_frame),
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.timeout_sec),
                )
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                    tf2_ros.TimeoutException) as ex:
                self.get_logger().warn(
                    f"TF lookup failed for {self.drone_frame} -> {self.target_frame}: {type(ex).__name__}: {ex}"
                )
                return
            
            self.do_position_control(t_latest)

    def do_position_control(self, t_latest: TransformStamped):
        kp_pos = self.get_parameter("kp_pos").get_parameter_value().double_value
        kp_yaw = self.get_parameter("kp_yaw").get_parameter_value().double_value

        forward_error = t_latest.transform.translation.x
        sideways_error = t_latest.transform.translation.z
        vertical_error = t_latest.transform.translation.y
        yaw_error = Rotation.from_quat([
            t_latest.transform.rotation.x,
            t_latest.transform.rotation.y,
            t_latest.transform.rotation.z,
            t_latest.transform.rotation.w
        ]).as_euler('xyz', degrees=True)[2]  # Yaw in degrees

        self.get_logger().info(f"Errors | Forward: {forward_error:.3f}, Sideways: {sideways_error:.3f}, Vertical: {vertical_error:.3f}, Yaw: {yaw_error:.3f}")
        forward_speed = kp_pos * forward_error
        sideways_speed = -1 * kp_pos * sideways_error
        vertical_speed = kp_pos * vertical_error \
            * self.get_parameter("kp_vert_mul").get_parameter_value().double_value
        yaw_speed = kp_yaw * yaw_error

        # self.get_logger().info(f"Control | Forward: {forward_speed:.1f}, Sideways: {sideways_speed:.1f}, Vertical: {vertical_speed:.1f}, Yaw: {yaw_speed:.1f}")

        self.set_rc_joystick(forward_speed, sideways_speed, vertical_speed, 0)

    def cmd_vel_callback(self, msg: Twist):
        forward = 50 if msg.linear.x > 0 else -50 if msg.linear.x < 0 else 0
        right = -50 if msg.linear.y > 0 else 50 if msg.linear.y < 0 else 0
        up = 50 if msg.linear.z > 0 else -50 if msg.linear.z < 0 else 0
        yaw = 50 if msg.angular.z > 0 else -50 if msg.angular.z < 0 else 0

        self.set_rc_joystick(forward, right, up, yaw)

    def broadcast_control_frame(self):
        yaw_deg = self.get_parameter("control_yaw_deg").get_parameter_value().double_value
        q = Rotation.from_euler('y', yaw_deg, degrees=True).as_quat()  # [x, y, z, w]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self._normalize(self.drone_frame)
        t.child_frame_id = self.control_frame
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        if self.use_drone:
            self.tello.send_rc_control(round(self.drone_right),
                                    round(self.drone_forward),
                                    round(self.drone_up),
                                    round(self.drone_yaw))

    def set_rc_joystick(self, forward: float, right: float, up: float, yaw: float):
        self.drone_forward = clamp(forward, -50, 50)
        self.drone_right = -1 * clamp(right, -50, 50)
        self.drone_up = clamp(up, -50, 50)
        self.drone_yaw = clamp(yaw, -50, 50)

    def manual_acquire_cb(self, _, response):
        self.get_logger().info("Manually acquiring drone!")
        self.drone_acquired = True
        self.calibrated = True
        return response

    def reset_calib_cb(self, _, response):
        self.calibrated = False
        self.drone_acquired = False
        self.set_rc_joystick(0, 0, 0, 0)
        return response

    def shutdown(self):
        if self.use_drone:
            self.tello.land()

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

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