#!/usr/bin/env python3
"""
ROS2 node that broadcasts a 'drone_target' TF frame which moves linearly
back and forth between two xyz coordinates at a configurable speed.

Usage:
    ros2 run drone_pointcontrol drone_target_broadcaster

Or with parameter overrides:
    ros2 run drone_pointcontrol drone_target_broadcaster \
        --ros-args \
        -p start_x:=0.0 -p start_y:=0.0 -p start_z:=1.0 \
        -p end_x:=2.0   -p end_y:=1.0  -p end_z:=1.5  \
        -p speed:=0.5   -p parent_frame:=world
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class DroneTargetBroadcaster(Node):
    """
    Broadcasts a TF frame named 'drone_target' that oscillates linearly
    between two configurable (x, y, z) waypoints at a fixed metres-per-second
    speed.

    Parameters
    ----------
    parent_frame : str    – TF parent frame  (default: "world")
    start_x/y/z  : float – First waypoint    (default: 0, 0, 1)
    end_x/y/z    : float – Second waypoint   (default: 2, 0, 1)
    speed        : float – Travel speed m/s  (default: 0.5)
    timer_hz     : float – Broadcast rate Hz (default: 50)
    """

    def __init__(self):
        super().__init__("drone_target_broadcaster")

        # ── Declare & read parameters ──────────────────────────────────────
        self.declare_parameter("parent_frame", "camera_link")
        self.declare_parameter("start_x",  .75)
        self.declare_parameter("start_y",  0.0)
        self.declare_parameter("start_z",  0.0)
        self.declare_parameter("end_x",    1.5)
        self.declare_parameter("end_y",    0.0)
        self.declare_parameter("end_z",    0.0)
        self.declare_parameter("speed",    0.0)   # m/s
        self.declare_parameter("timer_hz", 50.0)  # broadcast frequency

        self.parent_frame = (
            self.get_parameter("parent_frame").get_parameter_value().string_value
        )
        self.p_start = self._get_vec("start")
        self.p_end   = self._get_vec("end")
        self.speed   = (
            self.get_parameter("speed").get_parameter_value().double_value
        )
        timer_hz = (
            self.get_parameter("timer_hz").get_parameter_value().double_value
        )

        # ── Pre-compute path geometry ──────────────────────────────────────
        diff = [self.p_end[i] - self.p_start[i] for i in range(3)]
        self.path_length = math.sqrt(sum(d * d for d in diff))

        if self.path_length < 1e-6:
            self.get_logger().warn(
                "start and end positions are identical – target will be stationary."
            )
            self.unit_vec = [0.0, 0.0, 0.0]
        else:
            self.unit_vec = [d / self.path_length for d in diff]

        # ── Internal state ─────────────────────────────────────────────────
        self._t          = 0.0   # current interpolation param in [0, path_length]
        self._direction  = 1     # +1 → towards end,  -1 → towards start
        self._dt         = 1.0 / timer_hz

        # ── TF broadcaster & timer ─────────────────────────────────────────
        self._tf_broadcaster = TransformBroadcaster(self)
        self._timer = self.create_timer(self._dt, self._timer_callback)

        self.get_logger().info(
            f"drone_target_broadcaster started\n"
            f"  parent : {self.parent_frame}\n"
            f"  start  : {self.p_start}\n"
            f"  end    : {self.p_end}\n"
            f"  length : {self.path_length:.3f} m\n"
            f"  speed  : {self.speed:.3f} m/s\n"
            f"  period : {2 * self.path_length / max(self.speed, 1e-6):.2f} s\n"
            f"  rate   : {timer_hz:.1f} Hz"
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_vec(self, prefix: str):
        """Read x/y/z parameters with the given prefix into a list."""
        return [
            self.get_parameter(f"{prefix}_{axis}").get_parameter_value().double_value
            for axis in ("x", "y", "z")
        ]

    def _interpolate(self, t: float):
        """Return the xyz position at distance t along the path."""
        return [self.p_start[i] + self.unit_vec[i] * t for i in range(3)]

    # ── Timer callback ─────────────────────────────────────────────────────

    def _timer_callback(self):
        # Advance the position along the path
        self._t += self._direction * self.speed * self._dt

        # Bounce at the endpoints
        if self._t >= self.path_length:
            self._t = self.path_length
            self._direction = -1
        elif self._t <= 0.0:
            self._t = 0.0
            self._direction = 1

        pos = self._interpolate(self._t)

        # Build and publish the TF transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp    = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.parent_frame
        tf_msg.child_frame_id  = "drone_target"

        tf_msg.transform.translation.x = pos[0]
        tf_msg.transform.translation.y = pos[1]
        tf_msg.transform.translation.z = pos[2]

        # Identity rotation – target orientation stays fixed
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0

        self._tf_broadcaster.sendTransform(tf_msg)


# ── Entry point ────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = DroneTargetBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()