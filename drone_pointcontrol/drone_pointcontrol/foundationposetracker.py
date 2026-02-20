from cv_bridge import CvBridge
from tf2_ros.transform_broadcaster import (
    TransformBroadcaster,
    TransformStamped,
)
from sensor_msgs.msg import CameraInfo, Image

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Vector3, Quaternion

from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np

import cv2

import json

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class FoundationPoseClient:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def estimate(
        self,
        rgb,
        depth,
        intrinsics,
        mask_path=None,
    ):

        files = {
            "rgb" : ("rgb.bin", rgb.tobytes(), "application/octet-stream"),
            "depth" : ("depth.bin", depth.tobytes(), "application/octet-stream"),
        }

        if mask_path is not None:
            files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

        data = {
            "intrinsics": json.dumps(intrinsics),
            "rgb_shape": json.dumps(list(rgb.shape)),
            "depth_shape": json.dumps(list(depth.shape))
        }

        r = requests.post(self.url + "/pose", files=files, data=data, timeout=60)

        if not r.ok:
            raise RuntimeError(f"FP server {r.status_code}: {r.text[:500]}")
        return r.json()

    def reset(self):
        r = requests.post(self.url + "/reset", timeout=10)
        if not r.ok:
            raise RuntimeError(f"FP server {r.status_code}: {r.text[:500]}")

def rs_to_numpy(frame):
    """Convert RealSense frame to NumPy array."""
    return np.asanyarray(frame.get_data())

def rs_get_K(intr):
    """Build 3x3 K from pyrealsense2 intrinsics as float64 (fx, fy, ppx, ppy)."""
    # pyrealsense2.intrinsics exposes fx, fy, ppx, ppy. These map to cx, cy as in literature.
    # Keep float64 to match trimesh and your dataset reader's K.
    K = np.array([[intr.fx, 0.0, intr.ppx],
                  [0.0,     intr.fy, intr.ppy],
                  [0.0,     0.0,     1.0    ]], dtype=np.float64)
    return K

class PointerController(Node):
    def __init__(self):
        super().__init__('pointer_controller')

        self.cbgroup = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,   # typical for images
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

        self.color_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            qos_profile=qos
        )
        self.image_sub = Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw',
            qos_profile=qos
        )

        self.approximate_time_synchronizer = ApproximateTimeSynchronizer(
            [self.depth_sub, self.image_sub],
            queue_size=10,
            slop=0.1
        )
        self.approximate_time_synchronizer.registerCallback(self.image_callback)

        self.intrinsic_mat = None

        self.depth_image = None
        self.color_image = None

        self.processing_image = False

        self.FPclient = FoundationPoseClient(url="http://lamb.mech.northwestern.edu:4242")
        self.FPclient.reset()

        self.declare_parameter("mask_path", "")

        self.tfb = TransformBroadcaster(self)

    
    def image_callback(self, depth_msg, image_msg):
        if self.intrinsic_mat is None:
            self.get_logger().warn('Camera intrinsics not received yet.')
            return
        
        try:
            depth_image_u16 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.depth_image = depth_image_u16.astype(np.float32) * 0.001
            self.color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Error converting images: {e}')
            return

        depth_mm = (self.depth_image * 1000.0).astype(np.uint16)

        cv2.imwrite("depth_raw_mm.png", depth_mm)

        output = self.FPclient.estimate(
            rgb=self.color_image,
            depth=self.depth_image,
            intrinsics={"K": self.intrinsic_mat.tolist()},
            mask_path=self.get_parameter("mask_path").get_parameter_value().string_value,
        )

        T = np.asarray(output["pose"], dtype=np.float64)     # (4,4) numpy array
        Rmat = T[:3, :3]                               # rotation matrix
        t = T[:3, 3]                                   # position / translation vector

        # Convert t and rmat to ros tf
        quat = R.from_matrix(Rmat).as_quat()  # (x, y, z, w) format

        t_ros = Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2]))
        quat_ros = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_color_optical_frame"
        transform.child_frame_id = "drone_pose"
        transform.transform.translation = t_ros
        transform.transform.rotation = quat_ros
        self.tfb.sendTransform(transform)

        self.get_logger().info(f"t = {t}")

    def info_callback(self, info_msg):
        """
        Get the camera intrisics from the cameraInfo topic.

        :param info_msg: image from the cv2 pipeline
        """
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))

def main(args=None):
    """User Node."""
    rclpy.init(args=args)
    node = PointerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()