import math
from pathlib import Path
from cv_bridge import CvBridge
from tf2_ros.transform_broadcaster import (
    TransformBroadcaster,
    TransformStamped,
)
from sensor_msgs.msg import CameraInfo, Image

from tf2_geometry_msgs import do_transform_pose

from tf2_ros import (
    ConnectivityException, ExtrapolationException, LookupException
)

from scipy.spatial.transform import Rotation as R

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform

from message_filters import ApproximateTimeSynchronizer, Subscriber

from scipy.spatial.transform import Rotation as R

import numpy as np

import json

import requests

import rclpy
from rclpy import spin_once
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
        return r.json()

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
        self.camera_frame_id = None

        self.depth_image = None
        self.color_image = None

        self.processing_image = False

        self.FPclient = FoundationPoseClient(url="http://lamb.mech.northwestern.edu:4242")

        self.declare_parameter("mask_path", "")

    
    def image_callback(self, depth_msg, image_msg):
        if self.intrinsic_mat is None:
            self.get_logger().warn('Camera intrinsics not received yet.')
            return
        
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting images: {e}')
            return
        
        output = self.FPclient.estimate(
            rgb=self.color_image,
            depth=self.depth_image,
            intrinsics={"K": self.intrinsic_mat.tolist()},
            mask_path=self.get_parameter("mask_path").get_parameter_value().string_value,
        )

        T = np.asarray(output["pose"], dtype=np.float64)     # (4,4) numpy array
        Rmat = T[:3, :3]                               # rotation matrix
        t = T[:3, 3]                                   # position / translation vector

        # eul_xyz = R.from_matrix(Rmat).as_euler("xyz", degrees=True)  # roll, pitch, yaw in radians

        self.get_logger().info(f"t = {t}")

    def info_callback(self, info_msg):
        """
        Get the camera intrisics from the cameraInfo topic.

        :param info_msg: image from the cv2 pipeline
        """
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

def main(args=None):
    """User Node."""
    rclpy.init(args=args)
    node = PointerController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()