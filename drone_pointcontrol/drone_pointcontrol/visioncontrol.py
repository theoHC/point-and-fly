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

import rclpy
from rclpy import spin_once
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class FoundationPoseClient:
    def __init__(self, url):
        self.url = url.rstrip("/")

    async def estimate(
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

        async with aiohttp.ClientSession() as session:
            async with session.post(self.url + "/pose", data=data) as resp:
                resp.raise_for_status()
                result = await resp.json()

        return result

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
            qos
        )
        self.image_sub = Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw',
            qos
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

        pkg_dir = Path(__file__).resolve().parent

        self.map_path = pkg_dir / 'masks' / 'dronemask.png'

        self.create_timer(1.0, self.timer_callback)

    
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
    
    async def timer_callback(self):
        if self.depth_image is None or self.color_image is None:
            self.get_logger().info('Waiting for images...')
            return
        
        if self.processing_image:
            return
        
        self.processing_image = True

        task = asyncio.create_task(
            self.FPclient.estimate(
                rgb=self.color_image,
                depth=self.depth_image,
                intrinsics={"K": self.intrinsic_mat.tolist()},
                mask_path=self.map_path,
            )
        )

        result = await task

        self.get_logger().info(f'Pose estimation result: {result}')

        self.processing_image = False


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