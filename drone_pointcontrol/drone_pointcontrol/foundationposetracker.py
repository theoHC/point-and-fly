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

import json

import requests

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)

from std_srvs.srv import Empty
from std_msgs.msg import Empty as EmptyMsg

class FoundationPoseClient:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def estimate(
        self,
        rgb,
        depth,
        intrinsics,
        mask_path=None,
        mask_img=None,
        rescore=False
    ):

        files = {
            "rgb" : ("rgb.bin", rgb.tobytes(), "application/octet-stream"),
            "depth" : ("depth.bin", depth.tobytes(), "application/octet-stream"),
        }

        if mask_img is not None:
            files["masknp"] = ("mask.bin", mask_img.tobytes(), "application/octet-stream")

        if mask_path is not None:
            files["mask"] = ("mask.png", open(mask_path, "rb"), "image/png")

        data = {
            "intrinsics": json.dumps(intrinsics),
            "rgb_shape": json.dumps(list(rgb.shape)),
            "depth_shape": json.dumps(list(depth.shape)),
            "rescore": json.dumps(rescore)
        }
        if mask_img is not None:
            data["mask_shape"] = json.dumps(list(mask_img.shape))

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

class FPTracker(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        self.cbgroup = MutuallyExclusiveCallbackGroup()

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

        self.info_sub = Subscriber(
            self,
            CameraInfo,
            '/camera/camera/color/camera_info',
            qos_profile=image_qos,
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            qos_profile=image_qos
        )

        self.color_sub = Subscriber(
            self,
            Image,
            '/camera/camera/color/image_raw',
            qos_profile=image_qos
        )

        self.approximate_time_synchronizer = ApproximateTimeSynchronizer(
            [self.depth_sub, self.color_sub, self.info_sub],
            queue_size=1,
            slop=0.1
        )
        self.approximate_time_synchronizer.registerCallback(self.image_callback)

        self.acquired_publisher = self.create_publisher(EmptyMsg, '/drone_acquired', 10)

        self.intrinsic_mat = None

        self.depth_image = None
        self.color_image = None

        self.processing_image = False

        self.declare_parameter("server_url", "")
        server_url = self.get_parameter("server_url").get_parameter_value().string_value
        self.get_logger().info(f"FP server URL: '{server_url}'")

        self.FPclient = FoundationPoseClient(url=server_url)
        self.FPclient.reset()

        self.declare_parameter("mask_path", "")
        self.declare_parameter("use_mask_img", False)

        self.create_service(
            Empty,
            'reset_pose',
            self.reset_callback,
            callback_group=self.cbgroup)

        self.tfb = TransformBroadcaster(self)

        self.declare_parameter("stabilizing_frames", 10)
        self.stabilizing_frames = int(self.get_parameter("stabilizing_frames").get_parameter_value().integer_value)
        self.acquired = False
        self.score = -1.0

    
    def image_callback(self, depth_msg, image_msg, info_msg):
        self.info_callback(info_msg)
        
        try:
            depth_image_u16 = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            self.depth_image = depth_image_u16.astype(np.float32) * 0.001
            self.color_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Error converting images: {e}')
            return

        h, w = self.color_image.shape[:2]
        mask = np.full((h, w), 255, dtype=np.uint8)

        rescore = False
        if not self.acquired and self.stabilizing_frames <= 0:
            rescore = True

        if self.get_parameter("use_mask_img").get_parameter_value().bool_value:
            output = self.FPclient.estimate(
                rgb=self.color_image,
                depth=self.depth_image,
                intrinsics={"K": self.intrinsic_mat.tolist()},
                mask_path=self.get_parameter("mask_path").get_parameter_value().string_value,
                rescore=rescore
            )
        else:
            output = self.FPclient.estimate(
                rgb=self.color_image,
                depth=self.depth_image,
                intrinsics={"K": self.intrinsic_mat.tolist()},
                mask_img=mask,
                rescore=rescore
            )

        T = np.asarray(output["pose"], dtype=np.float64)     # (4,4) numpy array
        Rmat = T[:3, :3]                               # rotation matrix
        t = T[:3, 3]                                   # position / translation vector

        # Convert t and rmat to ros tf
        quat = R.from_matrix(Rmat).as_quat()  # (x, y, z, w) format

        outscore = float(output["score"])
        if outscore != -1.0:
            self.score = outscore

            if self.stabilizing_frames < 0 and not self.acquired:
                self.get_logger().info(f"Pose acquired with score {self.score:.2f}. Publishing transform and acquiring drone.")
                self.acquired_publisher.publish(Empty())
                self.acquired = True

        if self.stabilizing_frames > 0:
            if self.score >= 110:
                self.stabilizing_frames -= 1
                self.get_logger().info(f"High score {self.score:.2f} - stabilizing for {self.stabilizing_frames} more frames.")
            else:
                self.get_logger().warn(f"Rejecting frame due to low score: {self.score:.2f}")
                self.FPclient.reset()
                self.stabilizing_frames = int(self.get_parameter("stabilizing_frames").get_parameter_value().integer_value)
                self.score = -1.0  # clear stale score so next frame re-registers

        t_ros = Vector3(x=float(t[0]), y=float(t[1]), z=float(t[2]))
        quat_ros = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_color_optical_frame"
        transform.child_frame_id = "drone_pose"
        transform.transform.translation = t_ros
        transform.transform.rotation = quat_ros
        self.tfb.sendTransform(transform)

    def info_callback(self, info_msg):
        """
        Get the camera intrisics from the cameraInfo topic.

        :param info_msg: image from the cv2 pipeline
        """
        self.info_msg = info_msg
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
    
    def reset_callback(self, _, response):
        self.FPclient.reset()
        return response

def main(args=None):
    """User Node."""
    rclpy.init(args=args)
    node = FPTracker()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()