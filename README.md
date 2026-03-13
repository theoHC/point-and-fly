# point-and-fly
This is a ROS2 package for controlling a DJI Tello drone using only a realsense camera.

## Installation
Clone the repoistory to the src folder of a ros2 workspace and build as a normal ros2 package.

In the top level of the workspace, create a virtual environment which retains your system packages. Install ultralytics, djitellopy, and mediapipe without using Pip's dependency resolution.

On a computer able to run FoundationPose, install and set up the FoundationPose server fork: https://github.com/theoHC/FoundationPoseFlaskServer

## Use
This package requires that your laptop be connected both to the wifi network of a Tello drone and have access to the http server running FoundationPose. Launch the FoundationPose server. Once it's running, use

`ros2 launch drone_pointcontrol point_and_fly.launch.xml`

to launch the system. Use the Xterm teleop keyboard to fly the drone to the center of the frame. If the system does not automatically start tracking, run `ros2 service call /reset-pose std_srvs/srv/Empty` to reset the tracked pose. Once the drone is being tracked, run `ros2 topic pub -1 /acquire_drone std_msgs/msg/Empty` to begin automatic calibration of the drone and hand control over to the system.

If the Tello drone does not automatically hold position, print several copies of "floor anti-razzle dazzle.svg" (so named due to its geometric similarity to mid-20th century 'razzle-dazzle' naval camoflage, but for aiding, rather than hindering, localization) and tape them to the floor underneath where you launch the drone.