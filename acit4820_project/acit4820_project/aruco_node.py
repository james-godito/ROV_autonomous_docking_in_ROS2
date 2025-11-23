#!/usr/bin/env python3
# Code taken and readapted from:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main
# and from turlab ntnu aruco pose estimation repository
# https://turlab.itk.ntnu.no/turlab/ros2-aruco-pose-estimation/-/blob/main/aruco_pose_estimation/scripts/aruco_node.py?ref_type=heads

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

from acit4820_project.pose_estimation import pose_estimation

class ArucoDetectionNode(Node):
    def __init__(self):
        super().__init__('aruco_node')
        self.get_logger().info("Starting Aruco Detection Node...")
        
        # Initialize topic subscriptions and publishers
        self.camera_sub      = self.create_subscription(Image,      '/bluerov2/image',       self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/bluerov2/camera_info', self.info_callback,  10)

        self.poses_pub   = self.create_publisher(PoseArray,    '/aruco_poses',   10)
        self.markers_pub = self.create_publisher(ArucoMarkers, '/aruco_markers', 10)
        self.image_pub   = self.create_publisher(Image,        '/aruco_image',   10)

        # Initialize variables
        self.info_msg      = None
        self.intrinsic_mat = None
        self.distortion    = None
        self.camera_frame  = ""

        # Aruco parameters
        self.aruco_dict_type  = cv2.aruco.DICT_5X5_1000
        self.aruco_dict       = cv2.aruco.Dictionary_get(self.aruco_dict_type)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        
        self.info_msg = info_msg
        
        # get the intrinsic matrix and distortion coefficients from the camera info
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

        self.get_logger().info("Camera info received.")
        self.get_logger().info("Intrinsic matrix: {}".format(self.intrinsic_mat))
        self.get_logger().info("Distortion coefficients: {}".format(self.distortion))
        self.get_logger().info("Camera frame: {}x{}".format(self.info_msg.width, self.info_msg.height))

        # Assume that camera parameters will remain the same...
        self.destroy_subscription(self.camera_info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            self.get_logger().warning("No camera info has been received!")
            return

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)

        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # call the pose estimation function
        frame, pose_array, markers = pose_estimation(frame=gray, 
                                                     depth_frame=None,
                                                     aruco_dict_type=self.aruco_dict,
                                                     matrix_coefficients=self.intrinsic_mat,
                                                     distortion_coefficients=self.distortion, pose_array=pose_array, markers=markers)

        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "8UC1"))

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
