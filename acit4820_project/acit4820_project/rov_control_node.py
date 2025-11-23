#!/usr/bin/env python3
import numpy as np

# ros2 node related libraries
import rclpy
from rclpy.node import Node

# message libraries
from geometry_msgs.msg import Wrench, Vector3
from std_msgs.msg import Float64
from aruco_interfaces.msg import ArucoMarkers
from sensor_msgs.msg import PointCloud2, LaserScan
from geometry_msgs.msg import PoseStamped
from sensor_msgs_py.point_cloud2 import read_points

class RovControlNode(Node):
    def __init__(self):
        super().__init__('rov_control_node')

        num_thrusters = 6
        self.thruster_names = [num+1 for num in range(num_thrusters)]
        self.ns = 'bluerov2'

        # Publish to wrench_cmd from thruster manager node
        self.wrench_cmd_pub = self.create_publisher(Wrench, '/wrench_cmd', 10)
        
        # Publish errors for use with ros2 bag and to create csv's.
        self.rov_pos_error_pub   = self.create_publisher(Vector3, '/bluerov2/error/xy',    10)
        self.rov_depth_error_pub = self.create_publisher(Float64, '/bluerov2/error/depth', 10)
        self.rov_yaw_error_pub   = self.create_publisher(Float64, '/bluerov2/error/yaw',   10)

        # Subscribe to markers, odometry, and tf for ground truth (pseudo depth)
        self.aruco_marker_sub = self.create_subscription(ArucoMarkers, '/aruco_markers', self.marker_callback, 10)
        self.pc2_sub = self.create_subscription(PointCloud2, '/bluerov2/cloud', self.pc2_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/bluerov2/sonar', self.scan_callback, 10)
        
        self.ekf_yaw_sub = self.create_subscription(Float64, '/bluerov2/ekf_yaw', self.ekf_yaw_callback, 10) # rotation in euler (converted from quaternion)
        self.pose_sub = self.create_subscription(PoseStamped, '/bluerov2/pose', self.pose_callback, 10)

        self.aruco_ids = None
        self.aruco_poses = None
        self.attractive_angle = None
        self.ekf_yaw = None
        self.depth = None
        self.target_depth = None
        self.angles = None
        self.avg_front_dist = None
        self.points = None
        self.range_max = 19
        self.dist_threshold = 0.46 # arbitrary
        self.align_threshold = 0.1

        self.left_wall_id = 0
        self.right_wall_id = 1
        self.dock_align_id = 2
        self.right_id = 3 # relative to looking at dock entrance
        self.left_id = 4
        self.bottom_id = 5
        self.roof_id = 6
        self.back_id = 7
        
        # gains
        self.Kp_x = 0.9
        self.Kp_y = 0.5
        self.Kp_z = 150
        self.Kp_yaw = 1        

        self.entrance_ids = [self.left_wall_id, self.left_wall_id, self.dock_align_id]
        self.outer_wall_ids = [self.right_id, self.left_id, self.bottom_id, self.roof_id, self.back_id]

        # initial state
        self._STATES = ["SEARCHING", "DOCKING", "FINISHING"]
        self.state = "DOCKING"

        self.timer = self.create_timer(0.2, self.control)
        #self.timer = self.create_timer(0.2, self.maintain_depth)

    def marker_callback(self, msg: ArucoMarkers):
        self.aruco_ids = msg.marker_ids
        self.aruco_poses = msg.poses # list of poses
    
    def ekf_yaw_callback(self, msg: Float64):
        self.ekf_yaw = msg.data

    def pose_callback(self, msg: PoseStamped):
        self.depth = msg.pose.position.z
        self.target_depth = self.depth

    def scan_callback(self, msg: LaserScan):
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        self.range_max = msg.range_max - 1 # just to make sure it selects nearest point
        ranges = np.array(msg.ranges)
        len_ranges = len(ranges)
        self.angles = np.linspace(angle_min, angle_max, len_ranges) # create angle array corresponding to ranges

    def pc2_callback(self, msg: PointCloud2):
        points = []
        
        for p in read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            points.append((x, y, z))

        search_angle = 30 # as the lidars search range is 360, the relationship between the index of the desired angle and angle itself is equal.
        search_index = [int((len(self.angles)/2) - search_angle), int((len(self.angles)/2) + search_angle)]
        search_points = points[search_index[0]:search_index[1]]
        x_points = [i[0] for i in search_points]
        avg_front_dist = sum(x_points)/len(x_points)
        self.avg_front_dist = avg_front_dist
        self.points = points.copy()
        #self.get_logger().info(f"search points: {self.avg_front_dist}")
        #self.get_logger().info(f"search angle: {self.angles[search_index[0]]}, {self.angles[search_index[1]]}")

    def is_entrance(self, aruco_ids): # checks if rov is at entrance of docking station
        check_list = [i for i in self.entrance_ids if i in aruco_ids]
        
        if len(check_list) > 1:
            return True
        else:
            return False

    def search_for_dock(self):
        self.get_logger().info("Searching for dock...")

        if len(self.aruco_ids) > 0:
            pass

    def docking(self): # course docking with both entrace markers visible
        self.get_logger().info("Running docking function...")
        self.state = "DOCKING"
        #if distance <= self.dist_threshold:
        #   self.final_docking()
        pass

    def final_docking(self): # final docking procedure with only the inner marker, maybe supplement with data from lidar
        self.get_logger().info("Running final docking function...")
        self.state = "FINISHING"
        pass

    def control(self):
        """
        main code
        """
        if self.aruco_ids == None or self.aruco_poses == None or self.depth == None or self.target_depth == None:
            return
        self.get_logger().info(f"current_state: {self.state}")
        if self.state == "SEARCHING":
            depth_err = self.target_depth - self.depth
            self.get_logger().info(f"curr depth: {self.depth}")
            self.get_logger().info(f"target depth: {self.target_depth}")
            self.get_logger().info(f"depth err: {depth_err}")
            self.target_depth -= -1.0

            depth_msg = Wrench()
            depth_msg.force.z = float(depth_err * self.Kp_z)
            
            self.wrench_cmd_pub.publish(depth_msg) 
            
            if min(self.ranges) < self.range_max:
                pass

        if self.state == "FINISHING":
            if len(self.aruco_ids) == 1 and self.aruco_ids[0] == self.dock_align_id:
                self.get_logger().info(f"Finishing function running...")
                avg_x_err = self.aruco_poses[self.aruco_ids.index(self.dock_align_id)].position.z
                self.target_depth = self.depth - self.aruco_poses[self.aruco_ids.index(self.dock_align_id)].position.y 
                x_force = avg_x_err * self.Kp_x
                depth_err = self.target_depth - self.depth

                output_wrench = Wrench()
                output_wrench.force.x  = float(x_force)
                output_wrench.force.z = float(depth_err * self.Kp_z)
                
                self.get_logger().info(f"x error: {avg_x_err}")
                self.wrench_cmd_pub.publish(output_wrench)
                

            elif self.avg_front_dist <= self.dist_threshold:
                self.get_logger().info(f"Stopping...")
                self.wrench_cmd_pub.publish(Wrench())
                self.timer.cancel()
                

        if self.state == "DOCKING":
            """
            camera axis: z points forward, with x on the right side, y down
            """
            
            if len(self.aruco_ids) > 1 and self.is_entrance(self.aruco_ids):

                if self.right_wall_id in self.aruco_ids and self.left_wall_id in self.aruco_ids:
                    # self.get_logger().info(f"both ids present")
                    #self.get_logger().info(f"marker {self.aruco_ids[0]}: {self.aruco_poses[0].position.y}")
                    #self.get_logger().info(f"marker {self.aruco_ids[1]}: {self.aruco_poses[1].position.y}")
                    #self.get_logger().info(f"marker {self.aruco_ids[2]}: {self.aruco_poses[2].position.y}") 

                    #avg_z_err = 0 - ((self.aruco_poses[self.aruco_ids.index(self.right_wall_id)].position.y + self.aruco_poses[self.aruco_ids.index(self.left_wall_id)].position.y)/2) # y aruco frame, z for rov
                    self.target_depth = self.depth - ((self.aruco_poses[self.aruco_ids.index(self.right_wall_id)].position.y + self.aruco_poses[self.aruco_ids.index(self.left_wall_id)].position.y)/2)
                    avg_x_err = self.aruco_poses[self.aruco_ids.index(self.dock_align_id)].position.z # y aruco frame, z for rov
                    y_err = abs(self.aruco_poses[self.aruco_ids.index(self.right_wall_id)].position.x) - abs(self.aruco_poses[self.aruco_ids.index(self.left_wall_id)].position.x)
                    
                    #self.get_logger().info(f"x error: {avg_x_err}")
                    #self.get_logger().info(f"y error: {y_err}")
                    #self.get_logger().info(f"z error: {avg_z_err}")
                    
                    #z_force = avg_z_err * self.Kp_z
                    y_force = y_err * self.Kp_y
                    depth_err = self.target_depth - self.depth
                    #if abs(y_err) <= self.align_threshold and abs(avg_z_err) <= self.align_threshold:
                    if abs(y_err) <= self.align_threshold and abs(depth_err) <= self.align_threshold:
                        self.get_logger().info(f"moving forward...")
                        x_force = avg_x_err * self.Kp_x

                    else:
                        x_force = 0.0 # dont move forward until alignment is finished

                    output_wrench = Wrench()
                    output_wrench.force.x = float(x_force)
                    output_wrench.force.y = float(y_force)
                    output_wrench.force.z = float(depth_err * self.Kp_z)
                    
                    #output_wrench.force.z  = float(z_force)
                    #output_wrench.torque.x = float(-y_force)
                    
                    self.wrench_cmd_pub.publish(output_wrench)
            
            if len(self.aruco_ids) == 1 and self.aruco_ids[0] == self.dock_align_id:
                depth_err = self.target_depth - self.depth
                output_wrench = Wrench()
                output_wrench.force.z = float(depth_err * self.Kp_z)
                self.state = "FINISHING"
                self.wrench_cmd_pub.publish(output_wrench)
                return
        

def main(args=None):
    rclpy.init(args=args)
    node = RovControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.wrench_cmd_pub.publish(Wrench())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()