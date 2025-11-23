import rclpy
import math
from rclpy.node import Node

from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2

from tf2_ros import Buffer, TransformListener

class BluerovPoseNode(Node):
    def __init__(self):
        super().__init__('bluerov_pose_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.ekf_sub     = self.create_subscription(Odometry, '/odometry/filtered', self.ekf_sub_callback, 10)
        self.scan_sub    = self.create_subscription(LaserScan, '/bluerov2/sonar', self.scan_callback, 10)
        
        self.pub         = self.create_publisher(PoseStamped, '/bluerov2/pose',    10)
        self.ekf_yaw_pub = self.create_publisher(Float64,     '/bluerov2/ekf_yaw', 10)
        self.pc_pub      = self.create_publisher(PointCloud2, '/bluerov2/cloud',    10)

        self.ekf_msg = None
        self.timer = self.create_timer(0.1, self.timer_callback)

    def ekf_sub_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        ekf_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        self.ekf_msg = Float64()
        self.ekf_msg.data = float(ekf_yaw)
        
    def timer_callback(self):
        if self.ekf_msg == None:
            return
        
        try:
            t = self.tf_buffer.lookup_transform(
                'ocean', 'bluerov2', rclpy.time.Time()
            )

            pose = PoseStamped()
            pose.header = t.header
            pose.pose.position.x = round(t.transform.translation.x, 5)
            pose.pose.position.y = round(t.transform.translation.y, 5)
            pose.pose.position.z = round(t.transform.translation.z, 5)
            pose.pose.orientation.x = round(t.transform.rotation.x, 5)
            pose.pose.orientation.y = round(t.transform.rotation.y, 5)
            pose.pose.orientation.z = round(t.transform.rotation.z, 5)
            pose.pose.orientation.w = round(t.transform.rotation.w, 5)

            self.pub.publish(pose)
            self.ekf_yaw_pub.publish(self.ekf_msg)
        
        except Exception as e:
            self.get_logger().warn(str(e))

    def scan_callback(self, scan: LaserScan):
        points = []

        try:
            # Lookup transform once per scan (faster)
            tf = self.tf_buffer.lookup_transform(
                'bluerov2/base_link',
                'bluerov2/sonar',
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                r = scan.range_max + 1.0

            angle = scan.angle_min + i * scan.angle_increment

            # Point in lidar frame
            p_lidar = PointStamped()
            p_lidar.header = scan.header
            p_lidar.point.x = r * math.cos(angle)
            p_lidar.point.y = r * math.sin(angle)
            p_lidar.point.z = 0.0

            try:
                p_base = self.tf_buffer.transform(
                    p_lidar, 'bluerov2/base_link',
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                points.append([p_base.point.x, p_base.point.y, p_base.point.z])

            except Exception as e:
                self.get_logger().warn(f"Transform failed: {e}")

        # Create and publish cloud in base_link
        header = scan.header
        header.frame_id = 'bluerov2/base_link'
        pc2 = point_cloud2.create_cloud_xyz32(header, points)
        self.pc_pub.publish(pc2)


def main():
    rclpy.init()
    node = BluerovPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
