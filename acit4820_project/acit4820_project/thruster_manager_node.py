#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from std_msgs.msg import Float64
import yaml
import os
import numpy as np
from dataclasses import dataclass
from typing import List
from ament_index_python.packages import get_package_share_directory

@dataclass
class ThrusterLink:
    name: str
    position: np.ndarray
    direction: np.ndarray
    max_thrust: float
    min_thrust: float

    def as_column(self):
        f = self.direction
        tau = np.cross(self.position, f)
        return np.hstack((f, tau)).reshape((6, 1))


class ThrusterManagerPy:
    def __init__(self, thrusters: List[ThrusterLink]):
        self.thrusters = thrusters
        self.n = len(thrusters)
        self.update_tam()

    def update_tam(self):
        cols = [t.as_column() for t in self.thrusters]
        self.A = np.hstack(cols)  # 6 x n
        self.pinvA = np.linalg.pinv(self.A)

    def solve_wrench(self, wrench):
        wrench = np.asarray(wrench, dtype=float).reshape((6, 1))
        try:
            x = self.pinvA @ wrench
        except Exception as e:
            print(f"[ThrusterManagerPy] Matrix multiply failed: {e}")
            raise

        x = x.flatten()
        for i, t in enumerate(self.thrusters):
            x[i] = np.clip(x[i], t.min_thrust, t.max_thrust)
        return x


# ------------------------------
# ROS2 Node
# ------------------------------

class ThrusterManagerNode(Node):
    def __init__(self):
        super().__init__('thruster_manager_node')
        self.get_logger().info("starting manager...")
        # Declare and read config_file parameter
        self.declare_parameter('config_file', 'thruster.yaml')
        pkg_share = get_package_share_directory('acit4820_project')
        file_name = self.get_parameter('config_file').value
        config_path = os.path.join(pkg_share, 'config', file_name)

        # Load YAML
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            self.get_logger().error(f'Failed to read YAML file: {e}')
            raise

        thrusters = []
        for t in data.get('thrusters', []):
            thrusters.append(
                ThrusterLink(
                    name=t['name'],
                    position=np.array(t['position'], dtype=float),
                    direction=np.array(t['direction'], dtype=float),
                    min_thrust=float(t.get('min_thrust', -40.0)),
                    max_thrust=float(t.get('max_thrust', 40.0))
                )
            )

        if not thrusters:
            self.get_logger().error('No thrusters found in YAML file.')
            raise RuntimeError('Empty thruster list.')

        self.manager = ThrusterManagerPy(thrusters)
        #self.get_logger().info(f"TAM: {self.manager.A}")
        self.thruster_names = [t.name for t in thrusters]
        self.ns = 'bluerov2'

        # Create one publisher per thruster
        self.thruster_pubs = {
            name: self.create_publisher(Float64, f'/{self.ns}/cmd_thruster{name}', 10)
            for name in self.thruster_names
        }

        # Subscribe to wrench commands
        self.sub = self.create_subscription(
            Wrench,
            '/wrench_cmd',
            self.wrench_callback,
            10
        )

        #self.get_logger().info(f"ThrusterManagerNode started with {len(thrusters)} thrusters: {', '.join(str(self.thruster_names))}")

    def wrench_callback(self, msg: Wrench):
        wrench_rcvd = np.array([
            msg.force.x,
            msg.force.y,
            msg.force.z,
            msg.torque.x,
            msg.torque.y,
            msg.torque.z
        ])
        #self.get_logger().info(f"Rcvd wrench: {wrench_rcvd}")
        try:
            forces = self.manager.solve_wrench(wrench_rcvd)
            #self.get_logger().info(f"forces: {forces}")
            
        except Exception as e:
            self.get_logger().error(f"Error in solve_wrench: {e}")
            return
        
        for name, value in zip(self.thruster_names, forces):
            out = Float64()
            out.data = round(value, 2)
            self.thruster_pubs[name].publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ThrusterManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
