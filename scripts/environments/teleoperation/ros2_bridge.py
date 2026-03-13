#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from franka_msgs.action import Move
import socket
import json
import threading
import time

class ROS2Bridge(Node):
    def __init__(self, port=5005):
        super().__init__('isaaclab_ros2_bridge')
        self.get_logger().info(f'Starting ROS2 Bridge on port {port}...')

        # ROS2 Communication
        self._target_pub = self.create_publisher(PoseStamped, '/cartesian_target', 10)
        self._joint_target_pub = self.create_publisher(JointState, '/action_desired_joint', 10)
        self._action_client = ActionClient(self, Move, 'fr3_gripper/franka_gripper/move')
        self._real_pose_sub = self.create_subscription(PoseStamped, '/franka_robot_state_broadcaster/current_pose', self._real_pose_callback, 10)
        self._joint_sub = self.create_subscription(JointState, '/joint_states', self._joint_callback, 10)

        self.latest_real_pose = None
        self.latest_joints = None
        self.lock = threading.Lock()

        # UDP Communication
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('127.0.0.1', port))
        self.sock.settimeout(0.1)

        self.get_logger().info('Bridge is ready. Waiting for simulation data...')

    def _real_pose_callback(self, msg):
        with self.lock:
            self.latest_real_pose = {
                'pos': [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
                'quat': [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]
            }

    def _joint_callback(self, msg):
        arm_joints = [f'fr3_joint{i}' for i in range(1, 8)]
        pos, vel = [], []
        try:
            for name in arm_joints:
                idx = msg.name.index(name)
                pos.append(msg.position[idx])
                vel.append(msg.velocity[idx])
            with self.lock:
                self.latest_joints = {'pos': pos, 'vel': vel}
        except (ValueError, IndexError):
            pass

    def publish_cartesian(self, data):
        if len(data) < 7: return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fr3_link0'
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = data[0:3]
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = data[3:7]
        self._target_pub.publish(msg)
        self.get_logger().info(f"Published Cartesian Target: {data[0:3]}", throttle_duration_sec=0.5)

    def publish_joints(self, positions):
        if len(positions) < 7: return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base'
        msg.name = [f'fr3_joint{i}' for i in range(1, 8)]
        msg.position = [float(p) for p in positions[:7]]
        msg.velocity = [0.0] * 7
        msg.effort = [0.0] * 7
        self._joint_target_pub.publish(msg)
        # Throttled debug log (every 1s)
        self.get_logger().info(f"Published Joint Target: {[round(p, 3) for p in msg.position]}", throttle_duration_sec=0.2)

    def send_gripper(self, width):
        self.get_logger().info(f"Sending Gripper Goal: width={width}")
        goal_msg = Move.Goal()
        goal_msg.width = width
        goal_msg.speed = 0.2
        self._action_client.wait_for_server(timeout_sec=1.0)
        self._action_client.send_goal_async(goal_msg)

    def run(self):
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(1024)
                msg = json.loads(data.decode())
                
                if msg['type'] == 'cartesian':
                    # print(f"DEBUG: Received cartesian from {addr}")
                    self.publish_cartesian(msg['data'])
                elif msg['type'] == 'joint':
                    # print(f"DEBUG: Received joint from {addr}")
                    self.publish_joints(msg['data'])
                elif msg['type'] == 'gripper':
                    print(f"DEBUG: Received gripper command: {msg['width']}")
                    self.send_gripper(msg['width'])
                elif msg['type'] == 'query':
                    # print(f"DEBUG: Received query from {addr}")
                    # Send back latest state
                    with self.lock:
                        resp = {
                            'real_pose': self.latest_real_pose,
                            'joints': self.latest_joints
                        }
                    self.sock.sendto(json.dumps(resp).encode(), addr)
            except socket.timeout:
                pass
            except Exception as e:
                self.get_logger().error(f'Bridge error: {e}')
            
            rclpy.spin_once(self, timeout_sec=0.01)

def main():
    rclpy.init()
    bridge = ROS2Bridge()
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
