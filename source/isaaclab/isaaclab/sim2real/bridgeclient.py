import socket
import json
import time
import numpy as np

from franka_msgs.action import Move

class BridgeClient:
    def __init__(self, port=5005):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = ('127.0.0.1', port)
        self.sock.settimeout(0.5)
        self._last_print_time = 0


    def publish_command(self, actions):
        """Publish 7-DOF Cartesian target: [x, y, z, qx, qy, qz, qw]"""
        if hasattr(actions, 'cpu'):
            act_list = actions.cpu().numpy().flatten().tolist()
        else:
            act_list = np.array(actions).flatten().tolist()
        
        if len(act_list) < 7: return
        
        payload = {'type': 'cartesian', 'data': act_list[:7]}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        
        # Throttled debug print (every 1 second)
        curr_time = time.time()
        if curr_time - self._last_print_time > 1.0:
       #     print(f"DEBUG: Sent Cartesian to Bridge: {act_list}")
            self._last_print_time = curr_time

    def publish_joints(self, positions):
        """Publish 7-DOF Joint positions"""
        if hasattr(positions, 'cpu'):
            pos_list = positions.cpu().numpy().flatten().tolist()
        else:
            pos_list = np.array(positions).flatten().tolist()
        
        if len(pos_list) < 7: return
        
        payload = {'type': 'joint', 'data': pos_list[:7]}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        
        # Throttled debug print (every 1s)
        curr_time = time.time()
        if curr_time - self._last_print_time > 1.0:
            print(f"DEBUG: Sent Joint Target to Bridge: {[round(p, 3) for p in pos_list[:7]]}")
            self._last_print_time = curr_time

    def open_gripper(self):
        print("DEBUG: Sending Open Gripper to Bridge")
        payload = {'type': 'gripper', 'width': 0.08}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

    def close_gripper(self):
        print("DEBUG: Sending Close Gripper to Bridge")
        payload = {'type': 'gripper', 'width': 0.00}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)

    def get_robot_state(self):
        payload = {'type': 'query'}
        self.sock.sendto(json.dumps(payload).encode(), self.addr)
        try:
            data, _ = self.sock.recvfrom(2048)
            return json.loads(data.decode())
        except socket.timeout:
            return None

    def init_real(self, sim_pose):
        print("[INFO] Initializing real robot synchronization via bridge...")
        self.publish_command(sim_pose)
        target_pos = np.array(sim_pose[:3])
        
        while True:
            state = self.get_robot_state()
            if state and state.get('real_pose'):
                real_pos = np.array(state['real_pose']['pos'])
                dist = np.linalg.norm(target_pos - real_pos)
                print(f"[INFO] Distance to target: {dist:.4f}m")
                if dist < 0.05:
                    print("[INFO] Real robot synchronized with simulation!")
                    break
            else:
                print("[INFO] Waiting for bridge on 127.0.0.1:5005...")
            time.sleep(0.01)

