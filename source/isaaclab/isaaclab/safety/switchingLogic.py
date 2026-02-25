from collections import deque
import statistics
import numpy as np

class SwitchingLogic:
    def __init__(self, parameters, joints_to_check:list[int] =[0, 1, 2, 3, 4, 5, 6] , window_size: int = 10, peak_limit: int =2, horizon:int = 500):
        self.params = parameters
        self.joints = joints_to_check
        self.window_size : int = window_size
        self.horizon:int =horizon
        self.traj = [[0.0]*horizon for _ in range(len(joints_to_check))]
        self.windows= self._calc_windows()
        self.unsafe_windows = {joint_num : 0 for joint_num in self.joints}
        self.peak_limit = peak_limit
        self.window_confidence_level = 0.669
        self.triggered = False 

    
    def _calc_windows(self)-> list[deque] :
        # return a list of deque objects, one for each joint
        return [deque(maxlen=self.window_size) for _ in range(len(self.joints))]
    
    def halfway_check(self, step:int=0, grasp_state:bool=True):
        #check that the subtask has been comepleted 
        if (step==(round(self.horizon/2))) and not grasp_state:
            if not self.triggered:
                print(f'I got halfway and hadnt done subtask, lets try again')
                self.triggered =True
            #then we are halfway through and lost, try again
            return True
        return False


    def check(self, uncertainties, step) -> bool:
        counts = 0
        # 1. Update windows
        for joint_num, unc in enumerate(uncertainties):
            # Check for bounds if uncertainties is smaller than self.joints 
            self.traj[joint_num][step]=unc.item()
            
            #print(f'Action comp {joint_num} uncertainty at step {step}: {unc.item()}')
            if step > self.params[joint_num]['window_size']:
                
                test_window = self.traj[joint_num][step-self.params[joint_num]['window_size']:step]
                peaks = sum(1 for x in test_window if x>= self.params[joint_num]['confidence_level'])
                #print(f'action {joint_num}, window: {test_window}')
               # print(f"Action comp {joint_num}, {unc.item()} limit {self.params[joint_num]['confidence_level']} manually found peaks : {peaks}")
                if peaks>=self.params[joint_num]['max_peaks']:
                    counts+=1
                                      
                        
        if counts >2:
            if not self.triggered:
                self.triggered = True
                print(f"Uncertaity in joint {joint_num} triggered")
                print("triggered safety condition")
            return True
        return False