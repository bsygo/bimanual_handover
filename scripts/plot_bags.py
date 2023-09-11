#!/usr/bin/env python3

import rosbag
import rospkg
import glob
import matplotlib.pyplot as plt

def main():
    closing_joints = ['rh_FFJ2', 'rh_FFJ3', 'rh_MFJ2', 'rh_MFJ3', 'rh_RFJ2', 'rh_RFJ3', 'rh_LFJ2', 'rh_LFJ3', 'rh_THJ2']
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    path = pkg_path + "/data/"
    obs_joints_arr = []
    res_joints_arr = []
    for file_name in glob.iglob(f'{path}/closing_attempt_*.bag'):
        bag = rosbag.Bag(file_name)
        obs_joints = []
        res_joints = []
        for topic, msg, t in bag.read_messages():
            if topic == 'obs_joints':
                obs_joints.append(msg)
            elif topic == 'res_joints':
                res_joints.append(msg)
        bag.close()
        obs_joints_arr.append(obs_joints)
        res_joints_arr.append(res_joints)
    fig, axs = plt.subplots(3, 3)
    axs = axs.flatten()
    for joint in range(len(closing_joints)):
        for i in range(len(obs_joints_arr)):
            points = []
            for j in range(len(obs_joints_arr[i])):
                index = obs_joints_arr[i][j].name.index(closing_joints[joint])
                points.append(obs_joints_arr[i][j].position[index])
            axs[joint].plot(points)
        axs[joint].set_title(closing_joints[joint])
        #axs[joint].xlabel('Steps')
        #axs[joint].ylabel(clsoing_joints[joint])
    plt.show()

if __name__ == "__main__":
    main()
                

