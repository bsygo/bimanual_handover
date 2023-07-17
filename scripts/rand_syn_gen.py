#!/usr/bin/env python3

import rospy
import numpy as np
import random
import bimanual_handover.syn_grasp_gen as sgg

global syn_grasp_gen

def main():
    success = False
    count = 0
    while (not success) and count < 10:
        try:
            # (-1.1 -> 0.8), (-0.5 -> 0.4)
            alphas = np.array([0, 0, 0, 0, 0])#[0.5, -0.53, -0.55])#[-1.02628934, -0.53937113, -0.62126708])#np.array([random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5), random.gauss(0, 0.5)])
            syn_grasp_gen.move_joint_config(alphas)
            success = True
        except Exception as e:
            print(e)
            count += 1
            print(count)
    print('fin')
    return

if __name__ == "__main__":
    global syn_grasp_gen
    stop = False
    rospy.init_node('rand_syn_gen_node')
    syn_grasp_gen = sgg.SynGraspGen(display_state = True)
    while not stop:
        main()
        result = input("Please press 'y' if you want to generate a new pose.\n")
        if not result == 'y':
            stop = True
