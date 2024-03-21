#!/usr/bin/env python3

import rospy
from moveit_commander import PlanningSceneInterface

def main():
    rospy.init_node("test")
    psi = PlanningSceneInterface(ns = "/", synchronous = True)
    #psi.disable_collision_detections(['rh_ffdistal', 'rh_thdistal'], ['rh_ffdistal', 'rh_thdistal'])
    #psi.enable_collision_detections(['rh_ffdistal', 'rh_thdistal'], ['rh_ffdistal', 'rh_thdistal'])
    print(psi.get_collision_matrix().entry_names)
    rospy.spin()

if __name__ == "__main__":
    main()
