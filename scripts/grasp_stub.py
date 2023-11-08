#!/usr/bin/env python3

import rospy
import sys
from bimanual_handover_msgs.srv import GraspExec, GraspTesterSrv, HandCloserSrv

global grasp_tester_srv, hand_closer_srv

def grasp_exec(req):
    close_grasp = hand_closer_srv(req.mode).finished
    rospy.loginfo("Grasp result: {}".format(close_grasp))
    if close_grasp:
        return grasp_tester_srv('y').success
    return False

def main():
    global grasp_tester_srv, hand_closer_srv
    rospy.init_node('grasp_stub')
    rospy.wait_for_service('hand_closer_srv')
    hand_closer_srv = rospy.ServiceProxy('hand_closer_srv', HandCloserSrv)
    rospy.wait_for_service('grasp_tester')
    grasp_tester_srv = rospy.ServiceProxy('grasp_tester', GraspTesterSrv)
    rospy.Service('grasp_exec_srv', GraspExec, grasp_exec)
    rospy.spin()

if __name__ == "__main__":
    main()
