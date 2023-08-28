#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import GraspExec, CCM

global ccm_srv

def grasp_exec(req):
    if req.mode == "ccm":
        return ccm_srv(req.placeholder).finished
    elif req.mode == "pca":
        return False
    else:
        rospy.logerr("Unknown grasping mode {}.".format(req.mode))
        return False

def main():
    global ccm_srv
    rospy.init_node('grasp_stub')
    rospy.wait_for_service('ccm')
    ccm_srv = rospy.ServiceProxy('ccm', CCM)
    rospy.Service('grasp_exec_srv', GraspExec, grasp_exec)
    rospy.spin()

if __name__ == "__main__":
    main()
