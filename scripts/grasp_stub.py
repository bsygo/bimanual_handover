#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import GraspExec, CCM, GraspTesterSrv

global ccm_srv, grasp_tester_srv

def grasp_exec(req):
    close_grasp = ccm_srv(req.placeholder).finished
    rospy.loginfo("Closing grasp result: {}".format(close_grasp))
    if close_grasp:
        return grasp_tester_srv('placeholder').success
    return False

def main():
    global ccm_srv, grasp_tester_srv
    rospy.init_node('grasp_stub')
    rospy.wait_for_service('ccm')
    ccm_srv = rospy.ServiceProxy('ccm', CCM)
    rospy.wait_for_service('grasp_tester')
    grasp_tester_srv = rospy.ServiceProxy('grasp_tester', GraspTesterSrv)
    rospy.Service('grasp_exec_srv', GraspExec, grasp_exec)
    rospy.spin()

if __name__ == "__main__":
    main()
