#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import GraspExec, CCM, GraspTesterSrv

global ccm_srv

def grasp_exec(req):
    if ccm_srv(req.placeholder).finished:
        return grasp_tester_srv('placeholder').success
    return False

def main():
    global ccm_srv
    rospy.init_node('grasp_stub')
    rospy.wait_for_service('handover/ccm')
    ccm_srv = rospy.ServiceProxy('handover/ccm', CCM)
    rospy.wait_for_service('handover/grasp_tester')
    grasp_tester_srv = rospy.ServiceProxy('handover/grasp_tester', GraspTesterSrv)
    rospy.Service('handover/grasp_exec_srv', GraspExec, grasp_exec)
    rospy.spin()

if __name__ == "__main__":
    main()
