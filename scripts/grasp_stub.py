#!/usr/bin/env python3

import rospy
import sys
from bimanual_handover_msgs.srv import GraspExec, CCM, GraspTesterSrv, ModelMoverSrv

global ccm_srv, grasp_tester_srv, model_mover_srv

def grasp_exec(req):
    if req.mode == "ccm":
        close_grasp = ccm_srv(req.mode).finished
        rospy.loginfo("Closing grasp result: {}".format(close_grasp))
        if close_grasp:
            return grasp_tester_srv('placeholder').success
        return False
    elif req.mode == "pca":
        close_grasp = model_mover_srv(req.mode).finished
        rospy.loginfo("PCA grasp result: {}".format(close_grasp))
        if close_grasp:
            return grasp_tester_srv('placeholder').success
        return False
    else:
        rospy.logerr("Unknown grasping mode {}.".format(req.mode))
        return False

def main():
    global ccm_srv, grasp_tester_srv, model_mover_srv
    rospy.init_node('grasp_stub')
    if len(sys.argv) < 2:
        rospy.logerr("Missing argument for grasp_stub.")
        return
    elif sys.argv[1] == 'ccm':
        rospy.wait_for_service('ccm')
        ccm_srv = rospy.ServiceProxy('ccm', CCM)
    elif sys.argv[1] == 'pca':
        rospy.wait_for_service('model_mover_srv')
        model_mover_srv = rospy.ServiceProxy('model_mover_srv', ModelMoverSrv)
    else:
        rospy.logerr("Unknown argument {} for grasp_stub.".format(sys.argv[1]))
        return
    rospy.wait_for_service('grasp_tester')
    grasp_tester_srv = rospy.ServiceProxy('grasp_tester', GraspTesterSrv)
    rospy.Service('grasp_exec_srv', GraspExec, grasp_exec)
    rospy.spin()

if __name__ == "__main__":
    main()
