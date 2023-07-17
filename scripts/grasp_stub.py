#!/usr/bin/env python3

import rospy

global ccm_srv

def grasp_exec(req):
    return ccm_srv(req.placeholder)

def main():
    global ccm_srv
    rospy.Service('handover/grasp_exec_srv', GraspExec, grasp_exec)
    rospy.wait_for_service('handover/ccm')
    ccm_srv = rospy.ServiceProxy('handover/ccm', CCM)
    rospy.spin()

if __name__ == "__main__":
    main()
