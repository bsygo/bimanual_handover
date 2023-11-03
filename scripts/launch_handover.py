#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import HandoverControllerSrv
import sys

def main():
    rospy.wait_for_service('handover_controller_srv')
    srv = rospy.ServiceProxy('handover_controller_srv', HandoverControllerSrv)
    req = HandoverControllerSrv()
    if len(sys.argv) < 3:
        rospy.loginfo('Missing parameters to call handover_controller. Calling with default parameters.')
        req.grasp_type = 'ccm'
        req.handover_type = 'full'
    else:
        req.grasp_type = sys.argv[1]
        req.handover_type = sys.argv[2]
    result = srv(req)
    rospy.loginfo('Handover controller finished with {}.'.format(result.success))

if __name__ == "__main__":
    main()
