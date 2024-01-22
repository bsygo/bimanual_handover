#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import HandoverControllerSrv
import sys
import rosparam

def main():
    rospy.init_node("launch_handover_node")
    rospy.wait_for_service('/handover/handover_controller_srv')
    srv = rospy.ServiceProxy('/handover/handover_controller_srv', HandoverControllerSrv)
    if len(sys.argv) < 2:
        rospy.loginfo('Missing parameters to call handover_controller. Calling with default parameters.')
        object_type = 'book'
    else:
        object_type = sys.argv[1]
    handover_type = rosparam.get_param("/handover/handover_type")
    result = srv(handover_type, object_type)
    rospy.loginfo('Handover controller finished with {}.'.format(result.success))

if __name__ == "__main__":
    main()
