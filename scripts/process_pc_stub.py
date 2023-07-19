#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import ProcessPC
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2

global pub

def process_pc(req):
    global pub
    pub.publish(req.publish)
    rospy.wait_for_message('pc/pc_filtered', PointCloud2)
    return True

def main():
    global pub
    rospy.init_node('process_pc_stub')
    pub = rospy.Publisher('change_publish', Bool, queue_size = 5)
    rospy.Service('handover/process_pc_srv', ProcessPC, process_pc)
    rospy.spin()

if __name__ == "__main__":
    main()
