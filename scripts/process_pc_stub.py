#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import ProcessPC
from std_msgs.msg import Bool
from sensor_msgs.msg import PointCloud2

global pub, received

def process_pc(req):
    global pub, received
    pub.publish(req.publish)
    while not received:
        rospy.sleep(0.1)
    received = False
    return True

def pc_recevied(pc):
    global received
    received = True

def main():
    global pub, received
    rospy.init_node('process_pc_stub')
    received = False
    pub = rospy.Publisher('handover/pc/publish_pc', Bool, queue_size = 5)
    sub = rospy.Subscriber('handover/pc/pc_filtered', PointCloud2, pc_received)
    rospy.Service('handover/process_pc_srv', ProcessPC, process_pc)
    rospy.spin()

if __name__ == "__main__":
    main()
