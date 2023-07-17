#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import ProcessPC

global pub

def process_pc(req):
    global pub
    pub.publish(req.publish)
    return True

def main():
    global pub
    pub = rospy.Publisher('change_publish', bool)
    rospy.Service('handover/process_pc_srv', ProcessPC, process_pc)
    rospy.spin()

if __name__ == "__main__":
    main()
