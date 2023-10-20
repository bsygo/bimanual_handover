#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool

global value
global pub

def update_value(new_value):
    global value
    value = new_value

def main():
    global value, pub
    value = 0
    rospy.init_node("debug_snapshot_node")
    pub = rospy.Publisher("/handover/debug/debug_snapshot_const", Bool, queue_size = 1)
    sub = rospy.Subscriber("/handover/debug/debug_snapshot", Bool, update_value)
    rospy.loginfo("{}".format(rospy.is_shutdown()))
    while not rospy.is_shutdown():
        pub.publish(value)
    
if __name__ == "__main__":
    main()
