#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool

def main():
    rospy.init_node("learning_interrupt")
    pub = rospy.Publisher("handover/interrupt_learning", Bool, queue_size = 1)
    while not rospy.is_shutdown():
        response = input("Please press Enter to stop the current PPO learning process temporarily. It will stop before executing the next step.")
        pub.publish(True)
        if rospy.is_shutdown():
            break
        response = input("Please press Enter to resume the training process.")
        pub.publish(False)

if __name__ == "__main__":
    main()
