#!/usr/bin/env python3

import random
import rospy
from std_msgs.msg import Int32

def main():
    rospy.init_node("object_selector")
    pub = rospy.Publisher("/handover/object_selection", Int32, queue_size = 1)
    objects = [1, 2]#, 3]
    while not rospy.is_shutdown():
        input("Please press enter to select next object.")
        if rospy.is_shutdown():
            break
        chosen_object = random.choice(objects)
        rospy.loginfo("The next chosen object is {}.".format(chosen_object))
        chosen_object_msg = Int32()
        chosen_object_msg.data = chosen_object
        pub.publish(chosen_object)
        rospy.sleep(1)

if __name__ == "__main__":
    main()
