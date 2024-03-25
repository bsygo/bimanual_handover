#!/usr/bin/env python3

import random
import rospy
from std_msgs.msg import Int32

def main():
    '''
    Randomly shuffle preset object list.
    '''
    objects = [1, 2, 3]
    object_list = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    random.shuffle(object_list)
    print(object_list)

if __name__ == "__main__":
    main()
