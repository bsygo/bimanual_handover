#!/usr/bin/env python3

import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

class ActionServer:
    def __init__(self):
        self.server = actionlib.SimpleActionServer("/hand/rh_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction, self.execute, False)
        self.server.start()

    def execute(self, goal):
        self.server.set_succeeded()

if __name__ == "__main__":
    rospy.init_node("test_action_server")
    server = ActionServer()
    rospy.spin()
