#!/usr/bin/env python3

import rospy
from moveit_commander import roscpp_initialize, roscpp_shutdown, MoveGroupCommander
from bimanual_handover.srv import GraspTesterSrv
from geometry_msgs.msg import WrenchStamped

class GraspTester():

    def __init__(self):
        rospy.init_node('grasp_tester')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown())
        self.right_arm = MoveGroupCommander('right_arm')
        rospy.Service('grasp_tester', GraspTesterSrv, self.test_grasp)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def test_grasp(self, req):
        prev_ft = rospy.wait_for_message('/ft/l_gripper_motor', WrenchStamped)
        current_pose = self.right_arm.get_current_pose()
        current_pose.pose.position.z += 0.01
        plan, fraction = self.right_arm.compute_cartesian_path([current_pose.pose], 0.01, 0.0)
        self.right_arm.execute(plan)
        cur_ft = rospy.wait_for_message('/ft/l_gripper_motor', WrenchStamped)
        if prev_ft.wrench.force.z + 1 < cur_ft.wrench.force.z:
            return True
        else::
            return False

if __name__ == "__main__":
    gt = GraspTester()
