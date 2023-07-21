#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from bimanual_handover.srv import FinishHandoverSrv

class FinishHandover():

    def __init__(self):
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)
        self.gripper = MoveGroupCommander('left_gripper')
        self.left_arm = MoveGroupCommander('left_arm')
        self.right_arm = MoveGroupCommander('right_arm_pr2')
        self.psi = PlanningSceneInterface()
        rospy.Service('handover/finish_handover_srv', FinishHandoverSrv, self.finish_handover)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def finish_handover(self, req):
        self.psi.remove_world_object('can')
        self.gripper.set_named_target('open')
        self.gripper.go()
        self.left_arm.set_named_target('left_arm_to_side')
        self.left_arm.go()
        self.right_arm.set_named_target('right_arm_to_side')
        self.right_arm.go()
        return True

if __name__ == "__main__":
    rospy.init_node('finish_handover')
    fh = FinishHandover()
