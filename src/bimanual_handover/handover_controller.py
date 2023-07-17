#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import InitGripper, ProcessPC, GraspGen, MoveHandover, GraspExec

class HandoverCommander():

    def __init__(self):
        rospy.wait_for_service('handover/init_gripper_srv')
        self.init_gripper_srv = rospy.ServiceProxy('handover/init_gripper_srv', InitGripper)
        rospy.wait_for_service('handover/process_pc_srv')
        self.process_pc_srv = rospy.ServiceProxy('handover/process_pc_srv', ProcessPC)
        rospy.wait_for_service('handover/move_handover_srv')
        self.move_handover_srv = rospy.ServiceProxy('handover/move_handover_srv', MoveHandover)
        rospy.wait_for_service('handover/grasp_exec_srv')
        self.grasp_exec_srv = rospy.ServiceProxy('handover/grasp_exec_srv', GraspExec)

    def full_pipeline(self):
        if not self.init_gripper_srv('fixed'):
            rospy.logerr('Moving gripper to inital pose failed.')
            return
        self.process_pc_srv(True)
        self.move_handover_srv('fixed')
        self.grasp_exec_srv('placeholder')


if __name__ == "__main__":
    handover_commander = HandoverCommander()
