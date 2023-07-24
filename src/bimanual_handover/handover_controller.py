#!/usr/bin/env python3

import rospy
from bimanual_handover.srv import InitGripper, ProcessPC, MoveHandover, GraspExec, FinishHandoverSrv

class HandoverCommander():

    def __init__(self):
        rospy.init_node('handover_commander')
        rospy.loginfo('Handover commander started.')
        rospy.wait_for_service('handover/init_gripper_srv')
        self.init_gripper_srv = rospy.ServiceProxy('handover/init_gripper_srv', InitGripper)
        rospy.loginfo('init_gripper_srv initialized.')
        rospy.wait_for_service('handover/process_pc_srv')
        self.process_pc_srv = rospy.ServiceProxy('handover/process_pc_srv', ProcessPC)
        rospy.loginfo('process_pc_srv initialized.')
        rospy.wait_for_service('handover/move_handover_srv')
        self.move_handover_srv = rospy.ServiceProxy('handover/move_handover_srv', MoveHandover)
        rospy.loginfo('move_handover_srv initialized.')
        rospy.wait_for_service('handover/grasp_exec_srv')
        self.grasp_exec_srv = rospy.ServiceProxy('handover/grasp_exec_srv', GraspExec)
        rospy.loginfo('grasp_exec_srv initialized.')
        self.finish_handover_srv = rospy.ServiceProxy('handover/finish_handover_srv', FinishHandoverSrv)
        rospy.loginfo('finish_handover_srv initialized.')

    def full_pipeline(self):
        rospy.loginfo('Sending service request to init_gripper_srv.')
        if not self.init_gripper_srv('fixed'):
            rospy.logerr('Moving gripper to inital pose failed.')
            return
        rospy.loginfo('Sending service request to process_pc_srv.')
        self.process_pc_srv(True)
        rospy.loginfo('Sending service request to move_handover_srv.')
        self.move_handover_srv('fixed')
        rospy.loginfo('Sending service request to grasp_exec_srv.')
        self.grasp_exec_srv('placeholder')
        rospy.loginfo('Sending service request to finish_handover_srv.')
        self.finish_handover_srv('placeholder')
        rospy.loginfo('Handover finished.')


if __name__ == "__main__":
    handover_commander = HandoverCommander()
    handover_commander.full_pipeline()
