#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitGripper, ProcessPC, MoveHandover, GraspExec, FinishHandoverSrv, HandoverControllerSrv

class HandoverCommander():

    def __init__(self):
        rospy.init_node('handover_commander')
        rospy.loginfo('Handover commander started.')
        rospy.wait_for_service('init_gripper_srv')
        self.init_gripper_srv = rospy.ServiceProxy('init_gripper_srv', InitGripper)
        rospy.loginfo('init_gripper_srv initialized.')
        rospy.wait_for_service('process_pc_srv')
        self.process_pc_srv = rospy.ServiceProxy('process_pc_srv', ProcessPC)
        rospy.loginfo('process_pc_srv initialized.')
        rospy.wait_for_service('move_handover_srv')
        self.move_handover_srv = rospy.ServiceProxy('move_handover_srv', MoveHandover)
        rospy.loginfo('move_handover_srv initialized.')
        rospy.wait_for_service('grasp_exec_srv')
        self.grasp_exec_srv = rospy.ServiceProxy('grasp_exec_srv', GraspExec)
        rospy.loginfo('grasp_exec_srv initialized.')
        self.finish_handover_srv = rospy.ServiceProxy('finish_handover_srv', FinishHandoverSrv)
        rospy.loginfo('finish_handover_srv initialized.')
        self.handover_controller_srv = rospy.Service('handover_controller_srv', HandoverControllerSrv, self.handover_controller_srv)
        rospy.loginfo('handover_controller_srv initialized.')
        rospy.spin()

    def handover_controller_srv(self, req):
        return self.launch_handover(req.handover_type, req.grasp_type)

    def launch_handover(self, handover_type, grasp_type):
        if handover_type == "full":
            rospy.loginfo("Launchng full handover pipeline.")
            return self.full_pipeline(grasp_type)
        elif handover_type == "train":
            rospy.loginfo("Launching handover training setup.")
            return self.train_setup()
        else:
            rospy.loginfo("Unknown handover command {}.".format(handover_type))
            return False

    def full_pipeline(self, grasp_type):
        rospy.loginfo('Sending service request to init_gripper_srv.')
        if not self.init_gripper_srv('fixed'):
            rospy.logerr('Moving gripper to inital pose failed.')
            return False
        rospy.loginfo('Sending service request to process_pc_srv.')
        self.process_pc_srv(True)
        rospy.loginfo('Sending service request to move_handover_srv.')
        if not self.move_handover_srv('fixed'):
            rospy.logerr('Moving to handover pose failed.')
            return False
        rospy.loginfo('Sending service request to grasp_exec_srv.')
        grasp_response = self.grasp_exec_srv(grasp_type)
        rospy.loginfo('Grasp response: {}'.format(grasp_response))
        if not grasp_response:
            rospy.logerr('Executing grasp failed.')
            return False
        return True
        rospy.loginfo('Sending service request to finish_handover_srv.')
        self.finish_handover_srv('placeholder')
        rospy.loginfo('Handover finished.')

    def train_setup(self):
        rospy.loginfo('Sending service request to init_gripper_srv.')
        if not self.init_gripper_srv('fixed'):
            rospy.logerr('Moving gripper to inital pose failed.')
            return False
        rospy.loginfo('Sending service request to process_pc_srv.')
        self.process_pc_srv(True)
        rospy.loginfo('Sending service request to move_handover_srv.')
        self.move_handover_srv('fixed')
        rospy.loginfo('Training setup finished.')
        return True
        

if __name__ == "__main__":
    handover_commander = HandoverCommander()
