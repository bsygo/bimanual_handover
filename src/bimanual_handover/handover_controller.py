#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitialSetupSrv, ProcessPC, MoveHandover, GraspTesterSrv, FinishHandoverSrv, HandoverControllerSrv, HandCloserSrv

class HandoverCommander():

    def __init__(self):
        rospy.init_node('handover_commander')
        rospy.loginfo('Handover commander started.')

        rospy.wait_for_service('initial_setup_srv')
        self.initial_setup_srv = rospy.ServiceProxy('initial_setup_srv', InitialSetupSrv)
        rospy.loginfo('initial_setup_srv initialized.')

        rospy.wait_for_service('process_pc_srv')
        self.process_pc_srv = rospy.ServiceProxy('process_pc_srv', ProcessPC)
        rospy.loginfo('process_pc_srv initialized.')

        rospy.wait_for_service('handover_mover_srv')
        self.handover_mover_srv = rospy.ServiceProxy('handover_mover_srv', MoveHandover)
        rospy.loginfo('handover_mover_srv initialized.')

        rospy.wait_for_service('hand_closer_srv')
        self.hand_closer_srv = rospy.ServiceProxy('hand_closer_srv', HandCloserSrv)
        rospy.loginfo('hand_closer_srv initialized.')

        rospy.wait_for_service('grasp_tester_srv')
        self.grasp_tester_srv = rospy.ServiceProxy('grasp_tester_srv', GraspTesterSrv)
        rospy.loginfo('grasp_tester_srv initialized.')

        self.finish_handover_srv = rospy.ServiceProxy('finish_handover_srv', FinishHandoverSrv)
        rospy.loginfo('finish_handover_srv initialized.')
        self.handover_controller_srv = rospy.Service('handover_controller_srv', HandoverControllerSrv, self.handover_controller_srv)

        rospy.loginfo('handover_controller_srv initialized.')
        rospy.spin()

    def handover_controller_srv(self, req):
        if req.handover_type == "full":
            rospy.loginfo("Launching full handover pipeline.")
        elif req.handover_type == "train":
            rospy.loginfo("Launching handover training setup.")
        else:
            rospy.loginfo("Unknown handover command {}.".format(req.handover_type))
            return False
        return self.full_pipeline(req.handover_type, req.grasp_type, req.object_type)

    def full_pipeline(self, handover_type, grasp_type, object_type):
        rospy.loginfo('Sending service request to initial_setup_srv.')
        if not self.initial_setup_srv('fixed', None):
            rospy.logerr('Moving to inital setup failed.')
            return False

        rospy.loginfo('Sending service request to process_pc_srv.')
        self.process_pc_srv(True)

        rospy.loginfo('Sending service request to handover_mover_srv.')
        if not self.handover_mover_srv('sample', object_type):
            rospy.logerr('Moving to handover pose failed.')
            return False

        if handover_type == "train":
            rospy.loginfo('Training setup finished.')
            return True

        rospy.loginfo('Sending service request to hand_closer_srv.')
        closing_response = self.hand_closer_srv(object_type)
        rospy.loginfo('Closing response: {}'.format(closing_response))
        if not closing_response:
            rospy.logerr('Closing hand failed.')
            return False

        rospy.loginfo('Sending service request to grasp_tester_srv.')
        grasp_response = self.grasp_tester_srv('y', False)
        rospy.loginfo('Grasp response: {}'.format(grasp_response))
        if not grasp_response:
            rospy.logerr('Executing grasp failed.')
            return False

        rospy.loginfo('Sending service request to finish_handover_srv.')
        self.finish_handover_srv('placeholder')

        rospy.loginfo('Handover finished.')
        return True

if __name__ == "__main__":
    handover_commander = HandoverCommander()
