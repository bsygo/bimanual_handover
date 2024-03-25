#!/usr/bin/env python3

import rospy
from bimanual_handover_msgs.srv import InitialSetupSrv, ProcessPC, MoveHandover, GraspTesterSrv, FinishHandoverSrv, HandoverControllerSrv, HandCloserSrv
import rospkg
import rosbag
from datetime import datetime

class HandoverCommander():

    def __init__(self):
        rospy.init_node('handover_commander')
        rospy.on_shutdown(self.shutdown)
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

        self.record_attempt = rospy.get_param("record_attempt", False)
        if self.record_attempt:
            time = datetime.now().strftime("%d_%m_%Y_%H_%M")
            pkg_path = rospkg.RosPack().get_path('bimanual_handover')
            self.record_file = open("{}/data/records/handover_attempts_{}.txt".format(pkg_path, time), 'a')
            self.record_bag = rosbag.Bag("{}/data/bags/handover_attempts_{}.bag".format(pkg_path, time), 'w')

        rospy.spin()

    def shutdown(self):
        if self.record_attempt:
            self.record_file.close()
            self.record_bag.close()

    def handover_controller_srv(self, req):
        if req.handover_type == "full":
            rospy.loginfo("Launching full handover pipeline.")
        elif req.handover_type == "train":
            rospy.loginfo("Launching handover training setup.")
        else:
            rospy.loginfo("Unknown handover command {}.".format(req.handover_type))
            return False
        return self.full_pipeline(req.handover_type, req.object_type, req.side)

    def full_pipeline(self, handover_type, object_type, side):
        # Send request to initial_setup.
        rospy.loginfo('Sending service request to initial_setup_srv.')
        setup_mode = rospy.get_param("initial_setup")
        if not self.initial_setup_srv(setup_mode, None).success:
            rospy.logerr('Moving to inital setup failed.')
            return False

        # Send request to process_pc if required.
        grasp_pose_mode = rospy.get_param("handover_mover/grasp_pose_mode")
        if grasp_pose_mode == "pc":
            rospy.loginfo('Sending service request to process_pc_srv.')
            self.process_pc_srv(True)

        # Send request to handover_mover.
        rospy.loginfo('Sending service request to handover_mover_srv.')
        handover_pose_mode = rospy.get_param("handover_mover/handover_pose_mode")
        if self.record_attempt:
            self.record_file.write("Starting handover time: {} \n".format(rospy.Time.now()))
        handover_mover_result = self.handover_mover_srv(side, grasp_pose_mode, handover_pose_mode, object_type)
        if self.record_attempt:
            self.record_file.write("Finishing handover time: {} \n".format(rospy.Time.now()))
        if not handover_mover_result.success:
            rospy.logerr('Moving to handover pose failed.')
            return False

        # Stop here if only train setup is required.
        if handover_type == "train":
            rospy.loginfo('Training setup finished.')
            return True

        # Send request to hand_closer.
        rospy.loginfo('Sending service request to hand_closer_srv.')
        closing_response = self.hand_closer_srv(object_type)
        rospy.loginfo('Closing response: {}'.format(closing_response.finished))
        if not closing_response.finished:
            rospy.logerr('Closing hand failed.')
            return False

        # Send request to grasp_tester.
        closer_type = rospy.get_param("hand_closer/closer_type")
        if not closer_type == "demo":
            rospy.loginfo('Sending service request to grasp_tester_srv.')
            direction = rospy.get_param("grasp_tester/direction")
            grasp_response = self.grasp_tester_srv(direction, False, side)
            rospy.loginfo('Grasp response: {}'.format(grasp_response.success))
            # Stop if grasp failed, allow to continue during attempt recording.
            if not grasp_response.success:
                rospy.logerr('Executing grasp failed.')
                if self.record_attempt:
                    input("Grasped deemed as failure. Please press enter to continue executing the grasp after you have prepared to catch the object. \n")
                else:
                    return False

        rospy.loginfo('Sending service request to finish_handover_srv.')
        self.finish_handover_srv('placeholder')

        rospy.loginfo('Handover finished.')

        # Write data into file if attempt recording.
        if self.record_attempt:
            human_object = input("Please enter which object was used. \n")
            human_success = input("Handover finished. Please enter if the attempt was successful [0] or failed [1]. \n")
            comment = input("Please enter any additional comment about the attempt. \n")
            self.record_file.write("Closer type: {}; Object: {}; Side: {}; Test: {}; Result: {}; Sampling: {} \n".format(closer_type, human_object, side, grasp_response.success, human_success, handover_pose_mode))
            self.record_file.write("Comment: {} \n".format(comment))
            self.record_file.write("––– \n")
            self.record_bag.write('transform', handover_mover_result.transform)

        return True

if __name__ == "__main__":
    handover_commander = HandoverCommander()
