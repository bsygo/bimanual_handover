#!/usr/bin/env python3

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from bimanual_handover_msgs.srv import FinishHandoverSrv
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from tf.transformations import quaternion_multiply, quaternion_from_euler
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

class FinishHandover():

    def __init__(self):
        # Initialize
        rospy.init_node('finish_handover')
        roscpp_initialize('')
        rospy.on_shutdown(self.shutdown)

        # Setup move_groups
        self.gripper = MoveGroupCommander('left_gripper', ns = "/")
        self.left_arm = MoveGroupCommander('left_arm', ns = "/")
        self.right_arm = MoveGroupCommander('right_arm_pr2', ns = "/")
        self.right_hand = MoveGroupCommander('right_arm', ns = "/")

        # Setup tf buffer
        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer)

        # Debug
        self.debug = rospy.get_param("finish_handover/debug")
        if self.debug:
            self.debug_pub = rospy.Publisher('debug/finish_handover', PoseStamped, latch = True, queue_size = 1)

        # Start service
        rospy.Service('finish_handover_srv', FinishHandoverSrv, self.finish_handover)
        rospy.spin()

    def shutdown(self):
        roscpp_shutdown()

    def finish_handover(self, req):
        # Open the gripper
        self.gripper.set_named_target('open')
        self.gripper.go()

        # Retract the gripper
        gripper_pose = PoseStamped()
        gripper_pose.header.frame_id = "l_gripper_tool_frame"
        gripper_pose.pose.position.x += -0.1
        gripper_pose.pose.orientation.w = 1
        gripper_base_transform = self.tf_buffer.lookup_transform("base_footprint", "l_gripper_tool_frame", rospy.Time(0))
        gripper_pose = do_transform_pose(gripper_pose, gripper_base_transform)
        plan, fraction = self.left_arm.compute_cartesian_path([gripper_pose.pose], 0.01, 0.0)
        if fraction < 0.8:
            rospy.loginfo('Only {} of the path to remove the gripper from the object found.'.format(fraction))
            #return False
        self.left_arm.execute(plan)

        input("wait")

        # Move left arm to side
        self.left_arm.set_named_target('left_arm_to_side')
        self.left_arm.go()

        # Set final pose for right arm
        final_pose = PoseStamped()
        final_pose.header.frame_id = "base_footprint"
        final_pose.pose.position.x = 0.255
        final_pose.pose.position.y = -0.662
        final_pose.pose.position.z = 0.675
        final_pose.pose.orientation = Quaternion(*quaternion_from_euler(1.5708, 0, 1.5708))

        if self.debug:
            self.debug_pub.publish(final_pose)

        # Move right arm to final pose
        self.right_hand.set_pose_target(final_pose)
        self.right_hand.go()
        return True

if __name__ == "__main__":
    fh = FinishHandover()
