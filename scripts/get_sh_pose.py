import rospy
from moveit_commander import MoveGroupCommander
from tf2_ros import TransformListener, Buffer
from tf2_geometry_msgs import do_transform_pose

def main():
    rospy.init_node("get_pose")
    print("start")
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer)
    left_arm = MoveGroupCommander("left_arm")
    rospy.sleep(2)
    gripper_pose = left_arm.get_current_pose(end_effector_link = "l_gripper_tool_frame")
    base_hand_transform = tf_buffer.lookup_transform("rh_manipulator", "base_footprint", rospy.Time(0))
    relative_pose = do_transform_pose(gripper_pose, base_hand_transform)
    print(relative_pose)

if __name__ == "__main__":
    main()
