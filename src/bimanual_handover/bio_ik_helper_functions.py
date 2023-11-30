#!/usr/bin/env python3

import rospy
from bio_ik_msgs.msg import IKRequest
from bio_ik_msgs.srv import GetIK

def prepare_bio_ik_request(group_name, robot_state, timeout_seconds = 1):
    '''
    Prepares an IKRequest for the bio ik solver.

    :param string group_name: Name of the move_group for which this request
        is planned.
    :param float64 timeout_seconds: Number of seconds after which the solver
        stops if no pose was found.

    :return IKRequest: The prepared IKRequest.
    '''
    request = IKRequest()
    request.group_name = group_name
    request.approximate = True
    request.timeout = rospy.Duration.from_sec(timeout_seconds)
    request.avoid_collisions = True
    request.robot_state = robot_state
    return request

def filter_joint_state(joint_state, move_group):
    '''
    Filters the given joint state for only the joint specified in the given
    move group.

    :param JointState joint_state: The joint state to be filtered.
    :param MoveGroup move_group: The move group used as the filter.

    :return JointState: The filtered joint state.
    '''
    filtered_joint_state = joint_state
    joint_names = move_group.get_active_joints()
    indices = [filtered_joint_state.name.index(joint_name) for joint_name in joint_names]
    filtered_joint_state.name = joint_names
    filtered_joint_state.position = [joint_state.position[x] for x in indices]
    if joint_state.velocity:
        filtered_joint_state.velocity = [joint_state.velocity[x] for x in indices]
    if joint_state.effort:
        filtered_joint_state.effort = [joint_state.effort[x] for x in indices]
    return filtered_joint_state
