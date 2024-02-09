#!/usr/bin/env python3

import gymnasium as gym
from datetime import datetime
import bimanual_handover.env as handover_env
import rospy
import rospkg
import sys
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import torch as th
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
import re


def shutdown():
    roscpp_shutdown()

def main():
    # Initialization
    rospy.init_node('model_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(shutdown)
    rospack = rospkg.RosPack()
    path = rospack.get_path('bimanual_handover')
    fingers = MoveGroupCommander('right_fingers', ns="/")
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)

    # Load parameters
    timesteps = rospy.get_param("timesteps")
    model_type = rospy.get_param("model_type")
    env_check = rospy.get_param("env_check")
    checkpoint = rospy.get_param("checkpoint")
    model_path = rospy.get_param("model_path")
    actor_architecture = rospy.get_param("architecture/actor")
    critic_architecture = rospy.get_param("architecture/critic")
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=actor_architecture, qf=critic_architecture))

    # Set logging parameters
    if model_path == '':
        # Setup new parameters
        date = datetime.now()
        str_date = date.strftime("%d_%m_%Y_%H_%M")
        log_name = "{}_{}".format(model_type, str_date)
    else:
        # Load date from specified model
        model_path = path + model_path
        log_pattern = re.compile(r".*_(?P<date>\d\d_\d\d_\d\d\d\d_\d\d_\d\d).*")
        matches = log_pattern.match(model_path)
        str_date = matches.group('date')
        log_name = "{}_{}".format(model_type, str_date)

    # Setup environment
    rospy.loginfo('Setting up env.')
    env = handover_env.RealEnv(fingers, str_date)
    if env_check:
        check_env(env)
        rospy.loginfo('Env check completed.')

    # Setup model
    if model_path == '':
        # Create new model
        rospy.loginfo('Creating {} model.'.format(model_type))
        if model_type == "ppo":
            model = PPO("MlpPolicy", env, policy_kwargs = policy_kwargs, n_steps = 50, batch_size = 5, n_epochs = 50, verbose = 1, tensorboard_log = "{}/logs/tensorboard".format(path))
            rospy.loginfo('PPO model created.')
        elif model_type == "sac":
            model = SAC("MlpPolicy", env, policy_kwargs = policy_kwargs, batch_size = 50, buffer_size = 10000, verbose = 1, tensorboard_log = "{}/logs/tensorboard".format(path))
            rospy.loginfo('SAC model created.')

        # Write model info into file
        joint_values_input = rospy.get_param("observation_space/joint_values")
        pca_input = rospy.get_param("observation_space/pca")
        effort_input = rospy.get_param("observation_space/effort")
        tactile_input = rospy.get_param("observation_space/tactile")
        one_hot_input = rospy.get_param("observation_space/one_hot")
        filename = open("{}/models/model_configs.txt".format(path), "a")
        filename.write("Name: {}_model_{}; joint_values: {}; pca: {}; effort: {}; tactile: {}; one_hot: {}; actor_architecture: {}; critic_architecture: {} \n".format(model_type, str_date, joint_values_input, pca_input, effort_input, tactile_input, one_hot_input, actor_architecture, critic_architecture))
        filename.close()
    else:
        # Update parameters if loaded from checkpoint or finished model
        if checkpoint:
            # Update timesteps to correctly append to graphs
            rospy.loginfo('Load settings from checkpoint.')
            steps_pattern = re.compile(r".*_(?P<steps>\d+)_steps.*")
            matches = steps_pattern.match(model_path)
            steps = int(matches.group('steps'))
            timesteps = timesteps - (steps % timesteps)
            replay_buffer_location = "{}/models/checkpoints/sac_checkpoint_{}_replay_buffer_{}_steps".format(path, str_date, steps)
        else:
            replay_buffer_location = "{}_replay_buffer".format(model_path)
        # Load specified model
        rospy.loginfo('Loading {} model.'.format(model_type))
        if model_type == "ppo":
            model = PPO.load(model_path, env = env)
            rospy.loginfo('PPO model loaded.')
        elif model_type == "sac":
            model = SAC.load(model_path, env = env)
            rospy.loginfo('SAC model loaded.')
            rospy.loginfo('Loading Replay buffer.')
            model.load_replay_buffer(replay_buffer_location)

    # Set checkpoint callback
    checkpoint_callback = CheckpointCallback(save_freq = 100, save_path = "{}/models/checkpoints/".format(path), name_prefix = "{}_checkpoint_{}".format(model_type, str_date), save_replay_buffer = True, save_vecnormalize = True)

    # Train model
    rospy.loginfo('Start learning.')
    model.learn(total_timesteps=timesteps, progress_bar = True, callback = checkpoint_callback, tb_log_name = log_name, reset_num_timesteps = False)
    rospy.loginfo('Learning complete.')

    # Save model
    model.save("{}/models/{}_model_{}".format(path, model_type, str_date))
    if model_type == "sac":
        model.save_replay_buffer("{}/models/{}_model_{}_replay_buffer".format(path, model_type, str_date))
    roscpp_shutdown()

if __name__ == "__main__":
    main()
