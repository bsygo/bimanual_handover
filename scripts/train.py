#!/usr/bin/env python3

import gymnasium as gym
from datetime import datetime
import bimanual_handover.env as handover_env
import rospy
import rospkg
import sys
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
import re


def shutdown():
    roscpp_shutdown()

def main(argv):
    # Initialization
    rospy.init_node('model_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(shutdown)
    rospack = rospkg.RosPack()
    path = rospack.get_path('bimanual_handover')
    fingers = MoveGroupCommander('right_fingers', ns="/")
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)

    # Set model type, might change to parameter
    model_type = "sac"

    # Load config args from function call
    args_pattern = re.compile(r"(((?:--check)\s(?P<CHECK>True|False))?\s?((?:--checkpoint)\s(?P<CHECKPOINT>True|False))?\s?((?:--model)\s(?P<MODEL>.*))?)")
    args = {}
    if len(argv) > 1:
        argv = " ".join(argv[1:])
    else:
        argv = ""
    if match_object := args_pattern.match(argv):
        args = {k: v for k, v in match_object.groupdict().items() if v is not None}
    if 'CHECK' in args:
        check = args['CHECK'] == "True"
        rospy.loginfo('Check parameter set to {}.'.format(check))
    else:
        check = False
        rospy.loginfo('No check parameter specified, defaulting to False.')
    if 'CHECKPOINT' in args:
        checkpoint = args['CHECKPOINT'] == "True"
        rospy.loginfo('Checkpoint parameter set to {}.'.format(checkpoint))
    else:
        checkpoint = False
        rospy.loginfo('No checkpoint parameter specified, defaulting to False.')
    if 'MODEL' in args:
        if args['MODEL'] == 'None':
            model_path = None
        else:
            model_path = path + args['MODEL']
        rospy.loginfo('Model set to {}.'.format(args['MODEL']))
    else:
        model_path = None
        rospy.loginfo('No model parameter specified, defaulting to new model.')

    # Setup environment
    rospy.loginfo('Setting up env.')
    env = handover_env.RealEnv(fingers, env_type = "effort")
    if check:
        check_env(env)
        rospy.loginfo('Env check completed.')

    # Set default parameters
    date = datetime.now()
    str_date = date.strftime("%d_%m_%Y_%H_%M")
    timesteps = 10000
    log_name = "{}_{}".format(model_type, str_date)

    # Setup model
    if model_path is None:
        # Create new model
        rospy.loginfo('Creating {} model.'.format(model_type))
        if model_type == "ppo":
            model = PPO("MlpPolicy", env, n_steps = 50, batch_size = 5, n_epochs = 50, verbose = 1, tensorboard_log = "{}/logs/tensorboard".format(path))
            rospy.loginfo('PPO model created.')
        elif model_type == "sac":
            model = SAC("MlpPolicy", env, batch_size = 50, buffer_size = 10000, verbose = 1, tensorboard_log = "{}/logs/tensorboard".format(path))
            rospy.loginfo('SAC model created.')
    else:
        # Load date from specified model
        log_pattern = re.compile(r".*_(?P<date>\d\d_\d\d_\d\d\d\d_\d\d_\d\d).*")
        matches = log_pattern.match(model_path)
        str_date = matches.group('date')
        log_name = "{}_{}".format(model_type, str_date)
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
    argv = rospy.myargv(argv = sys.argv)
    main(argv)
