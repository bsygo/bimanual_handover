#!/usr/bin/env python3

import gymnasium as gym
from datetime import datetime
import bimanual_handover.env as handover_env
import rospy
import rospkg
import sys
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown, PlanningSceneInterface
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from sensor_msgs.msg import PointCloud2
from tqdm.auto import tqdm
from stable_baselines3.common.callbacks import CheckpointCallback
import re


def shutdown():
    roscpp_shutdown()

def main(argv):
    rospy.init_node('ppo_trainer')
    roscpp_initialize('')
    rospy.on_shutdown(shutdown)
    rospack = rospkg.RosPack()
    path = rospack.get_path('bimanual_handover')
    fingers = MoveGroupCommander('right_fingers', ns="/")
    fingers.set_max_velocity_scaling_factor(1.0)
    fingers.set_max_acceleration_scaling_factor(1.0)

    args_pattern = re.compile(r"(((?:--check)\s(?P<CHECK>True|False))?\s?((?:--checkpoint)\s(?P<CHECKPOINT>True|False))?\s?((?:--model)\s(?P<MODEL>.*))?)")
    args = {}
    if len(argv) > 1:
        argv = " ".join(argv[1:])
    else:
        argv = ""
    if match_object := args_pattern.match(argv):
        args = {k: v for k, v in match_object.groupdict().items() if v is not None}
    if 'CHECK' in args:
        check = args ['CHECK']
    else:
        check = False
        rospy.loginfo('No check parameter specified, defaulting to False.')
    if 'CHECKPOINT' in args:
        checkpoint = args['CHECKPOINT']
    else:
        checkpoint = False
        rospy.loginfo('No checkpoint parameter specified, defaulting to False.')
    if 'MODEL' in args:
        if args['MODEL'] == 'None':
            model_path = None
        else:
            model_path = path + args['MODEL']
    else:
        model_path = None #path + "/models/ppo_model"
        rospy.loginfo('No model parameter specified, defaulting to new model.')

#    rospy.loginfo('Waiting for pc.')
#    pc = rospy.wait_for_message('pc/pc_filtered', PointCloud2, 20)
    rospy.loginfo('Setting up env.')
    env = handover_env.RealEnv(fingers)
#    env = handover_env.MimicEnv(fingers)

    date = datetime.now()
    str_date = date.strftime("%d_%m_%Y_%H_%M")
    timesteps = 10000#100000
    log_name = "PPO_" + str_date

    if check:
        check_env(env)
        rospy.loginfo('Env check completed.')
    checkpoint_callback = CheckpointCallback(save_freq = 100, save_path = path + "/models/checkpoints/", name_prefix = "ppo_checkpoint_" + str_date, save_replay_buffer = True, save_vecnormalize = True)
    if model_path is None:
        model = PPO("MlpPolicy", env, n_steps = 50, batch_size = 5, n_epochs = 50, verbose = 1, tensorboard_log=path + "/logs/tensorboard")
    else:
        model = PPO.load(model_path, env = env, print_system_info = True)
        if checkpoint:
            steps_pattern = re.compile(r".*_(?P<steps>\d+)_steps.*")
            matches = steps_pattern.match(model_path)
            steps = int(matches.group('steps'))
            timesteps = timesteps - steps
            log_pattern = re.compile(r".*ppo_checkpoint_(?P<date>\d\d_\d\d_\d\d\d\d_\d\d_\d\d)_.*")
            matches = log_pattern.match(model_path)
            date = matches.group('date')
            log_name = "PPO_" + date
    rospy.loginfo('Start learning.')
    model.learn(total_timesteps=timesteps, progress_bar = True, callback = checkpoint_callback, tb_log_name = log_name, reset_num_timesteps = False)
    rospy.loginfo('Learning complete.')
    model.save(path + "/models/ppo_model_" + str_date)
    roscpp_shutdown()

if __name__ == "__main__":
    argv = rospy.myargv(argv = sys.argv)
    main(argv)
