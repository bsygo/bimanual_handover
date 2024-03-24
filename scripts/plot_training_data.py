#!/usr/bin/env python3

import rospy
import rospkg
import matplotlib.pyplot as plt
import json

def plot_entropy_coefficient():
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    filenames = ['sac_26_02_2024_10_24_ent.json', 'sac_27_02_2024_12_38_ent.json', 'sac_28_02_2024_10_08_ent.json']
    labels = ['Attempt 1', 'Attempt 2', 'Attempt 3']
    x_data, y_data = read_data(filenames, pkg_path)
    f = plt.figure(figsize=(5,5))
    for i in range(len(x_data)):
        plt.plot(x_data[i], y_data[i], label = labels[i])
    plt.xlabel("Steps")
    plt.ylabel("Entropy Coefficient")
    plt.title("Adaptable Entropy Coefficient")
    plt.legend()
    plt.show()

def plot_episode_length():
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    filenames = ['sac_26_02_2024_10_24_episode.json', 'sac_27_02_2024_12_38_episode.json', 'sac_28_02_2024_10_08_episode.json', 'sac_28_02_2024_13_15_episode.json', 'sac_28_02_2024_17_09_episode.json', 'sac_29_02_2024_13_37_episode.json']
    labels = ['Attempt 1', 'Attempt 2', 'Attempt 3', 'Attempt 4', 'Attempt 5', 'Attempt 6']
    x_data, y_data = read_data(filenames, pkg_path)
    f = plt.figure(figsize=(5,5))
    for i in range(len(x_data)):
        plt.plot(x_data[i], y_data[i], label = labels[i])
    plt.xlabel("Steps")
    plt.ylabel("Episode Length")
    plt.title("Average Episode Length")
    plt.legend()
    plt.show()

def plot_reward():
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    #filenames = ['sac_19_03_2024_15_34_reward.json']
    #labels = ['Top Grasp Model']
    filenames = ['sac_26_02_2024_10_24_reward.json', 'sac_27_02_2024_12_38_reward.json', 'sac_28_02_2024_10_08_reward.json', 'sac_28_02_2024_13_15_reward.json', 'sac_28_02_2024_17_09_reward.json', 'sac_29_02_2024_13_37_reward.json']
    labels = ['Attempt 1', 'Attempt 2', 'Attempt 3', 'Attempt 4', 'Attempt 5', 'Attempt 6']
    x_data, y_data = read_data(filenames, pkg_path)
    f = plt.figure(figsize=(5,5))
    for i in range(len(x_data)):
        plt.plot(x_data[i], y_data[i], label = labels[i])
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("Average Reward")
    plt.legend()
    plt.show()

def read_data(filenames, pkg_path):
    full_x_data = []
    full_y_data = []
    for filename in filenames:
        filepath = pkg_path + "/data/plots/" + filename
        with open(filepath, "r") as file:
            data = json.load(file)
        x_data = []
        y_data = []
        for datapoint in data:
            x_data.append(datapoint[1])
            y_data.append(datapoint[2])
        full_x_data.append(x_data)
        full_y_data.append(y_data)
    return full_x_data, full_y_data

def main():
    rospy.init_node("training_plotter")
    plot_entropy_coefficient()
    plot_episode_length()
    plot_reward()

if __name__ == "__main__":
    main()
