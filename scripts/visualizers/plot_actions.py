#!/usr/bin/env python3

import rospy
import rospkg
import matplotlib.pyplot as plt
import json

def plot_actions_together(filenames):
    '''
    Plot all actions of each file in filenames together.
    '''
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    labels = ['First Action', 'Second Action', 'Third Action']
    first_action, second_action, third_action = read_data(filenames, pkg_path)
    for i in range(len(filenames)):
        plt.figure(figsize=(5,5))
        plt.plot(first_action[i], label = labels[0])
        plt.plot(second_action[i], label = labels[1])
        plt.plot(third_action[i], label = labels[2])
        plt.xlabel("Timestep")
        plt.ylabel("Action")
        plt.title("All Actions")
        plt.legend()
        plt.show()

def plot_actions_seperate(filenames):
    '''
    Plot each action for each file in filenames seperately.
    '''
    pkg_path = rospkg.RosPack().get_path('bimanual_handover')
    labels = ['Chips Can', 'Bleach Bottle', 'Paper Roll']
    first_action, second_action, third_action = read_data(filenames, pkg_path)
    plt.figure(figsize=(5,5))
    for i in range(len(first_action)):
        plt.plot(first_action[i], label = labels[i])
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title("First Action")
    plt.legend()
    plt.show()
    plt.figure(figsize=(5,5))
    for i in range(len(second_action)):
        plt.plot(second_action[i], label = labels[i])
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title("Second Action")
    plt.legend()
    plt.show()
    plt.figure(figsize=(5,5))
    for i in range(len(third_action)):
        plt.plot(third_action[i], label = labels[i])
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title("Third Action")
    plt.legend()
    plt.show()

def read_data(filenames, pkg_path):
    '''
    Helper function to read action data from json file.
    '''
    full_first_action_data = []
    full_second_action_data = []
    full_third_action_data = []
    for filename in filenames:
        filepath = pkg_path + "/data/actions/" + filename
        with open(filepath, "r") as file:
            data = json.load(file)
        first_action_data = []
        second_action_data = []
        third_action_data = []
        for datapoint in data:
            first_action_data.append(datapoint[0])
            second_action_data.append(datapoint[1])
            third_action_data.append(datapoint[2])
        full_first_action_data.append(first_action_data)
        full_second_action_data.append(second_action_data)
        full_third_action_data.append(third_action_data)
    return full_first_action_data, full_second_action_data, full_third_action_data

def main():
    rospy.init_node("action_plotter")
    filenames = ['action_can_side.json', 'action_bleach_side.json', 'action_roll_side.json']
    plot_actions_seperate(filenames)
    filenames = ['action_can_top.json', 'action_bleach_top.json', 'action_roll_top.json']
    plot_actions_seperate(filenames)
    filenames = ['action_can_constant_input.json']
    plot_actions_together(filenames)

if __name__ == "__main__":
    main()
