#!/usr/bin/env python3

import rospy
import rospkg
import rosbag
import os
from visualization_msgs.msg import Marker
from std_msgs.msg import String, ColorRGBA, Int64, Bool
from bimanual_handover_msgs.msg import Layers, Volume
from geometry_msgs.msg import Vector3, PoseStamped, Quaternion, Point
from tf.transformations import quaternion_from_euler
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from bimanual_handover.workspace_analyzer import StepTransform, TransformHandler
from datetime import datetime
from copy import deepcopy

class WorkspaceVisualizerV2():

    def __init__(self):
        self.data = None
        self.pkg_path = rospkg.RosPack().get_path('bimanual_handover')
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")

        # Initialize publishers and subscribers
        self.load_json_sub = rospy.Subscriber("workspace_visualizer/load_json", String, self.load_json)
        self.combine_jsons_sub = rospy.Subscriber("workspace_visualizer/combine_jsons", String, self.combine_jsons)
        self.print_min_max_sub = rospy.Subscriber("workspace_visualizer/print_min_max", String, self.print_min_max_values)
        self.cut_data_sub = rospy.Subscriber("workspace_visualizer/cut_data", Int64, self.cut_data)
        self.write_bag_sub = rospy.Subscriber("workspace_visualizer/write_bag", Int64, self.write_transforms_to_rosbag_receiver)
        self.plot_transform_data_sub = rospy.Subscriber("workspace_visualizer/plot_transform_data", Bool, self.plot_transform_data)
        self.write_intersection_bag_sub = rospy.Subscriber("workspace_visualizer/write_intersection_bag", Int64, self.write_intersection_to_rosbag)
        self.volume_percentage_sub = rospy.Subscriber("workspace_visualizer/set_volume_percentage", Volume, self.publish_volume_percentage)
        self.volume_sub = rospy.Subscriber("workspace_visualizer/set_volume", Volume, self.publish_volume)
        self.volume_pub = rospy.Publisher("workspace_visualizer/pub_volume", Marker, queue_size = 1, latch = True)
        self.hand_marker_sub = rospy.Subscriber("workspace_visualizer/set_hand_marker", String, self.publish_initial_hand_positions)
        self.hand_marker_pub = rospy.Publisher("workspace_visualizer/pub_hand_marker", Marker, queue_size = 1, latch = True)
        self.intersection_sub = rospy.Subscriber("workspace_visualizer/set_intersection", Int64, self.publish_intersection)
        self.intersection_pub = rospy.Publisher("workspace_visualizer/pub_intersection", Marker, queue_size = 1, latch = True)

        rospy.loginfo("VisualizerV2 ready.")
        rospy.spin()

    def load_json(self, file_name):
        '''
        Loads the data from the specified .json file.
        '''
        self.data = TransformHandler.load_independent(self.pkg_path + "/data/workspace_analysis/" + file_name.data)
        rospy.loginfo("Data from file {} loaded.".format(file_name.data))

    def combine_jsons(self, filename):
        # Load first file with load_json and call this with second file to combine
        second_data = TransformHandler.load_independent(self.pkg_path + "/data/workspace_analysis/" + filename.data)
        rospy.loginfo("Data from file {} loaded.".format(filename.data))
        for key, value in second_data.items():
            self.data[key] = value
        filepath = self.pkg_path + "/data/workspace_analysis/workspace_analysis_" + self.time + ".json"
        TransformHandler.save_independent(self.data, filepath)
        rospy.loginfo("Saved combined data.")

    def get_min_max_values(self, data = None):
        '''
        Return the min and max value along all three axes for the given data.
        If no data is specified, use the currently loaded data.
        '''
        if data is None:
            data = self.data
        min_max_values = {}
        min_max_values["x_min"] = 1000
        min_max_values["y_min"] = 1000
        min_max_values["z_min"] = 1000
        min_max_values["x_max"] = -1000
        min_max_values["y_max"] = -1000
        min_max_values["z_max"] = -1000
        for transform in data.values():
            if transform.x < min_max_values["x_min"]:
                min_max_values["x_min"] = transform.x
            if transform.y < min_max_values["y_min"]:
                min_max_values["y_min"] = transform.y
            if transform.z < min_max_values["z_min"]:
                min_max_values["z_min"] = transform.z
            if transform.x > min_max_values["x_max"]:
                min_max_values["x_max"] = transform.x
            if transform.y > min_max_values["y_max"]:
                min_max_values["y_max"] = transform.y
            if transform.z > min_max_values["z_max"]:
                min_max_values["z_max"] = transform.z
        return min_max_values

    def print_min_max_values(self, msg):
        '''
        Print the min and max values along all three axes for the currently loaded data.
        '''
        min_max_values = self.get_min_max_values()
        rospy.loginfo("x min: {}".format(min_max_values["x_min"]))
        rospy.loginfo("y min: {}".format(min_max_values["y_min"]))
        rospy.loginfo("z min: {}".format(min_max_values["z_min"]))
        rospy.loginfo("x max: {}".format(min_max_values["x_max"]))
        rospy.loginfo("y max: {}".format(min_max_values["y_max"]))
        rospy.loginfo("z max: {}".format(min_max_values["z_max"]))

    def get_transforms_layers(self):
        '''
        Get the values of the currently loaded data dict as three dicts, where
        each dict contains all data split into layers along either the x, y, or z axes.
        '''
        x_layers = {}
        y_layers = {}
        z_layers = {}
        for transform in self.data.values():
            if transform.x in x_layers.keys():
                x_layers[transform.x].append(transform)
            else:
                x_layers[transform.x] = [transform]
            if transform.y in y_layers.keys():
                y_layers[transform.y].append(transform)
            else:
                y_layers[transform.y] = [transform]
            if transform.z in z_layers.keys():
                z_layers[transform.z].append(transform)
            else:
                z_layers[transform.z] = [transform]
        rospy.loginfo("Split data into layers.")
        return x_layers, y_layers, z_layers

    def cut_data(self, threshold):
        '''
        Remove unnecessary data, specifically layers that contain no valid solutions.
        '''
        # Threshold is maximum number of failed solution allowed
        threshold = threshold.data
        # Decide if to keep one additional buffer layer around the data
        add_buffer = True
        x_layers, y_layers, z_layers = self.get_transforms_layers()
        cut_data = deepcopy(self.data)

        # Go through each layer and remove invalid layers
        for layer in x_layers.keys():
            cut = True
            for transform in x_layers[layer]:
                if transform.number_solutions <= threshold:
                    cut = False
                    break
            if cut:
                for transform in x_layers[layer]:
                    if transform.key() in cut_data.keys():
                        del cut_data[transform.key()]
        for layer in y_layers.keys():
            cut = True
            for transform in y_layers[layer]:
                if transform.number_solutions <= threshold:
                    cut = False
                    break
            if cut:
                for transform in y_layers[layer]:
                    if transform.key() in cut_data.keys():
                        del cut_data[transform.key()]
        for layer in z_layers.keys():
            cut = True
            for transform in z_layers[layer]:
                if transform.number_solutions <= threshold:
                    cut = False
                    break
            if cut:
                for transform in z_layers[layer]:
                    if transform.key() in cut_data.keys():
                        del cut_data[transform.key()]

        # If desired, add the buffer again
        if add_buffer:
            min_max_values = self.get_min_max_values(data = cut_data)
            print(min_max_values)
            if min_max_values["x_min"] - 1 in x_layers.keys():
                for transform in x_layers[min_max_values["x_min"] - 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform
            if min_max_values["x_max"] + 1 in x_layers.keys():
                for transform in x_layers[min_max_values["x_max"] + 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform
            if min_max_values["y_min"] - 1 in y_layers.keys():
                for transform in y_layers[min_max_values["y_min"] - 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform
            if min_max_values["y_max"] + 1 in y_layers.keys():
                for transform in y_layers[min_max_values["y_max"] + 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform
            if min_max_values["z_min"] - 1 in z_layers.keys():
                for transform in z_layers[min_max_values["z_min"] - 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform
            if min_max_values["z_max"] + 1 in z_layers.keys():
                for transform in z_layers[min_max_values["z_max"] + 1]:
                    if not transform.key() in cut_data.keys():
                        cut_data[transform.key()] = transform

        # Cut off buffer excess
        excess_keys = []
        for transform in cut_data.values():
            if transform.x < (min_max_values["x_min"] - 1):
                excess_keys.append(transform.key())
            if transform.x > (min_max_values["x_max"] + 1):
                excess_keys.append(transform.key())
            if transform.y < (min_max_values["y_min"] - 1):
                excess_keys.append(transform.key())
            if transform.y > (min_max_values["y_max"] + 1):
                excess_keys.append(transform.key())
            if transform.z < (min_max_values["z_min"] - 1):
                excess_keys.append(transform.key())
            if transform.z > (min_max_values["z_max"] + 1):
                excess_keys.append(transform.key())
        for key in excess_keys:
            if key in cut_data.keys():
                del cut_data[key]

        filepath = self.pkg_path + "/data/workspace_analysis/workspace_analysis_" + self.time + ".json"
        TransformHandler.save_independent(cut_data, filepath)
        rospy.loginfo("Saved cut data.")

    def write_transforms_to_rosbag_receiver(self, threshold):
        '''
        Helper function to allow writing data to a rosbag through a rostopic call.
        '''
        self.write_transforms_to_rosbag(threshold.data)

    def write_transforms_to_rosbag(self, threshold, data = None):
        '''
        Writes the current data, that is below the given threshold, as transform_msgs
        in a rosbag.
        '''
        # Threshold is maximum number of failed solution allowed
        path = self.pkg_path + "/data/workspace_analysis/"
        bag = rosbag.Bag('{}workspace_analysis_{}.bag'.format(path, self.time), 'w')
        if data is None:
            data = self.data.values()
        for transform in data:
            if transform.number_solutions <= threshold:
                msgs = transform.get_transform_msgs()
                for i in range(len(msgs)):
                    bag.write('transforms', msgs[i])
        bag.close()
        rospy.loginfo("Transforms written into rosbag.")

    def get_data_as_lists(self, ignore_invalid_transforms = True):
        '''
        Get the data as lists for each metric.
        '''
        number_solutions_data = []
        min_score_data = []
        avg_score_data = []
        recalculated_avg_score = self.recalculate_avg_score()
        for transform in self.data.values():
            if ignore_invalid_transforms:
                number_solutions_data.append(343 - transform.number_solutions)
                min_score_data.append(transform.min_score)
                avg_score_data.append(recalculated_avg_score[transform.key()])
            else:
                if not transform.number_solutions == 343:
                    number_solutions_data.append(343 - transform.number_solutions)
                if not transform.min_score == 1.0:
                    min_score_data.append(transform.min_score)
                if not recalculated_avg_score[transform.key()] == 1.0:
                    avg_score_data.append(recalculated_avg_score[transform.key()])
        return number_solutions_data, min_score_data, avg_score_data

    def get_percentage_cutoff(self, percentage, mode = "smaller", ignore_invalid = True):
        '''
        Get the cutoff value that splits the loaded data along the specified percentage.
        '''
        number_solutions_data, min_score_data, avg_score_data = self.get_data_as_lists()
        if not ignore_invalid:
            number_transforms = len(self.data.keys())
            cutoff = int(number_transforms/100 * percentage)
        else:
            number_transforms = len(number_solutions_data)
            cutoff = int(number_transforms/100 * percentage)

        if mode == "smaller":
            number_solutions_data.sort(reverse = True)
            min_score_data.sort()
            avg_score_data.sort()
        elif mode == "greater":
            number_solutions_data.sort()
            min_score_data.sort(reverse = True)
            avg_score_data.sort(reverse = True)

        rospy.loginfo("The best {} percent of transforms have equal or above {} solutions.".format(percentage, number_solutions_data[cutoff]))
        rospy.loginfo("The best {} percent of transforms have equal or above {} min_score.".format(percentage, min_score_data[cutoff]))
        rospy.loginfo("The best {} percent of transforms have equal or above {} avg_score.".format(percentage, avg_score_data[cutoff]))

        return number_solutions_data[cutoff], min_score_data[cutoff], avg_score_data[cutoff]

    def plot_transform_data(self, msg):
        '''
        Plot the data as three plots, one for each metric.
        '''
        number_solutions_data, min_score_data, avg_score_data = self.get_data_as_lists(msg.data)

        plt.figure(figsize=(5,5))
        plt.hist(number_solutions_data, bins = int(max(number_solutions_data)), range=(0, max(number_solutions_data)))
        plt.title("Valid Solution")
        plt.xlabel("Number of Valid Solutions", fontsize = 12)
        plt.ylabel("Number of Positions", fontsize = 12)
        plt.show()

        plt.figure(figsize=(5,5))
        plt.hist(min_score_data, bins = 100, range=(0, 1))
        plt.title("Minimal Cost")
        plt.xlabel("Cost", fontsize = 12)
        plt.ylabel("Number of Positions", fontsize = 12)
        plt.show()

        plt.figure(figsize=(5,5))
        plt.hist(avg_score_data, bins = 100, range=(0, 1))
        plt.title("Average Cost")
        plt.xlabel("Cost", fontsize = 12)
        plt.ylabel("Number of Positions", fontsize = 12)
        plt.show()

    def publish_initial_hand_positions(self, msg):
        '''
        Publish the initial hand position.
        '''
        transform = self.data["[0, 0, 0]"]
        print(transform.x)
        print(transform.y)
        print(transform.z)
        marker = create_marker("hand_markers")
        marker.scale = Vector3(0.01, 0.01, 0.01)
        for i in range(len(transform.hand_positions)):
            if transform.scores[i] == 1.0:
                marker.colors.append(ColorRGBA(1, 0, 0, 1))
            else:
                marker.colors.append(ColorRGBA(0, 1, 0, 1))
            marker.points.append(Point(transform.hand_positions[i].x, transform.hand_positions[i].y, transform.hand_positions[i].z))
        self.hand_marker_pub.publish(marker)

    def get_intersection_data(self, percentage):
        '''
        Get the data from the currently loaded data that belongs to the intersection
        of all three metrics below the given percentage.
        '''
        number_solutions_cutoff, min_score_cutoff, avg_score_cutoff = self.get_percentage_cutoff(percentage)
        recalculated_data = self.recalculate_avg_score()
        filtered_inverted_values = []

        filtered_transforms = []
        colors = []
        for transform in self.data.values():
            inverted_value = (343 - transform.number_solutions)
            if inverted_value >= number_solutions_cutoff:
                if recalculated_data[transform.key()] <= avg_score_cutoff:
                    if transform.min_score <= min_score_cutoff:
                        filtered_transforms.append(transform)
                        colors.append(ColorRGBA(transform.min_score, 1 - transform.min_score, 0, 1))
        return filtered_transforms, colors

    def write_intersection_to_rosbag(self, percentage):
        '''
        Helper function to allow writing the intersection of all three metrics
        below the given percentage as transform_msgs into a rosbag.
        '''
        filtered_transforms, _ = self.get_intersection_data(percentage.data)
        self.write_transforms_to_rosbag(343, filtered_transforms)

    def publish_intersection(self, percentage):
        '''
        Publish the intersection of all three metrics below the given percentage
        as markers for RVIZ.
        '''
        filtered_transforms, colors = self.get_intersection_data(percentage.data)
        markers = self.create_markers(filtered_transforms, colors)
        self.intersection_pub.publish(markers)
        rospy.loginfo("Intersection volume published.")

    def publish_volume_percentage(self, msg):
        '''
        Publish the subset of the loaded data specified in the msg as markers in
        RVIZ. Interpret the message as a percentage call.
        '''
        number_cutoff, min_cutoff, avg_cutoff = self.get_percentage_cutoff(msg.threshold, mode = msg.mode)
        if msg.data_type == "number_solutions":
            msg.threshold = number_cutoff
            if msg.mode == "smaller":
                msg.mode = "greater"
            else:
                msg.mode = "smaller"
        elif msg.data_type == "avg_score":
            msg.threshold = avg_cutoff
        elif msg.data_type == "min_score":
            msg.threshold = min_cutoff
        print(msg)
        self.publish_volume(msg)

    def publish_volume(self, msg):
        '''
        Publish the subset of the loaded data specified in the msg as markers in
        RVIZ.
        '''

        data_type = msg.data_type
        threshold = msg.threshold
        mode = msg.mode
        # Decide if haling the data along an axis or make the alpha value
        # variable (only number solutions)
        alpha_scaling = False
        xy_halved = False
        xz_halved = False

        # Ignore invalid solutions for avg_score
        if data_type == "avg_score":
            recalculated_data = self.recalculate_avg_score()
        elif data_type == "number_solutions":
            filtered_inverted_values = []

        if xy_halved or xz_halved:
            min_max_values = self.get_min_max_values()

        filtered_transforms = []
        colors = []
        # Go through the loaded data and filter after given criteria
        for transform in self.data.values():
            if xy_halved:
                if transform.z > (min_max_values["z_max"] + min_max_values["z_min"])/2 + 1:
                    continue
            elif xz_halved:
                if transform.y < (min_max_values["y_max"] + min_max_values["y_min"])/2:
                    continue
            if data_type == "number_solutions":
                inverted_value = (343 - transform.number_solutions)
                if mode == "greater":
                    if inverted_value > threshold:
                        filtered_transforms.append(transform)
                        filtered_inverted_values.append(inverted_value)
                elif mode == "smaller":
                    if inverted_value < threshold:
                        filtered_transforms.append(transform)
                        filtered_inverted_values.append(inverted_value)
            elif data_type == "avg_score":
                if mode == "greater":
                    if recalculated_data[transform.key()] > threshold:
                        filtered_transforms.append(transform)
                        colors.append(ColorRGBA(recalculated_data[transform.key()], 1 - recalculated_data[transform.key()], 0, 1))
                elif mode == "smaller":
                    if recalculated_data[transform.key()] < threshold:
                        filtered_transforms.append(transform)
                        colors.append(ColorRGBA(recalculated_data[transform.key()], 1 - recalculated_data[transform.key()], 0, 1))
            elif data_type == "min_score":
                if mode == "greater":
                    if transform.min_score > threshold:
                        filtered_transforms.append(transform)
                        colors.append(ColorRGBA(transform.min_score, 1 - transform.min_score, 0, 1))
                elif mode == "smaller":
                    if transform.min_score < threshold:
                        filtered_transforms.append(transform)
                        colors.append(ColorRGBA(transform.min_score, 1 - transform.min_score, 0, 1))

        # Add the color values for the number solutions metric
        if data_type == "number_solutions":
            max_value = max(filtered_inverted_values)
            for value in filtered_inverted_values:
                if max_value == 0:
                    color_value = 0
                else:
                    color_value = value/max_value
                if alpha_scaling:
                    colors.append(ColorRGBA(1 - color_value, color_value, 0, color_value))
                else:
                    colors.append(ColorRGBA(1 - color_value, color_value, 0, 1))

        markers = self.create_markers(filtered_transforms, colors)
        self.volume_pub.publish(markers)
        rospy.loginfo("Volume published.")

    def recalculate_avg_score(self):
        '''
        Correct the avg score data by no longer counting invalid solutions.
        '''
        data = {}
        for transform in self.data.values():
            unnormalized_data = transform.avg_score * 343
            unnormalized_data = unnormalized_data - transform.number_solutions
            if unnormalized_data == 0:
                data[transform.key()] = 1.0
            else:
                normalized_data = unnormalized_data/(343 - transform.number_solutions)
                data[transform.key()] = normalized_data
        return data

    def create_markers(self, transforms, colors):
        '''
        Helper function to create markers for the given transforms and colors.
        '''
        markers = create_marker("volume")
        markers.colors = colors
        gripper_pose = get_gripper_pose()
        for transform in transforms:
            markers.points.append(Point(gripper_pose.pose.position.x + transform.x * 0.06, gripper_pose.pose.position.y + transform.y * 0.06, gripper_pose.pose.position.z + transform.z * 0.06))
        return markers

def get_gripper_pose():
    '''
    Get the static gripper start pose.
    '''
    gripper_pose = PoseStamped()
    gripper_pose.header.frame_id = "base_footprint"
    gripper_pose.pose.position.x = 0.4753863391864514
    gripper_pose.pose.position.y = 0.03476345653124885
    gripper_pose.pose.position.z = 0.6746350873056409
    gripper_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -1.5708))
    return gripper_pose

def create_marker(name):
    '''
    Return an empty marker for a cube list.
    '''
    marker = Marker()
    marker.header.frame_id = "base_footprint"
    marker.ns = name
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.points = []
    marker.colors = []
    marker.scale = Vector3(0.06, 0.06, 0.06)
    return marker

def main():
    rospy.init_node("workspace_visualizer")
    WorkspaceVisualizerV2()

if __name__ == "__main__":
    main()
