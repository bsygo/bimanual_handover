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

class WorkspaceVisualizer():

    def __init__(self):

        # Initialize datafields of current loaded bag
        self.all_score = None
        self.all_results = None
        self.combined = None
        self.min_score = None
        self.score = None
        self.hand_markers = None
        self.hand_points = None
        self.hand_colors = None
        self.gripper_markers = None
        self.gripper_points = None
        self.gripper_colors = None

        self.recolor = True

        self.bag_path = rospkg.RosPack().get_path('bimanual_handover') + "/data/bags/"

        self.load_bag_sub = rospy.Subscriber("workspace_visualizer/load_bag", String, self.load_bag)
        self.layer_sub = rospy.Subscriber("workspace_visualizer/set_layers", Layers, self.publish_layers)
        self.layer_pub = rospy.Publisher("workspace_visualizer/pub_layers", Marker, queue_size = 1, latch = True)
        self.volume_sub = rospy.Subscriber("workspace_visualizer/set_volume", Volume, self.publish_volume)
        self.volume_pub = rospy.Publisher("workspace_visualizer/pub_volume", Marker, queue_size = 1, latch = True)
        self.intersection_sub = rospy.Subscriber("workspace_visualizer/set_intersection", String, self.publish_intersection)
        self.intersection_pub = rospy.Publisher("workspace_visualizer/pub_intersection", Marker, queue_size = 1, latch = True)
        self.analyse_rotations_sub = rospy.Subscriber("workspace_visualizer/analyse_rotations", String, self.analyse_rotations)

        rospy.loginfo("Workspace visualizer ready.")
        rospy.spin()

    def publish_layers(self, msg):
        data_type = msg.data_type
        direction = msg.direction
        selected_layers = msg.selected_layers.data

        # Set desired data type
        if data_type == "combined":
            data = self.combined
        elif data_type == "min_score":
            data = self.min_score
        elif data_type == "score":
            data = self.score

        # Seperate data into layers
        layers = {}
        for i in range(len(data.points)):
            if direction == 'x':
                if not data.points[i].x in layers.keys():
                    layers[data.points[i].x] = [i]
                else:
                    layers[data.points[i].x].append(i)
            elif direction == 'y':
                if not data.points[i].y in layers.keys():
                    layers[data.points[i].y] = [i]
                else:
                    layers[data.points[i].y].append(i)
            elif direction == 'z':
                if not data.points[i].z in layers.keys():
                    layers[data.points[i].z] = [i]
                else:
                    layers[data.points[i].z].append(i)
        layer_indices = sorted(layers.keys())

        # Fill marker msg with requested layers
        marker = create_marker("layers")
        for selected_layer in selected_layers:
            for i in layers[layer_indices[selected_layer]]:
                marker.points.append(data.points[i])
                marker.colors.append(data.colors[i])

        # Publish marker
        self.layer_pub.publish(marker)
        rospy.loginfo("Requested layers published.")

    def publish_volume(self, msg):
        data_type = msg.data_type
        threshold = msg.threshold
        mode = msg.mode

        # Set desired data type
        if data_type == "combined":
            points = self.combined
            if self.recolor:
                colors = self.recolor_combined()
            else:
                colors = points.colors
            # 343 - value because during data collection, 1 meant failure and 0
            # success by 343 tested values and data is just sum of values
            #matplotlib.use('Qt5Agg')
            inverted_data = [343.0 - data for data in self.all_results.data]
            plt.hist(inverted_data, bins = int(max(inverted_data)), range=(0, max(inverted_data)))
            plt.show()
            if mode == "greater":
                indices = [i for i in range(len(inverted_data)) if inverted_data[i] > threshold]
            elif mode == "smaller":
                indices = [i for i in range(len(inverted_data)) if inverted_data[i] < threshold]
        elif data_type == "min_score":
            points = self.min_score
            colors = points.colors
            plt.hist(self.all_min_scores.data, bins = 100, range=(0, 1.0))
            plt.show()
            if mode == "greater":
                indices = [i for i in range(len(self.all_min_scores.data)) if self.all_min_scores.data[i] > threshold]
            elif mode == "smaller":
                indices = [i for i in range(len(self.all_min_scores.data)) if self.all_min_scores.data[i] < threshold]
        elif data_type == "score":
            points = self.score
            if self.recolor:
                colors, score_data = self.recolor_avg_score()
            else:
                colors = points.colors
                score_data = self.all_svg_scores.data
            plt.hist(score_data, bins = 100, range=(0, 1.0))
            plt.show()
            if mode == "greater":
                indices = [i for i in range(len(score_data)) if score_data[i] > threshold]
            elif mode == "smaller":
                indices = [i for i in range(len(score_data)) if score_data[i] < threshold]

        # Fill marker msg with requested volume
        marker = create_marker("volume")
        for index in indices:
            marker.points.append(points.points[index])
            marker.colors.append(colors[index])

        rospy.loginfo("Fraction of tested solutions: {}".format(len(marker.points)/len(points.points)))

        # Publish marker
        self.volume_pub.publish(marker)
        rospy.loginfo("Requested volume published.")

    def publish_intersection(self, req):
        points = self.min_score
        colors = points.colors
        combined_indices = [i for i in range(len(self.all_results.data)) if 343 - self.all_results.data[i] > 75.0]
        min_score_indices = [i for i in range(len(self.all_min_scores.data)) if self.all_min_scores.data[i] < 0.19]
        _, score_data = self.recolor_avg_score()
        avg_score_indices = [i for i in range(len(score_data)) if score_data[i] < 0.32]

        indices = []
        for index in combined_indices:
            if (index in min_score_indices) and (index in avg_score_indices):
                indices.append(index)

        # Fill marker msg with requested volume
        marker = create_marker("volume")
        for index in indices:
            marker.points.append(points.points[index])
            marker.colors.append(colors[index])

        self.print_steps(indices)
        rospy.loginfo(indices)

        rospy.loginfo("Fraction of tested solutions: {}".format(len(marker.points)/len(points.points)))

        # Publish marker
        self.volume_pub.publish(marker)
        rospy.loginfo("Requested volume published.")

    def print_steps(self, indices):
        index_step_dict = {}
        current_index = 0
        for x in range(-5, 3):
            for y in range(-4, 9):
                for z in range(-6, 5):
                    index_step_dict[current_index] = [x, y, z]
                    current_index += 1

        min_values = [10, 10, 10]
        min_values_count = [0, 0, 0]
        max_values = [-10, -10, -10]
        max_values_count = [0, 0, 0]
        for index in indices:
            rospy.loginfo(index_step_dict[index])
            if index_step_dict[index][0] < min_values[0]:
                min_values[0] = index_step_dict[index][0]
                min_values_count[0] = 0
            if index_step_dict[index][0] == min_values[0]:
                min_values_count[0] += 1
            if index_step_dict[index][0] > max_values[0]:
                max_values[0] = index_step_dict[index][0]
                max_values_count[0] = 0
            if index_step_dict[index][0] == max_values[0]:
                max_values_count[0] += 1
            if index_step_dict[index][1] < min_values[1]:
                min_values[1] = index_step_dict[index][1]
                min_values_count[1] = 0
            if index_step_dict[index][1] == min_values[1]:
                min_values_count[1] += 1
            if index_step_dict[index][1] > max_values[1]:
                max_values[1] = index_step_dict[index][1]
                max_values_count[1] = 0
            if index_step_dict[index][1] == max_values[1]:
                max_values_count[1] += 1
            if index_step_dict[index][2] < min_values[2]:
                min_values[2] = index_step_dict[index][2]
                min_values_count[2] = 0
            if index_step_dict[index][2] == min_values[2]:
                min_values_count[2] += 1
            if index_step_dict[index][2] > max_values[2]:
                max_values[2] = index_step_dict[index][2]
                max_values_count[2] = 0
            if index_step_dict[index][2] == max_values[2]:
                max_values_count[2] += 1

        rospy.loginfo("Min steps: {}".format(min_values))
        rospy.loginfo("Min steps counts: {}".format(min_values_count))
        rospy.loginfo("Max steps: {}".format(max_values))
        rospy.loginfo("Max steps counts: {}".format(max_values_count))

    def recolor_combined(self):
        max_results = 0
        colors = []
        values = []
        for index in range(len(self.all_results.data)):
            value = 343 - self.all_results.data[index]
            if value > max_results:
                max_results = value
            values.append(value)
        for value in values:
            normalized_value = value/max_results
            colors.append(ColorRGBA(1 - normalized_value, normalized_value, 0, 1))
        return colors

    def recolor_avg_score(self):
        colors = []
        data = []
        for index in range(len(self.all_avg_scores.data)):
            unnormalized_data = self.all_avg_scores.data[index] * 343
            unnormalized_data = unnormalized_data - self.all_results.data[index]
            if unnormalized_data == 0:
                colors.append(ColorRGBA(1, 0, 0, 1))
                data.append(1.0)
            else:
                normalized_data = unnormalized_data/(343 - self.all_results.data[index])
                colors.append(ColorRGBA(normalized_data, 1 - normalized_data, 0, 1))
                data.append(normalized_data)
        return colors, data

    def correct_all_results_values(self):
        # Due to a bag during data collection, when the first transform checked
        # for each handover point was invalid, 2 instead of 1 was added to
        # all_results. This function corrects this mistake
        all_results = list(self.all_results.data)
        for i in range(len(all_results)):
            if self.hand_colors[i][0].r == 1.0:
                all_results[i] = all_results[i] - 1
        self.all_results.data = all_results

    def analyse_rotations(self, req):
        rotation_step_dict = {}
        current_index = 0
        for x in range(-3, 4):
            for y in range(-3, 4):
                for z in range(-3, 4):
                    rotation_step_dict[current_index] = [x, y, z]
                    current_index += 1

        successful_rotations = [0 for i in range(len(self.hand_colors[0]))]
        for value in self.hand_colors.values():
            for index in range(len(value)):
                if value[index].g == 1.0:
                    successful_rotations[index] += 1
        useless_rotations_indices = []
        for i in range(len(successful_rotations)):
            if successful_rotations[i] == 0:
                useless_rotations_indices.append(i)
        rospy.loginfo("Indices of useless rotations: {}".format(useless_rotations_indices))
        rospy.loginfo("Number of rotations without results: {}".format(len(useless_rotations_indices)))
        useless_rotations_list = []
        usefull_rotations_list = []
        for key in rotation_step_dict.keys():
            if key in useless_rotations_indices:
                useless_rotations_list.append(rotation_step_dict[key])
            else:
                usefull_rotations_list.append(rotation_step_dict[key])
        rospy.loginfo("Useless rotations list: {}".format(useless_rotations_list))
        rospy.loginfo("Usefull rotations list: {}".format(usefull_rotations_list))

    def group_points_and_colors_from_markers(self, markers):
        point_dict = {}
        color_dict = {}
        for i in range(int(len(markers.points)/343)):
            point_dict[i] = markers.points[i * 343 : (i + 1) * 343]
            color_dict[i] = markers.colors[i * 343 : (i + 1) * 343]
        return point_dict, color_dict

    def load_bag(self, bag_name):
        bag = rosbag.Bag(self.bag_path + bag_name.data)
        for topic, msg, t in bag.read_messages():
            if topic == "all_avg_scores":
                self.all_avg_scores = msg
            elif topic == "all_min_scores":
                self.all_min_scores = msg
            elif topic == "all_results":
                self.all_results = msg
            elif topic == "combined":
                self.combined = msg
            elif topic == "min_score":
                self.min_score = msg
            elif topic == "score":
                self.score = msg
            elif topic == "hand":
                self.hand_markers = msg
                self.hand_points, self.hand_colors = self.group_points_and_colors_from_markers(self.hand_markers)
            elif topic == "gripper":
                self.gripper_markers = msg
                self.gripper_points, self.gripper_colors = self.group_points_and_colors_from_markers(self.gripper_markers)
        bag.close()
        #self.correct_all_results_values()
        rospy.loginfo("Loaded bag: {}".format(bag_name.data))

class WorkspaceVisualizerV2():

    def __init__(self):
        self.data = None
        self.pkg_path = rospkg.RosPack().get_path('bimanual_handover')
        self.time = datetime.now().strftime("%d_%m_%Y_%H_%M")

        self.load_json_sub = rospy.Subscriber("workspace_visualizer/load_json", String, self.load_json)
        self.combine_jsons_sub = rospy.Subscriber("workspace_visualizer/combine_jsons", String, self.combine_jsons)
        self.print_min_max_sub = rospy.Subscriber("workspace_visualizer/print_min_max", String, self.print_min_max_values)
        self.cut_data_sub = rospy.Subscriber("workspace_visualizer/cut_data", Int64, self.cut_data)
        self.write_bag_sub = rospy.Subscriber("workspace_visualizer/write_bag", Int64, self.write_transforms_to_rosbag_receiver)
        self.plot_transform_data_sub = rospy.Subscriber("workspace_visualizer/plot_transform_data", Bool, self.plot_transform_data)
        self.write_intersection_bag_sub = rospy.Subscriber("workspace_visualizer/write_intersection_bag", Int64, self.write_intersection_to_rosbag)
        #self.layer_sub = rospy.Subscriber("workspace_visualizer/set_layers", Layers, self.publish_layers)
        #self.layer_pub = rospy.Publisher("workspace_visualizer/pub_layers", Marker, queue_size = 1, latch = True)
        self.volume_sub = rospy.Subscriber("workspace_visualizer/set_volume", Volume, self.publish_volume)
        self.volume_pub = rospy.Publisher("workspace_visualizer/pub_volume", Marker, queue_size = 1, latch = True)
        self.intersection_sub = rospy.Subscriber("workspace_visualizer/set_intersection", Int64, self.publish_intersection)
        self.intersection_pub = rospy.Publisher("workspace_visualizer/pub_intersection", Marker, queue_size = 1, latch = True)

        rospy.loginfo("VisualizerV2 ready.")
        rospy.spin()

    def load_json(self, file_name):
        self.data = TransformHandler.load_independent(self.pkg_path + "/data/workspace_analysis/" + file_name.data)
        rospy.loginfo("Data from file {} loaded.".format(file_name.data))

    def combine_jsons(self, filename):
        # Load first file with load_json and call this with second file to
        # combine
        second_data = TransformHandler.load_independent(self.pkg_path + "/data/workspace_analysis/" + filename.data)
        rospy.loginfo("Data from file {} loaded.".format(filename.data))
        for key, value in second_data.items():
            self.data[key] = value
        filepath = self.pkg_path + "/data/workspace_analysis/workspace_analysis_" + self.time + ".json"
        TransformHandler.save_independent(self.data, filepath)
        rospy.loginfo("Saved combined data.")

    def print_min_max_values(self, msg):
        min_max_values = {}
        min_max_values["x_min"] = 1000
        min_max_values["y_min"] = 1000
        min_max_values["z_min"] = 1000
        min_max_values["x_max"] = -1000
        min_max_values["y_max"] = -1000
        min_max_values["z_max"] = -1000
        for transform in self.data.values():
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
        rospy.loginfo("x min: {}".format(min_max_values["x_min"]))
        rospy.loginfo("y min: {}".format(min_max_values["y_min"]))
        rospy.loginfo("z min: {}".format(min_max_values["z_min"]))
        rospy.loginfo("x max: {}".format(min_max_values["x_max"]))
        rospy.loginfo("y max: {}".format(min_max_values["y_max"]))
        rospy.loginfo("z max: {}".format(min_max_values["z_max"]))

    def get_transforms_layers(self):
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
        # Threshold is maximum number of failed solution allowed
        threshold = threshold.data
        x_layers, y_layers, z_layers = self.get_transforms_layers()
        cut_data = deepcopy(self.data)
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
        filepath = self.pkg_path + "/data/workspace_analysis/workspace_analysis_" + self.time + ".json"
        TransformHandler.save_independent(cut_data, filepath)
        rospy.loginfo("Saved cut data.")

    def write_transforms_to_rosbag_receiver(self, threshold):
        self.write_transforms_to_rosbag(threshold.data)

    def write_transforms_to_rosbag(self, threshold, data = None):
        # Threshold is maximum number of failed solution allowed
        path = self.pkg_path + "/data/workspace_analysis/"
        bag = rosbag.Bag('{}workspace_analysis_{}.bag'.format(path, self.time), 'w')
        if data is None:
            data = self.data.values()
        for transform in data:
            if transform.number_solutions <= threshold:
                msgs = transform.get_transform_msgs()
                for i in range(len(msgs)):
                    if transform.scores[i] < 1.0:
                        bag.write('transforms', msgs[i])
        bag.close()
        rospy.loginfo("Transforms written into rosbag.")

    def get_data_as_lists(self, ignore_invalid_transforms = True):
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

    def get_percentage_cutoff(self, percentage):
        number_transforms = len(self.data.keys())
        cutoff = int(number_transforms/100 * percentage)
        number_solutions_data, min_score_data, avg_score_data = self.get_data_as_lists()

        number_solutions_data.sort(reverse = True)
        min_score_data.sort()
        avg_score_data.sort()

        rospy.loginfo("The best {} percent of transforms have equal or above {} solutions.".format(percentage, number_solutions_data[cutoff]))
        rospy.loginfo("The best {} percent of transforms have equal or above {} min_score.".format(percentage, min_score_data[cutoff]))
        rospy.loginfo("The best {} percent of transforms have equal or above {} avg_score.".format(percentage, avg_score_data[cutoff]))

        return number_solutions_data[cutoff], min_score_data[cutoff], avg_score_data[cutoff]

    def plot_transform_data(self, msg):
        number_solutions_data, min_score_data, avg_score_data = self.get_data_as_lists(msg.data)

        plt.hist(number_solutions_data, bins = int(max(number_solutions_data)), range=(0, max(number_solutions_data)))
        plt.show()

        plt.hist(min_score_data, bins = 100, range=(0, 1))
        plt.show()

        plt.hist(avg_score_data, bins = 100, range=(0, 1))
        plt.show()

    def get_intersection_data(self, percentage):
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
        filtered_transforms, _ = self.get_intersection_data(percentage.data)
        self.write_transforms_to_rosbag(343, filtered_transforms)

    def publish_intersection(self, percentage):
        filtered_transforms, colors = self.get_intersection_data(percentage.data)
        markers = self.create_markers(filtered_transforms, colors)
        self.intersection_pub.publish(markers)
        rospy.loginfo("Intersection volume published.")

    def publish_volume(self, msg):
        data_type = msg.data_type
        threshold = msg.threshold
        mode = msg.mode

        # Ignore invalid solutions for avg_score
        if data_type == "avg_score":
            recalculated_data = self.recalculate_avg_score()
        elif data_type == "number_solutions":
            filtered_inverted_values = []

        filtered_transforms = []
        colors = []
        for transform in self.data.values():
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

        if data_type == "number_solutions":
            max_value = max(filtered_inverted_values)
            for value in filtered_inverted_values:
                color_value = value/max_value
                colors.append(ColorRGBA(1 - color_value, color_value, 0, 1))

        markers = self.create_markers(filtered_transforms, colors)
        self.volume_pub.publish(markers)
        rospy.loginfo("Volume published.")

    def recalculate_avg_score(self):
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
        markers = create_marker("volume")
        markers.colors = colors
        gripper_pose = get_gripper_pose()
        for transform in transforms:
            markers.points.append(Point(gripper_pose.pose.position.x + transform.x * 0.06, gripper_pose.pose.position.y + transform.y * 0.06, gripper_pose.pose.position.z + transform.z * 0.06))
        return markers

def get_gripper_pose():
    gripper_pose = PoseStamped()
    gripper_pose.header.frame_id = "base_footprint"
    gripper_pose.pose.position.x = 0.4753863391864514
    gripper_pose.pose.position.y = 0.03476345653124885
    gripper_pose.pose.position.z = 0.6746350873056409
    gripper_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, -1.5708))
    return gripper_pose

def create_marker(name):
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
