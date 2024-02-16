#!/usr/bin/env python3

import rospy
import rospkg
import rosbag
import os
from visualization_msgs.msg import Marker
from std_msgs.msg import String, ColorRGBA
from bimanual_handover_msgs.msg import Layers, Volume
from geometry_msgs.msg import Vector3

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
            if mode == "greater":
                indices = [i for i in range(len(self.all_results.data)) if 343 - self.all_results.data[i] > threshold]
            elif mode == "smaller":
                indices = [i for i in range(len(self.all_results.data)) if 343 - self.all_results.data[i] < threshold]
        elif data_type == "min_score":
            points = self.min_score
            colors = points.colors
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
            if mode == "greater":
                indices = [i for i in range(len(score_data)) if score_data[i] > threshold]
            elif mode == "smaller":
                indices = [i for i in range(len(score_data)) if score_data[i] < threshold]

        # Fill marker msg with requested volume
        marker = create_marker("volume")
        for index in indices:
            marker.points.append(points.points[index])
            marker.colors.append(colors[index])

        # Publish marker
        self.volume_pub.publish(marker)
        rospy.loginfo("Requested volume published.")

    def publish_intersection(self, req):
        points = self.min_score
        colors = points.colors
        combined_indices = [i for i in range(len(self.all_results.data)) if 343 - self.all_results.data[i] > 80.0]
        min_score_indices = [i for i in range(len(self.all_min_scores.data)) if self.all_min_scores.data[i] < 0.21]
        _, score_data = self.recolor_avg_score()
        avg_score_indices = [i for i in range(len(score_data)) if score_data[i] < 0.33]

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
    WorkspaceVisualizer()

if __name__ == "__main__":
    main()
