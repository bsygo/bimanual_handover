#!/usr/bin/env python3

import rospy
import rospkg
import rosbag
import os
from visualization_msgs.msg import Marker
from std_msgs.msg import String
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

        self.bag_path = rospkg.RosPack().get_path('bimanual_handover') + "/data/bags/"

        self.load_bag_sub = rospy.Subscriber("workspace_visualizer/load_bag", String, self.load_bag)
        self.layer_sub = rospy.Subscriber("workspace_visualizer/set_layers", Layers, self.publish_layers)
        self.layer_pub = rospy.Publisher("workspace_visualizer/pub_layers", Marker, queue_size = 1, latch = True)
        self.volume_sub = rospy.Subscriber("workspace_visualizer/set_volume", Volume, self.publish_volume)
        self.volume_pub = rospy.Publisher("workspace_visualizer/pub_volume", Marker, queue_size = 1, latch = True)

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
            data = self.combined
            indices = [i for i in range(len(self.all_results)) if self.all_results[i] > threshold]
        elif data_type == "min_score":
            data = self.min_score
            indices = [i for i in range(len(self.all_min_scores)) if self.all_min_scores[i] > threshold]
        elif data_type == "score":
            data = self.score
            indices = [i for i in range(len(self.all_avg_scores)) if self.all_avg_scores[i] > threshold]

        # Fill marker msg with requested volume
        marker = create_marker("volume")
        for index in indices:
            marker.points.append(data.points[i])
            marker.colors.append(data.points[i])

        # Publish marker
        self.volume_pub.publish(marker)
        rospy.loginfo("Requested volume published.")

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
        bag.close()
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
