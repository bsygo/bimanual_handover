#include <ros/ros.h>
#include <ros/console.h>
#include <sensor_msgs/PointCloud2.h>
#include <iostream>

#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <moveit/move_group_interface/move_group_interface.h>

ros::Publisher cropped_pub;
moveit::planning_interface::MoveGroupInterface left_arm("left_gripper");

void crop_pc(const sensor_msgs::PointCloud2ConstPtr& input_cloud){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input_cloud, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::CropBox<pcl::PointXYZRGB> box;
    geometry_msgs::PoseStamped gripper_pose = left_arm.getCurrentPose(); 
    box.setMin(Eigen::Vector4f(gripper_pose.pose.position.x - 0.1, gripper_pose.pose.position.y - 0.1, gripper_pose.pose.position.z - 0.1, 1.0));
    box.setMax(Eigen::Vector4f(gripper_pose.pose.position.x + 0.1, gripper_pose.pose.position.y + 0.1, gripper_pose.pose.position.z + 0.1, 1.0));
    box.setInputCloud(cloud);
    box.filter(*cloud_filtered);
    pcl::toPCLPointCloud2(*cloud_filtered, pcl_pc2);
    sensor_msgs::PointCloud2 cloud_filtered_msg;
    pcl_conversions::moveFromPCL(pcl_pc2, cloud_filtered_msg);
    cloud_filtered_msg.header.stamp = ros::Time::now();
    cropped_pub.publish(cloud_filtered_msg);
}
int main(int argc, char **argv){
    ros::init(argc, argv, "pc_cropping");
    ros::NodeHandle n;
    cropped_pub = n.advertise<sensor_msgs::PointCloud2>("pc_cropped", 10, true);
    ros::Subscriber sub = n.subscribe("pc_raw", 10, crop_pc);
    ros::spin();
    return 0;
}
