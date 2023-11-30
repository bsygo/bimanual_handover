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
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

ros::Publisher cropped_pub;
moveit::planning_interface::MoveGroupInterface *left_arm;
tf2_ros::Buffer *tfBuffer;
tf2_ros::TransformListener *tfListener;

void cropPC(const sensor_msgs::PointCloud2ConstPtr& input_cloud){
    ros::AsyncSpinner spinner(1);
    spinner.start();
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input_cloud, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::CropBox<pcl::PointXYZRGB> box;
    geometry_msgs::PoseStamped gripper_pose = left_arm->getCurrentPose(); 
    geometry_msgs::PoseStamped transformed_pose;
    geometry_msgs::TransformStamped base_azure_transform = tfBuffer->lookupTransform("azure_kinect_rgb_camera_link", "base_footprint", ros::Time(0));
    tf2::doTransform(gripper_pose, transformed_pose, base_azure_transform);
    box.setMin(Eigen::Vector4f(transformed_pose.pose.position.x - 0.1, transformed_pose.pose.position.y - 0.1, transformed_pose.pose.position.z - 0.1, 1.0));
    box.setMax(Eigen::Vector4f(transformed_pose.pose.position.x + 0.1, transformed_pose.pose.position.y + 0.1, transformed_pose.pose.position.z + 0.1, 1.0));
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
    tfBuffer = new tf2_ros::Buffer();
    tfListener = new tf2_ros::TransformListener(*tfBuffer);
    // Call this way to set correct namespace
    moveit::planning_interface::MoveGroupInterface::Options opt("left_arm", "robot_description", ros::NodeHandle("/move_group"));
    left_arm = new moveit::planning_interface::MoveGroupInterface(opt);
    ros::Subscriber sub = n.subscribe("pc_raw", 10, cropPC);
    cropped_pub = n.advertise<sensor_msgs::PointCloud2>("pc_cropped", 10, true);
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
}
