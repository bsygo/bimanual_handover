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
#include <tf2_eigen/tf2_eigen.h>

ros::Publisher cropped_pub;
ros::Publisher final_pub;
ros::Publisher debug_pub;
ros::Publisher debug_gripper_pose_pub;
ros::Publisher debug_sample_pose_pub;
moveit::planning_interface::MoveGroupInterface *left_arm;
tf2_ros::Buffer *tfBuffer;
tf2_ros::TransformListener *tfListener;

tf2::Transform getSamplePoseTransform(){
    geometry_msgs::PoseStamped sample_pose;
    sample_pose.header.frame_id = "base_footprint";
    geometry_msgs::Point position;
    position.x = 0.4753863391864514;
    position.y = 0.03476345653124885;
    position.z = 0.6746350873056409;
    tf2::Quaternion rotation;
    rotation.setRPY(0, 0, -1.5708);
    geometry_msgs::Quaternion rotation_msg = tf2::toMsg(rotation);
    sample_pose.pose.position = position;
    sample_pose.pose.orientation = rotation_msg;
    debug_sample_pose_pub.publish(sample_pose);
    //ROS_INFO_STREAM(sample_pose);
    
    tf2::Transform sample_pose_transform;
    tf2::fromMsg(sample_pose.pose, sample_pose_transform);
    return sample_pose_transform;
}

void cropPC(const sensor_msgs::PointCloud2ConstPtr& input_cloud){
    ros::AsyncSpinner spinner(1);
    spinner.start();

    // Convert sensor_msgs::PointCloud2 into pcl::PointCloud2 for further use with pcl
    const pcl::PCLPointCloud2Ptr pcl_pc2(new pcl::PCLPointCloud2);
    pcl::PCLPointCloud2 filtered_pcl_pc2;
    pcl_conversions::toPCL(*input_cloud, *pcl_pc2);

    // Set parameters for CropBox filter and apply filter
    pcl::CropBox<pcl::PCLPointCloud2> box;
    geometry_msgs::TransformStamped gripper_azure_transform = tfBuffer->lookupTransform("azure_kinect_rgb_camera_link", "l_gripper_tool_frame", ros::Time(0));
    Eigen::Affine3d gripper_azure_transform_matrix_d = tf2::transformToEigen(gripper_azure_transform);
    Eigen::Affine3f gripper_azure_transform_matrix = gripper_azure_transform_matrix_d.cast <float> ();
    Eigen::Vector3f euler_rotation = gripper_azure_transform_matrix.rotation().matrix().eulerAngles(0, 1, 2);
    Eigen::Vector3f translation = gripper_azure_transform_matrix.translation();
    box.setTranslation(translation);
    box.setRotation(euler_rotation);
    box.setMin(Eigen::Vector4f(-0.1, -0.1, -0.3, 1));
    box.setMax(Eigen::Vector4f(0.1, 0.1, 0.0, 1));
    box.setInputCloud(pcl_pc2);
    box.filter(filtered_pcl_pc2);

    // Convert filtered pointcloud back to sensor_msgs::PointCloud2 and publish it
    sensor_msgs::PointCloud2 cloud_filtered_msg;
    pcl_conversions::moveFromPCL(filtered_pcl_pc2, cloud_filtered_msg);
    cloud_filtered_msg.header.stamp = ros::Time::now();
    debug_pub.publish(cloud_filtered_msg);
    cropped_pub.publish(cloud_filtered_msg);
}

void transformPC(const sensor_msgs::PointCloud2ConstPtr& input_cloud){
    ros::AsyncSpinner spinner(1);
    spinner.start();

    geometry_msgs::PoseStamped gripper_pose = left_arm->getCurrentPose(); 

    // Transform pointcloud to sampling pose
    tf2::Transform gripper_pose_transform;
    tf2::Transform sample_pose_transform;

    debug_gripper_pose_pub.publish(gripper_pose);
    //ROS_INFO_STREAM(gripper_pose);
    tf2::fromMsg(gripper_pose.pose, gripper_pose_transform);
    sample_pose_transform = getSamplePoseTransform();

    geometry_msgs::Pose sample_pose;
    tf2::toMsg(sample_pose_transform, sample_pose);
    tf2::Transform pc_transform = gripper_pose_transform.inverseTimes(sample_pose_transform);
    geometry_msgs::Transform pc_transform_msg = tf2::toMsg(pc_transform);

    //debug_pub.publish(input_cloud);
    sensor_msgs::PointCloud2 gripper_transformed_cloud_filtered_msg;
    geometry_msgs::TransformStamped base_gripper_transform_msg = tfBuffer->lookupTransform("l_gripper_tool_frame", "base_footprint", ros::Time(0));
    Eigen::Matrix4f base_gripper_transform_matrix;
    pcl_ros::transformAsMatrix(base_gripper_transform_msg.transform, base_gripper_transform_matrix);
    pcl_ros::transformPointCloud(base_gripper_transform_matrix, *input_cloud, gripper_transformed_cloud_filtered_msg);
    gripper_transformed_cloud_filtered_msg.header.frame_id = "l_gripper_tool_frame";
    
    sensor_msgs::PointCloud2 transformed_cloud_filtered_msg;
    Eigen::Matrix4f pc_transform_matrix;
    pcl_ros::transformAsMatrix(pc_transform_msg, pc_transform_matrix);
    pcl_ros::transformPointCloud(pc_transform_matrix, gripper_transformed_cloud_filtered_msg, transformed_cloud_filtered_msg);

    sensor_msgs::PointCloud2 base_transformed_cloud_filtered_msg;
    geometry_msgs::TransformStamped gripper_base_transform_msg = tfBuffer->lookupTransform("base_footprint", "l_gripper_tool_frame", ros::Time(0));
    Eigen::Matrix4f gripper_base_transform_matrix;
    pcl_ros::transformAsMatrix(gripper_base_transform_msg.transform, gripper_base_transform_matrix);
    pcl_ros::transformPointCloud(gripper_base_transform_matrix, transformed_cloud_filtered_msg, base_transformed_cloud_filtered_msg);
    base_transformed_cloud_filtered_msg.header.frame_id = "base_footprint";

    final_pub.publish(base_transformed_cloud_filtered_msg);
}


int main(int argc, char **argv){
    ros::init(argc, argv, "pc_cropping");
    ros::NodeHandle n;
    tfBuffer = new tf2_ros::Buffer();
    tfListener = new tf2_ros::TransformListener(*tfBuffer);
    // Call this way to set correct namespace
    moveit::planning_interface::MoveGroupInterface::Options opt("left_arm", "robot_description", ros::NodeHandle("/move_group"));
    left_arm = new moveit::planning_interface::MoveGroupInterface(opt);
    // Make sure state monitor is initialized
    ros::Duration(0.5).sleep();
    //geometry_msgs::PoseStamped gripper_pose = left_arm->getCurrentPose(); 
    ros::Subscriber raw_sub = n.subscribe("pc_raw", 10, cropPC);
    cropped_pub = n.advertise<sensor_msgs::PointCloud2>("pc_cropped", 10, true);
    ros::Subscriber filtered_sub = n.subscribe("pc_filtered", 10, transformPC);
    final_pub = n.advertise<sensor_msgs::PointCloud2>("pc_final", 10, true);
    debug_pub = n.advertise<sensor_msgs::PointCloud2>("debug/pc_cropped_orig", 10, true);
    debug_gripper_pose_pub = n.advertise<geometry_msgs::PoseStamped>("debug/pc_gripper_pose", 10, true);
    debug_sample_pose_pub = n.advertise<geometry_msgs::PoseStamped>("debug/pc_sample_pose", 10, true);
    ROS_INFO_STREAM("pc_crop node ready");
    ros::AsyncSpinner spinner(1);
    spinner.start();
    ros::waitForShutdown();
}
