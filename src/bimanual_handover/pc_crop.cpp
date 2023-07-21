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

ros::Publisher cropped_pub;


void crop_pc(const sensor_msgs::PointCloud2ConstPtr& input_cloud){
    pcl::PCLPointCloud2 pcl_pc2;
    pcl_conversions::toPCL(*input_cloud, pcl_pc2);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromPCLPointCloud2(pcl_pc2, *cloud);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    //pcl::PassThrough<pcl::PointXYZRGB> pass;
    //pass.setInputCloud(cloud);
    //pass.setFilterFieldName("z");
    //pass.setFilterLimits(0.0,1.0);
    //pass.filter(*cloud_filtered);
    pcl::CropBox<pcl::PointXYZRGB> box;
    box.setMin(Eigen::Vector4f(-0.3, -0.3, 0.0, 1.0));
    box.setMax(Eigen::Vector4f(0.3, 0.3, 1.0, 1.0));
    box.setInputCloud(cloud);
    box.filter(*cloud_filtered);
    pcl::toPCLPointCloud2(*cloud_filtered, pcl_pc2);
    sensor_msgs::PointCloud2 cloud_filtered_msg;
    pcl_conversions::moveFromPCL(pcl_pc2, cloud_filtered_msg);
    cropped_pub.publish(cloud_filtered_msg);
}
int main(int argc, char **argv){
    ros::init(argc, argv, "pc_cropping");
    ros::NodeHandle n;
    cropped_pub = n.advertise<sensor_msgs::PointCloud2>("handover/pc/pc_cropped", 10, true);
    ros::Subscriber sub = n.subscribe("/cloud_pcd", 10, crop_pc);
    ros::spin();
    return 0;
}
