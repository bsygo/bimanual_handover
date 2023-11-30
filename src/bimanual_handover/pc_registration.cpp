#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/io.h>
#include <bimanual_handover_msgs/AggregatePC.h>

sensor_msgs::PointCloud2 aggregated_pc;

void return_pc(bimanual_handover_msgs::AggregatedPC::Request &req, bimanual_handover_msgs::AggregatedPC::Response &res){
    res.pc = aggregated_pc;
    aggregated_pc = sensor_msgs::PointCloud2();
}

void aggregate_pc(const sensor_msgs::PointCloud2ConstPtr& new_cloud){
}

sensor_msgs::PointCloud2 registration(const sensor_msgs::PointCloud2ConstPtr& cloud1, const sensor_msgs::PointCloud2ConstPtr& cloud2){
    sensor_msgs::PointCloud2 cloud_out;
    pcl::PCLPointCloud2 pcl_cloud1;
    pcl::PCLPointCloud2 pcl_cloud2;
    pcl::PCLPointCloud2 pcl_cloud_out;
    pcl_conversions::toPCL(*cloud1, pcl_cloud1);
    pcl_conversions::toPCL(*cloud2, pcl_cloud2);
    bool success = pcl::concatenate(pcl_cloud1, pcl_cloud2, pcl_cloud_out);
    pcl_conversions::moveFromPCL(pcl_cloud_out, cloud_out);
    return cloud_out;
}

int main(int argc, char *+argv){
    ros::init(argc, argv, "pc_registration");
    ros::NodeHandle n;
    ros::Subscriber pc_sub = n.subscribe("pc_filtered", 10, aggregate_pc);
    ros::ServiceServer aggregated_srv = n.advertiseService("pc_aggregated", return_pc);
    ros::spin();
    return 0;
}
