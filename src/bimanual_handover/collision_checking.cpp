#include <ros/ros.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/CollisionObject.h>
#include <moveit_msgs/AttachedCollisionObject.h>
#include <moveit_msgs/RobotState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <Eigen/Geometry>
#include <iostream>

class CollisionDetector{
public:
    planning_scene::PlanningScene* ps;

    CollisionDetector(geometry_msgs::Pose gripper_pose){
        ros::NodeHandle handle = ros::NodeHandle();
        robot_model_loader::RobotModelLoader loader = robot_model_loader::RobotModelLoader();
        moveit::core::RobotModelConstPtr robot_model = loader.getModel();
        ps = new planning_scene::PlanningScene(robot_model);

        moveit_msgs::CollisionObject gripper;
        gripper.header.frame_id = "base_footprint";
        gripper.id = "gripper";
        gripper.pose = gripper_pose;
        gripper.operation = moveit_msgs::CollisionObject::ADD;
        shape_msgs::SolidPrimitive gripper_primitive;
        gripper_primitive.type = shape_msgs::SolidPrimitive::BOX;
        gripper_primitive.dimensions = std::vector<double>{1, 1, 1};
        gripper.primitives = std::vector<shape_msgs::SolidPrimitive>{gripper_primitive};
        moveit_msgs::AttachedCollisionObject attached_gripper;
        attached_gripper.object = gripper;
        attached_gripper.link_name = "base_footprint";

        ps->processAttachedCollisionObjectMsg(attached_gripper);

        moveit_msgs::CollisionObject sh;
        sh.header.frame_id = "base_footprint";
        sh.id = "sh";
        geometry_msgs::Pose sh_pose;
        sh_pose.position.x = 0;
        sh_pose.position.y = 0;
        sh_pose.position.z = 0;
        sh.pose = sh_pose;
        sh.operation = moveit_msgs::CollisionObject::ADD;
        shape_msgs::SolidPrimitive sh_primitive;
        sh_primitive.type = shape_msgs::SolidPrimitive::BOX;
        sh_primitive.dimensions = std::vector<double>{1, 1, 1};
        sh.primitives = std::vector<shape_msgs::SolidPrimitive>{sh_primitive};

        ps->processCollisionObjectMsg(sh);

        std::vector<std::string> link_names = robot_model->getLinkModelNames();
        collision_detection::AllowedCollisionMatrix acm = ps->getAllowedCollisionMatrixNonConst();
//        for(auto i = link_names.begin(); i != link_names.end(); ++i){
//            acm.setEntry("gripper", *i, true);
//            acm.setEntry("sh", *i, true);
//        }
    }

    void move_sh(geometry_msgs::Pose new_pose){
        std::cout << "test";
        moveit_msgs::CollisionObject new_sh;
        new_sh.id = "sh";
        new_sh.operation = moveit_msgs::CollisionObject::MOVE;
        new_sh.pose = new_pose;
        ps->processCollisionObjectMsg(new_sh);
    }

    bool collision_checking(geometry_msgs::Pose sh_pose){
        this->move_sh(sh_pose);
        return ps->isStateColliding("gripper");
    }
};

int main(int argc, char **argv){
    ros::init(argc, argv, "collision_checking");
    geometry_msgs::Pose gripper;
    gripper.position.x = 0;
    gripper.position.y = 0;
    gripper.position.z = 0;
    CollisionDetector cd = CollisionDetector(gripper);
    geometry_msgs::Pose sh;
    sh.position.x = 40;
    sh.position.y = 0;
    sh.position.z = 0;
    std::cout << cd.collision_checking(sh) << std::endl;
    cd.ps->printKnownObjects();
    std::vector<std::string> colliding_links;
    cd.ps->getCollidingLinks(colliding_links);
    for(auto i = colliding_links.begin(); i != colliding_links.end(); ++i){
        std::cout << *i;
    }
    //ros::spin();
}
