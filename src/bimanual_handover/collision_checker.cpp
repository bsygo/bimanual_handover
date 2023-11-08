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
#include <bimanual_handover_msgs/CollisionChecking.h>

class CollisionDetector{
public:
    planning_scene::PlanningScene* ps;
    collision_detection::AllowedCollisionMatrix* acm;

    CollisionDetector(){
        // Initialize new planning scene in which to check for collisions
        ros::NodeHandle handle = ros::NodeHandle();
        robot_model_loader::RobotModelLoader loader = robot_model_loader::RobotModelLoader();
        moveit::core::RobotModelConstPtr robot_model = loader.getModel();
        ps = new planning_scene::PlanningScene(robot_model);
        ps->setName("collision_detection_ps");

        // Spawn initial gripper attached collision object at 0, 0, 0
        moveit_msgs::CollisionObject gripper;
        gripper.header.frame_id = "base_footprint";
        gripper.id = "gripper";
        geometry_msgs::Pose gripper_pose;
        gripper_pose.position.x = 0;
        gripper_pose.position.y = 0;
        gripper_pose.position.z = 0;
        gripper.pose = gripper_pose;
        gripper.operation = moveit_msgs::CollisionObject::ADD;
        shape_msgs::SolidPrimitive gripper_primitive;
        gripper_primitive.type = shape_msgs::SolidPrimitive::BOX;
        // Change to proper dimensions
        gripper_primitive.dimensions = std::vector<double>{1, 1, 1};
        gripper.primitives = std::vector<shape_msgs::SolidPrimitive>{gripper_primitive};
        moveit_msgs::AttachedCollisionObject attached_gripper;
        attached_gripper.object = gripper;
        attached_gripper.link_name = "base_footprint";
        ps->processAttachedCollisionObjectMsg(attached_gripper);

        // Spawn initial shadow hand collision object at 0, 0, 0
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
        // Change to proper dimensions
        sh_primitive.dimensions = std::vector<double>{1, 1, 1};
        sh.primitives = std::vector<shape_msgs::SolidPrimitive>{sh_primitive};
        ps->processCollisionObjectMsg(sh);

        // Allow collisions with the robots links to speed up collision checking
        // Only check collisions between the two collision objects
        std::vector<std::string> link_names = robot_model->getLinkModelNames();
        acm = &ps->getAllowedCollisionMatrixNonConst();
        for(auto i = link_names.begin(); i != link_names.end(); ++i){
            acm->setEntry("gripper", *i, true);
            acm->setEntry("sh", *i, true);
        }

        // Provide Service to move collision objects
        ros::ServiceServer collision_service = handle.advertiseService("collision_service", &CollisionDetector::collision_checking, this);
        ros::spin();
    }

    // Move the initially created gripper attached collision object into the provided pose
    void move_gripper(geometry_msgs::Pose new_pose){
        moveit_msgs::AttachedCollisionObject new_gripper;
        new_gripper.object.id = "gripper";
        new_gripper.object.operation = moveit_msgs::CollisionObject::MOVE;
        new_gripper.object.pose = new_pose;
        ps->processAttachedCollisionObjectMsg(new_gripper);
    }

    // Move the initially created shadow hand collision object into the provided pose
    void move_sh(geometry_msgs::Pose new_pose){
        moveit_msgs::CollisionObject new_sh;
        new_sh.id = "sh";
        new_sh.operation = moveit_msgs::CollisionObject::MOVE;
        new_sh.pose = new_pose;
        ps->processCollisionObjectMsg(new_sh);
    }

    // Receive the collision request, move collision objects into provided poses and check for collisions
    bool collision_checking(bimanual_handover_msgs::CollisionChecking::Request &req, bimanual_handover_msgs::CollisionChecking::Response &res){
        this->move_sh(req.sh_pose);
        this->move_gripper(req.gripper_pose);
        const collision_detection::CollisionRequest col_req = collision_detection::CollisionRequest();
        collision_detection::CollisionResult col_res = collision_detection::CollisionResult();
        ps->checkCollision(col_req, col_res, ps->getCurrentState(), *acm);
        res.collision = col_res.collision; 
        return true;
    }
};

int main(int argc, char **argv){
    ros::init(argc, argv, "collision_checking");
    CollisionDetector cd = CollisionDetector();
}
