#ifndef _ROBOT_RL_SIM_
#define _ROBOT_RL_SIM_

#include <rclcpp/rclcpp.hpp>
#include "robot_rl_sim/robot_rl_sdk.hpp"
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include "robot_rl_sim/observation_buffer.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"


class RobotRlSim : public RL, public rclcpp::Node
{
    public:
        RobotRlSim();
        ~RobotRlSim();
    
    private:
    //variables
        // history buffer
        ObservationBuffer history_obs_buf;
        torch::Tensor history_obs;


        // ROS2 entities
        rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
        rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
        rclcpp::TimerBase::SharedPtr run_model_timer_;
        rclcpp::TimerBase::SharedPtr control_timer_;
        rclcpp::TimerBase::SharedPtr keyboard_timer_;
        // Latest messages
        sensor_msgs::msg::Imu::SharedPtr latest_imu_msg_;
        sensor_msgs::msg::JointState::SharedPtr latest_joint_state_msg_;
        geometry_msgs::msg::Twist cmd_vel;
        rclcpp::Client<rcl_interfaces::srv::GetParameters>::SharedPtr param_client;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_publisher_;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr limit_joint_publisher_;

        // Mutex for thread-safety
        std::mutex imu_mutex_;
        std::mutex joint_state_mutex_;

    //functions
        torch::Tensor Forward();
        void RunModel();
        void JointController();
        void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
        void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
        void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
};

#endif