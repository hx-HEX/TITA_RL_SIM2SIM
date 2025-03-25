#include "robot_rl_sim/robot_rl_sim.hpp"
// robot_rl_sim.hpp
rclcpp::Time last_run_model_time_;
rclcpp::Time last_run_model_time_1;

RobotRlSim::RobotRlSim()
    : Node("robot_rl_sim")
{
    // get params from param_node
    param_client = this->create_client<rcl_interfaces::srv::GetParameters>("/param_node/get_parameters");
    while (!param_client->wait_for_service(std::chrono::seconds(1)))
    {
        if (!rclcpp::ok()) {
            std::cout  << "Interrupted while waiting for param_node service. Exiting." << std::endl;
            return;
        }
        std::cout  << "Waiting for param_node service to be available..." << std::endl;
    }
    auto request = std::make_shared<rcl_interfaces::srv::GetParameters::Request>();
    request->names = {"robot_name"};

    // Use a timeout for the future
    auto future = param_client->async_send_request(request);
    auto status = rclcpp::spin_until_future_complete(this->get_node_base_interface(), future, std::chrono::seconds(5));

    if (status == rclcpp::FutureReturnCode::SUCCESS)
    {
        auto result = future.get();
        this->robot_name = result->values[0].string_value;
        std::cout  << "Get param robot_name: " << this->robot_name << std::endl;
    }
    else
    {
        std::cout  << "Failed to call param_node service" << std::endl;
    }

    // read params from yaml
    this->ReadYaml(this->robot_name);
    for (std::string &observation : this->params.observations)
    {
        if (observation == "ang_vel")
        {
            observation = "ang_vel_body";
        }
    }

    // init rl
    torch::autograd::GradMode::set_enabled(false);
    if (this->params.observations_history.size() != 0)
    {
        this->history_obs_buf = ObservationBuffer(1, this->params.num_observations, this->params.observations_history.size());
    }
    this->InitObservations();
    this->InitOutputs();
    this->InitControl();

    // model
    std::string model_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + this->robot_name + "/" + this->params.model_name;
    std::cout << model_path << std::endl;
    this->model = torch::jit::load(model_path);

    // subscriber
    imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
        "/tita/imu_sensor_broadcaster/imu", 10,
        std::bind(&RobotRlSim::imuCallback, this, std::placeholders::_1));

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/tita/joint_states", 10,
        std::bind(&RobotRlSim::jointStateCallback, this, std::placeholders::_1));

    cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "/cmd_vel", 10,
        std::bind(&RobotRlSim::cmdVelCallback, this, std::placeholders::_1));

    // publisher
    effort_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/tita/effort_controller/commands", 10);

    action_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/tita/action", 10);

    limit_joint_publisher_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/tita/limit_joint", 10);

    // loop
    run_model_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(this->params.dt * this->params.decimation* 1000)), std::bind(&RobotRlSim::RunModel, this));

    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(this->params.dt* 1000)), std::bind(&RobotRlSim::JointController, this));
    
    keyboard_timer_= this->create_wall_timer(
        std::chrono::duration<double>(0.05), std::bind(&RobotRlSim::KeyboardInterface, this));

    //info
    std::cout << "RL_Sim start" << std::endl;
}

RobotRlSim::~RobotRlSim()
{
    this->run_model_timer_.reset();
    this->control_timer_.reset();
    this->keyboard_timer_.reset();
}

torch::Tensor RobotRlSim::Forward()
{
    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor clamped_obs = this->ComputeObservation();
    // std::cout << clamped_obs << std::endl;
    torch::Tensor actions;
    if (this->params.observations_history.size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.observations_history);
        actions = this->model.forward({this->history_obs}).toTensor();
    }
    else
    {
        actions = this->model.forward({clamped_obs}).toTensor();
    }

    if (this->params.clip_actions_upper.numel() != 0 && this->params.clip_actions_lower.numel() != 0)
    {
        return torch::clamp(actions, this->params.clip_actions_lower, this->params.clip_actions_upper);
    }
    else
    {
        return actions;
    }
}

double limit_position(double position) {
    const double TWO_PI = 2 * M_PI;  
    if(position > 0 || position == 0)
        position = position - std::trunc((position + TWO_PI) / (2*TWO_PI)) * 2*TWO_PI;
    else
        position = position - std::trunc((position - TWO_PI) / (2*TWO_PI)) * 2*TWO_PI;

    return position;
}

void RobotRlSim::RunModel()
{
    rclcpp::Time current_time = this->now();

    if (last_run_model_time_.nanoseconds() != 0)
    {
        // 计算时间间隔
        // rclcpp::Duration interval = current_time - last_run_model_time_;
        // RCLCPP_INFO(this->get_logger(), "RunModel Interval: %.3f ms", interval.seconds() * 1000);
    }
    last_run_model_time_ = current_time;
    
    // 获取最新的 IMU 和 JointState 消息
    sensor_msgs::msg::Imu::SharedPtr imu_msg;
    sensor_msgs::msg::JointState::SharedPtr joint_msg;

    {
        std::lock_guard<std::mutex> imu_lock(imu_mutex_);
        imu_msg = latest_imu_msg_;
    }

    {
        std::lock_guard<std::mutex> joint_lock(joint_state_mutex_);
        joint_msg = latest_joint_state_msg_;
    }
    // 检查是否接收到所有必要的消息
    if (!imu_msg || !joint_msg)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Waiting for both IMU and JointState messages.");
        return;std::cout << "forward" << std::endl;
    } 

    this->obs.ang_vel = torch::tensor({imu_msg->angular_velocity.x, 
        imu_msg->angular_velocity.y, 
        imu_msg->angular_velocity.z}).unsqueeze(0); 
    this->obs.commands = torch::tensor({{this->control.v, this->control.w, this->control.height}});
    this->obs.base_quat = torch::tensor({imu_msg->orientation.x, 
        imu_msg->orientation.y, 
        imu_msg->orientation.z,
        imu_msg->orientation.w}).unsqueeze(0);
    
    this->obs.dof_pos = torch::tensor(joint_msg->position).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
    // // 直接对具体索引位置赋值
    this->obs.dof_pos[0][3] = 0;
    this->obs.dof_pos[0][7] = 0;

    this->obs.dof_vel = torch::tensor(joint_msg->velocity).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
    // auto start_time = std::chrono::high_resolution_clock::now();
    torch::Tensor clamped_actions = this->Forward();
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto inference_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    // 打印推理时间（以毫秒为单位）
    // RCLCPP_INFO(this->get_logger(), "Inference time: %ld microseconds (%.3f ms)",
    //             inference_duration, inference_duration / 1000.0);

    this->obs.actions = clamped_actions;
    this->actons_filtered = this->last_actions * 0.2 + this->obs.actions * 0.8;
    this->last_actions = this->actons_filtered;

    torch::Tensor origin_output_torques = this->ComputeTorques(this->actons_filtered);
    this->output_torques = torch::clamp(origin_output_torques, -(this->params.torque_limits), this->params.torque_limits);
    this->output_dof_pos = this->ComputePosition(this->actons_filtered);

}

void RobotRlSim::JointController()
{
    rclcpp::Time current_time = this->now();

    if (last_run_model_time_1.nanoseconds() != 0)
    {
        // // 计算时间间隔
        // rclcpp::Duration interval = current_time - last_run_model_time_1;
        // RCLCPP_INFO(this->get_logger(), "JointController Interval: %.3f ms", interval.seconds() * 1000);
    }
    last_run_model_time_1 = current_time;

    sensor_msgs::msg::JointState::SharedPtr joint_msg;

    {
        std::lock_guard<std::mutex> joint_lock(joint_state_mutex_);
        joint_msg = latest_joint_state_msg_;
    }
    // 检查是否接收到所有必要的消息
    if (!joint_msg)
    {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, "Waiting for both IMU and JointState messages.");
        return;
    } 

    this->obs.dof_pos = torch::tensor(joint_msg->position).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);
    // // 直接对具体索引位置赋值
    this->obs.dof_pos[0][3] = 0;
    this->obs.dof_pos[0][7] = 0;

    this->obs.dof_vel = torch::tensor(joint_msg->velocity).narrow(0, 0, this->params.num_of_dofs).unsqueeze(0);

    torch::Tensor origin_output_torques = this->ComputeTorques(this->actons_filtered);
    this->output_torques = torch::clamp(origin_output_torques, -(this->params.torque_limits), this->params.torque_limits);

    std_msgs::msg::Float64MultiArray effort_msg, action_msg, joint_msg_;
    effort_msg.data.resize(this->params.num_of_dofs);
    action_msg.data.resize(this->params.num_of_dofs);
    joint_msg_.data.resize(this->params.num_of_dofs);
    
    // 确保 output_torques 是 double 类型
    torch::Tensor torques_cpu = this->output_torques.to(torch::kDouble).contiguous();
    torch::Tensor action_cpu = this->output_dof_pos.to(torch::kDouble).contiguous();
    
    // 获取指针
    double* torque_data = torques_cpu.data_ptr<double>();
    double* action_data = action_cpu.data_ptr<double>();
    
    // 复制数据
    for (size_t i = 0; i < this->params.num_of_dofs; ++i)
    {
        // if(i == 3 || i ==7)
        // {
        //     torque_data[i] = 10 * action_data[i] - 0.5 * joint_msg->velocity[i];
        // }
        effort_msg.data[i] = torque_data[i];
        action_msg.data[i] = action_data[i];
        joint_msg_.data[i] = joint_msg->position[i];
    }
    
    // 发布消息
    effort_publisher_->publish(effort_msg);
    action_publisher_->publish(action_msg);
    limit_joint_publisher_->publish(joint_msg_);
    
}

void RobotRlSim::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(imu_mutex_);
    this->latest_imu_msg_ = msg;
}

void RobotRlSim::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    std::vector<std::string> desired_order = {
        "joint_left_hip",
        "joint_left_thigh",
        "joint_left_calf",
        "joint_left_wheel",
        "joint_right_hip",
        "joint_right_thigh",
        "joint_right_calf",
        "joint_right_wheel"};

    std::vector<double> sorted_position(desired_order.size());
    std::vector<double> sorted_velocity(desired_order.size());
    
    for (size_t i = 0; i < desired_order.size(); ++i)
    {
        auto it = std::find(msg->name.begin(), msg->name.end(), desired_order[i]);
        if (it != msg->name.end())
        {
            size_t index = std::distance(msg->name.begin(), it);
            sorted_position[i] = msg->position[index];
            sorted_velocity[i] = msg->velocity[index];
        }
    }

    auto sorted_msg = std::make_shared<sensor_msgs::msg::JointState>();
    sorted_msg->header = msg->header;
    sorted_msg->name = desired_order;
    sorted_msg->position = sorted_position;
    sorted_msg->velocity = sorted_velocity;

    this->latest_joint_state_msg_ = sorted_msg;
    this->latest_joint_state_msg_->position[3] = limit_position(this->latest_joint_state_msg_->position[3]);
    this->latest_joint_state_msg_->position[7] = limit_position(this->latest_joint_state_msg_->position[7]);
}

void RobotRlSim::cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    this->cmd_vel = *msg;
    this->control.v = this->cmd_vel.linear.x;
    this->control.w = this->cmd_vel.linear.y;
    this->control.height = this->cmd_vel.angular.z;
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RobotRlSim>());
    rclcpp::shutdown();
    return 0;
}