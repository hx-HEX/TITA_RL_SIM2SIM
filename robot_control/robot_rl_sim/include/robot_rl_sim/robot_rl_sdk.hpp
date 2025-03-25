#ifndef _ROBOT_RL_SDK_
#define _ROBOT_RL_SDK_

#include <torch/script.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <mutex>

#include <yaml-cpp/yaml.h>

struct Control
{
    double v = 0.0;
    double w = 0.0;
    double height = 0.0;
};

struct ModelParams
{
    std::string model_name;
    std::string framework;
    double dt;
    int decimation;
    int num_observations;
    std::vector<std::string> observations;
    std::vector<int> observations_history;
    double damping;
    double stiffness;
    double action_scale;
    double hip_scale_reduction;
    std::vector<int> hip_scale_reduction_indices;
    int num_of_dofs;
    double lin_vel_scale;
    double ang_vel_scale;
    double dof_pos_scale;
    double dof_vel_scale;
    double clip_obs;
    torch::Tensor clip_actions_upper;
    torch::Tensor clip_actions_lower;
    torch::Tensor torque_limits;
    torch::Tensor rl_kd;
    torch::Tensor rl_kp;
    torch::Tensor fixed_kp;
    torch::Tensor fixed_kd;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;
    std::vector<std::string> joint_controller_names;
};

struct Observations
{
    torch::Tensor lin_vel;
    torch::Tensor ang_vel;
    torch::Tensor gravity_vec;
    torch::Tensor commands;
    torch::Tensor base_quat;
    torch::Tensor dof_pos;
    torch::Tensor dof_vel;
    torch::Tensor actions;
};

class RL
{
public:
    RL() {};
    ~RL() {};

    ModelParams params;
    Observations obs;

    // init
    void InitObservations();
    void InitOutputs();
    void InitControl();

    // rl functions
    torch::Tensor ComputeObservation();
    torch::Tensor ComputeTorques(torch::Tensor actions);
    torch::Tensor ComputePosition(torch::Tensor actions);
    torch::Tensor QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework);

    // yaml params
    void ReadYaml(std::string robot_name);

    // control
    Control control;
    void KeyboardInterface();

    // others
    std::string robot_name;
    torch::Tensor actons_filtered;
    torch::Tensor last_actions;
    
protected:
    // rl module
    torch::jit::script::Module model;
    // output buffer
    torch::Tensor output_torques;
    torch::Tensor output_dof_pos;
};

#endif