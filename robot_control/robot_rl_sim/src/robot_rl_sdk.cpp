#include "robot_rl_sim/robot_rl_sdk.hpp"

torch::Tensor RL::ComputeObservation()
{
    std::vector<torch::Tensor> obs_list;

    for (const std::string &observation : this->params.observations)
    {
        if (observation == "lin_vel")
        {
            obs_list.push_back(this->obs.lin_vel * this->params.lin_vel_scale);
        }
        else if (observation == "ang_vel_body")
        {
            obs_list.push_back(this->obs.ang_vel * this->params.ang_vel_scale);
        }
        else if (observation == "ang_vel_world")
        {
            obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.ang_vel, this->params.framework) * this->params.ang_vel_scale);
        }
        else if (observation == "gravity_vec")
        {
            obs_list.push_back(this->QuatRotateInverse(this->obs.base_quat, this->obs.gravity_vec, this->params.framework));
        }
        else if (observation == "commands")
        {
            obs_list.push_back(this->obs.commands * this->params.commands_scale);
        }
        else if (observation == "dof_pos")
        {
            obs_list.push_back((this->obs.dof_pos - this->params.default_dof_pos) * this->params.dof_pos_scale);
        }
        else if (observation == "dof_vel")
        {
            obs_list.push_back(this->obs.dof_vel * this->params.dof_vel_scale);
        }
        else if (observation == "actions")
        {
            obs_list.push_back(this->obs.actions);
        }
    }

    torch::Tensor obs = torch::cat(obs_list, 1);
    torch::Tensor clamped_obs = torch::clamp(obs, -this->params.clip_obs, this->params.clip_obs);
    return clamped_obs;
}

void RL::InitObservations()
{
    this->obs.lin_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.ang_vel = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.gravity_vec = torch::tensor({{0.0, 0.0, -1.0}});
    this->obs.commands = torch::tensor({{0.0, 0.0, 0.0}});
    this->obs.base_quat = torch::tensor({{0.0, 0.0, 0.0, 1.0}});
    this->obs.dof_pos = this->params.default_dof_pos;
    this->obs.dof_vel = torch::zeros({1, this->params.num_of_dofs});
    this->obs.actions = torch::zeros({1, this->params.num_of_dofs});
    this->actons_filtered = torch::zeros({1, this->params.num_of_dofs});
    this->last_actions = torch::zeros({1, this->params.num_of_dofs});
}

void RL::InitOutputs()
{
    this->output_torques = torch::zeros({1, this->params.num_of_dofs});
    this->output_dof_pos = this->params.default_dof_pos;
}

void RL::InitControl()
{
    this->control.v = 0.0;
    this->control.w = 0.0;
    this->control.height = 0.3;
}

torch::Tensor RL::ComputeTorques(torch::Tensor actions)
{
    torch::Tensor actions_scaled = actions * this->params.action_scale;
    torch::Tensor output_torques = this->params.rl_kp * (actions_scaled + this->params.default_dof_pos - this->obs.dof_pos) - this->params.rl_kd * this->obs.dof_vel;
    // // 定义 index
    // std::vector<int64_t> indices = {3, 7};

    // // 转成 Tensor
    // auto indices_tensor = torch::tensor(indices, torch::kLong);

    // // 选取所有 batch
    // auto batch_indices = torch::indexing::Slice();

    // // 进行 index_put_
    // output_torques.index_put_(
    //     {batch_indices, indices_tensor},
    //     this->params.rl_kp.index({batch_indices, indices_tensor}) * 10 * (actions_scaled.index({batch_indices, indices_tensor}) + this->params.default_dof_pos.index({batch_indices, indices_tensor}))
    //     - 0.5 * this->params.rl_kd.index({batch_indices, indices_tensor}) * this->obs.dof_vel.index({batch_indices, indices_tensor}));

    return output_torques;

}

torch::Tensor RL::ComputePosition(torch::Tensor actions)
{
    torch::Tensor actions_scaled = actions * this->params.action_scale;
    return actions_scaled + this->params.default_dof_pos;
}

torch::Tensor RL::QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string &framework)
{
    torch::Tensor q_w;
    torch::Tensor q_vec;
    if (framework == "isaacsim")
    {
        q_w = q.index({torch::indexing::Slice(), 0});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(1, 4)});
    }
    else if (framework == "isaacgym")
    {
        q_w = q.index({torch::indexing::Slice(), 3});
        q_vec = q.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});
    }
    c10::IntArrayRef shape = q.sizes();

    torch::Tensor a = v * (2.0 * torch::pow(q_w, 2) - 1.0).unsqueeze(-1);
    torch::Tensor b = torch::cross(q_vec, v, -1) * q_w.unsqueeze(-1) * 2.0;
    torch::Tensor c = q_vec * torch::bmm(q_vec.view({shape[0], 1, 3}), v.view({shape[0], 3, 1})).squeeze(-1) * 2.0;
    return a - b + c;
}

#include <termios.h>
#include <sys/ioctl.h>
static bool kbhit()//检查输入缓冲区是否有数据
{
    //获取当前终端属性
    termios term;
    tcgetattr(0, &term);

    //临时修改终端属性
    termios term2 = term;
    term2.c_lflag &= ~ICANON;//关闭 ICANON 模式（规范模式），使终端工作在非规范模式下，输入不会等待换行符即可处理,终端会直接将按下的每个按键放入输入缓冲区，而不需要按下 Enter 键
    tcsetattr(0, TCSANOW, &term2);//设置修改后的终端属性，TCSANOW 表示立即生效

    //检查输入缓冲区
    int byteswaiting;
    ioctl(0, FIONREAD, &byteswaiting);//ioctl 是一个系统调用，FIONREAD 用于获取文件描述符的输入缓冲区中有多少字节（字符）

    //恢复终端属性
    tcsetattr(0, TCSANOW, &term);//恢复终端到最初的属性，确保程序对终端的修改不会影响后续使用

    return byteswaiting > 0;
}

void RL::KeyboardInterface()
{
    if (kbhit())
    {
        int c = fgetc(stdin);//从标准输入（stdin）中读取一个字符
        switch (c)
        {
        case 'h':
            this->control.height += 0.05;
            break;
        case 'j':
            this->control.height -= 0.05;
            break;
        case 'w':
            this->control.v += 0.1;
            break;
        case 's':
            this->control.v -= 0.1;
            break;
        case 'a':
            this->control.w += 0.1;
            break;
        case 'd':
            this->control.w -= 0.1;
            break;
        case 'i':
            break;
        case 'k':
            break;
        default:
            break;
        }
    }
}

//YAML::Node 是 yaml-cpp 库中的一个核心类，表示一个 YAML 数据结构的节点,YAML 文件的内容在加载后会被解析为一个树形结构
//从一个 YAML 节点中读取数据，并将其转换为一个包含指定类型 T 的 std::vector
template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }
    return values;
}

template <typename T>
std::vector<T> ReadVectorFromYaml(const YAML::Node &node, const std::string &framework, const int &rows, const int &cols)
{
    std::vector<T> values;
    for (const auto &val : node)
    {
        values.push_back(val.as<T>());
    }

    if (framework == "isaacsim")
    {
        std::vector<T> transposed_values(cols * rows);
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                transposed_values[c * rows + r] = values[r * cols + c];
            }
        }
        return transposed_values;
    }
    else if (framework == "isaacgym")
    {
        return values;
    }
    else
    {
        throw std::invalid_argument("Unsupported framework: " + framework);
    }
}

//读取yaml配置文件，把相应的节点内容赋给相应的变量，
//其中标量直接取出并使用yaml:cpp的Node::as<T>()转化为的对应的类型，向量则调用自定义的ReadVectorFromYaml函数，如果需要参与网络计算的变量，需要转化为torch张量
//.view({1, -1}) 的具体含义:1 表示行数为1,新的张量将有一行。
//                        -1 表示自动推导列数：-1 是一个占位符，表示该维度的大小由 PyTorch 根据原张量的总元素数量和其他维度的大小自动计算
void RL::ReadYaml(std::string robot_name)
{
    // The config file is located at "rl_sar/src/rl_sar/models/<robot_name>/config.yaml"
    std::string config_path = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/models/" + robot_name + "/config.yaml";
    YAML::Node config;
    //try-catch 是 C++ 中的异常处理机制
    try
    {
        config = YAML::LoadFile(config_path)[robot_name];
    }
    catch (YAML::BadFile &e)
    {
        std::cout << "The file '" << config_path << "' does not exist" << std::endl;
        return;
    }
    this->params.model_name = config["model_name"].as<std::string>();
    this->params.framework = config["framework"].as<std::string>();
    int rows = config["rows"].as<int>();
    int cols = config["cols"].as<int>();
    this->params.dt = config["dt"].as<double>();
    this->params.decimation = config["decimation"].as<int>();
    this->params.num_observations = config["num_observations"].as<int>();
    this->params.observations = ReadVectorFromYaml<std::string>(config["observations"]);

    if (config["observations_history"].IsNull())
    {
        this->params.observations_history = {};
    }
    else
    {
        this->params.observations_history = ReadVectorFromYaml<int>(config["observations_history"]);
    }
    this->params.clip_obs = config["clip_obs"].as<double>();
    if (config["clip_actions_lower"].IsNull() && config["clip_actions_upper"].IsNull())
    {
        this->params.clip_actions_upper = torch::tensor({}).view({1, -1});
        this->params.clip_actions_lower = torch::tensor({}).view({1, -1});
    }
    else
    {
        this->params.clip_actions_upper = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_upper"], this->params.framework, rows, cols)).view({1, -1});
        this->params.clip_actions_lower = torch::tensor(ReadVectorFromYaml<double>(config["clip_actions_lower"], this->params.framework, rows, cols)).view({1, -1});
    }
    this->params.action_scale = config["action_scale"].as<double>();
    this->params.hip_scale_reduction = config["hip_scale_reduction"].as<double>();
    this->params.hip_scale_reduction_indices = ReadVectorFromYaml<int>(config["hip_scale_reduction_indices"]);
    this->params.num_of_dofs = config["num_of_dofs"].as<int>();
    this->params.lin_vel_scale = config["lin_vel_scale"].as<double>();
    this->params.ang_vel_scale = config["ang_vel_scale"].as<double>();
    this->params.dof_pos_scale = config["dof_pos_scale"].as<double>();
    this->params.dof_vel_scale = config["dof_vel_scale"].as<double>();
    // this->params.commands_scale = torch::tensor(ReadVectorFromYaml<double>(config["commands_scale"])).view({1, -1});
    this->params.commands_scale = torch::tensor({this->params.lin_vel_scale, this->params.lin_vel_scale, this->params.ang_vel_scale});
    this->params.rl_kp = torch::tensor(ReadVectorFromYaml<double>(config["rl_kp"], this->params.framework, rows, cols)).view({1, -1});
    this->params.rl_kd = torch::tensor(ReadVectorFromYaml<double>(config["rl_kd"], this->params.framework, rows, cols)).view({1, -1});
    this->params.fixed_kp = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kp"], this->params.framework, rows, cols)).view({1, -1});
    this->params.fixed_kd = torch::tensor(ReadVectorFromYaml<double>(config["fixed_kd"], this->params.framework, rows, cols)).view({1, -1});
    this->params.torque_limits = torch::tensor(ReadVectorFromYaml<double>(config["torque_limits"], this->params.framework, rows, cols)).view({1, -1});
    this->params.default_dof_pos = torch::tensor(ReadVectorFromYaml<double>(config["default_dof_pos"], this->params.framework, rows, cols)).view({1, -1});
    this->params.joint_controller_names = ReadVectorFromYaml<std::string>(config["joint_controller_names"], this->params.framework, rows, cols);
}