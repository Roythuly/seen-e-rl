# seen-e-rl 技术文档

`seen-e-rl` 是一个高度模块化、具备强扩展性且鲁棒的强化学习训练框架，基于 PyTorch 构建，专为连续控制任务设计。

它不仅支持基于 Gymnasium 的轻量化算法原型验证，同时也无缝原生支持 NVIDIA Isaac Lab 平台的大规模并行化机器人仿真控制训练。框架在底层将两种环境封装为统一形式，并共享所有的算法和模型层。

---

## 🚀 1. 核心设计特性

- **算法抽象化（Algorithm Registry）**: 框架内建 `SAC`, `TD3`, `PPO`, 和 `OBAC` 的实现。所有的算法被抽象注册，与 Trainer 解耦。
- **环境工厂（Environment Factory）**: 为单环境 Gym、向量化并行 Gym 和 Isaac Lab 提供了完全一致的 Batch 输入输出接口。1 个和 10000 个环境对 Trainer 没有任何区别。
- **模型工厂（Model Factory）**: 所有的网络结构（Actor/Critic）剥离至 YAML 设定的 `model.*` 中进行动态构建，支持灵活修改隐藏层维度和架构而无需碰触业务代码。
- **统一且完备的配置系统**: 基于 YAML 字典实现，支持 `_base_` 继承和基于点语法 `.` 的嵌套参数 CLI 命令行覆盖。配置随运行自动持久化保存。
- **无缝恢复与端到端评估**: 仅依靠给定的 checkpoint 路径即可自动反向推导恢复（Infer）全套实验配置。支持通过 `--resume` 无感恢复训练，或借由 `--checkpoint` 独立评估渲染。
- **PPO 增强与对齐**: 引入了 State-of-the-Art 的工程化细节，包含：Per-mini-batch Advantage 标准化、价值函数截断（Value Clipping）以及稳定化双截断目标（Dual-Clip）。
- **标准化日志输出**: 内建 `[TRAIN]`, `[EVAL]`, `[INFO]` 等格式化终端样式，开箱即支持挂载 TensorBoard 标量曲线与完整的 Weights & Biases (wandb) 实验管理支持。

---

## 📂 2. 软件架构解析

项目自底向上被划分为如下层次：

### 2.1 配置解析层 (`seenerl/config.py`)
使用层级化的深拷贝（deep copy）实现继承，并在最终转换为拥有属性访问权限的 `Config` 字典对象类。在保存 Checkpoint 时，配置被完整冻结录入，用作未来恢复的状态锚点。

### 2.2 环境适配层 (`seenerl/envs/`)
在 `SingleGymEnv`、`VectorGymEnv` 以及 `IsaacLabEnv` 之间建立一层垫片。无论使用哪种环境：
- `obs` 和 `actions` 在进出环境时一定是被拉平为 2D 且限定为 `float32` 类型的 Tensor 批次形态。
- Gym 底层的截断与终止被自动重置（Auto-reset），并使用同样的字典 `final_observation` 进行截取补回。

### 2.3 算法与网络构建 (`seenerl/algorithms/`, `seenerl/models/`, `seenerl/networks/`)
- `registry.py`: 为指定名称分配 `on_policy`（基于采样池批次训练）或 `off_policy`（基于重放回放池训练）分类。
- `BaseAlgorithm`: 通用的算法模板抽象。规定了必须实现的核心接口 `select_action()`, `update_parameters()`, `get_state_dict()` 等。
- `networks/base.py`: 基于 PyTorch 的各种标准化子模型，如提供给 SAC 的 `GaussianActor`、提供给 PPO 的 `GaussianFixedStdActor` 等。通过 `@register_actor` 被模型工厂装载。

### 2.4 训练核心循环 (`seenerl/trainers/`)
包含 `OnPolicyTrainer` 和 `OffPolicyTrainer` 两个骨干循环，负责组装所有要素，与对应类型的 Buffer 交互采样，定期调用 evaluator，以及派发 checkpoint。

---

## 🛠️ 3. 配置字典详解 (Config Schema)

配置被拆解成专门针对实验设计、算法设定以及环境部署等区段。框架同样完整保留了以前 `env_name`、`hidden_size` 的后向兼容映射能力。

### 📌 环境声明区块 (Environment Block)

```yaml
env:
  backend: "gymnasium"      # 后端: "gymnasium" 或 "isaaclab"
  id: "Pendulum-v1"         # 任务名
  num_envs: 8               # 并行实例数量
  kwargs: {}                # 建构附带值
  isaaclab:                 # Isaac Lab 专属设定区块
    headless: true
    use_fabric: true
    task_imports: 
      - "isaaclab_tasks.manager_based.manipulation.pick_place"
```

### 📌 模型声明区块 (Model Block)

可选。如果缺省则采用对应算法注册时的默认默认。

```yaml
model:
  actor:
    name: "gaussian"
    hidden_dim: 256
    kwargs:
      squash: false
  critic:
    name: "q_network"
    hidden_dim: 256
```

---

## ⚙️ 4. 实验指南

我们建议使用 `uv` 来极速搭建干净的依赖虚拟环境：

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 4.1 开启一场训练

Gym 训练直接使用命令执行，算法与环境通过 `--config` 指定，也可动态采用 `--env.num_envs` 伸缩资源：

```bash
uv run train.py --config configs/sac.yaml --env_name HalfCheetah-v4
uv run train.py --config configs/ppo.yaml --env_name Pendulum-v1 --env.num_envs 16
```

针对 Isaac Lab 环境的大规模强化学习仿真，在安装有 Isaac Sim 虚拟环境的系统中：
```bash
python train.py --config configs/isaaclab_pickplace_ppo.yaml --env.num_envs 64
```

### 4.2 训练恢复与断点重放
无需指定任何其它配置，读取 checkpoint 即可自适应组装原定策略模型：

```bash
uv run train.py --resume results/Pendulum-v1/PPO/.../checkpoints/latest.pt
```

### 4.3 渲染与结果评估

脱离 Trainer 进行独立验证或生成可视化。脚本会自动识别 config，所以只需要传入训练保存下的 checkpoint：

```bash
# 仅仅统计数据
uv run evaluate.py --checkpoint results/xxx/checkpoints/best.pt --num_episodes 10

# 调用屏幕渲染器
uv run render/renderer.py --checkpoint results/xxx/checkpoints/best.pt --episodes 5
```

---

## 🧪 5. 定制与扩展

1. **添加新算法**:
   - 在 `seenerl/algorithms/` 添加 `your_algo.py`。
   - 实现包含 `select_action`, `update_parameters` 四大主心骨继承于 `BaseAlgorithm` 的类。
   - 于 `__init__.py` 或 `registry.py` 中注册 `trainer_kind` 分片。
   
2. **添加自定义网络 (VLA, Transformer 等)**:
   - 在 `seenerl/networks/` 实现自定义 PyTorch Model，并用 `@register_actor("my_vla")` 包装。
   - 在 YAML 配置文件中使用 `model.actor.name: "my_vla"` 无缝切换。

## 许可证
MIT License
