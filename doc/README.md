# seen-e-rl 技术文档

## 1. 架构概述

seen-e-rl 是一个模块化的强化学习训练框架，基于 PyTorch 构建，支持 off-policy（SAC, TD3）和 on-policy（PPO）算法。

### 核心设计原则

- **模块化**：算法、网络、缓冲区、训练器各自独立，通过接口组合
- **可扩展**：网络注册表支持自定义架构（VLA等大模型）
- **配置驱动**：YAML 配置 + CLI 覆盖，灵活管理实验参数
- **标准化日志**：统一的 `[TRAIN]/[EVAL]/[INFO]` 格式，TensorBoard + wandb

---

## 2. 模块说明

### 2.1 配置系统 (`seenerl/config.py`)

YAML 配置支持继承链（`_base_` 字段），CLI 参数优先级最高。

```
优先级: CLI > 算法YAML > default.yaml
```

嵌套参数通过点分隔符覆盖: `--checkpoint.save_buffer false`

### 2.2 算法模块 (`seenerl/algorithms/`)

所有算法继承 `BaseAlgorithm`，提供统一接口:

| 方法 | 说明 |
|------|------|
| `select_action(state, evaluate)` | 选择动作 |
| `update_parameters(...)` | 更新参数，返回 loss 字典 |
| `get_state_dict()` | 获取可保存状态 |
| `load_state_dict(state, evaluate)` | 加载状态 |

**SAC**: Soft Actor-Critic，支持 Gaussian/Deterministic 策略，自动entropy调节  
**TD3**: Twin Delayed DDPG，延迟策略更新 + 目标策略平滑  
**PPO**: Proximal Policy Optimization，clipped surrogate + GAE

### 2.3 网络模块 (`seenerl/networks/`)

- `BaseActor` / `BaseCritic`: 抽象基类
- `GaussianActor`: 高斯策略（SAC, PPO）
- `DeterministicActor`: 确定性策略（TD3）
- `MLPCritic`: 双Q网络
- `MLPValue`: 状态价值网络（PPO）
- `registry.py`: 使用 `@register_actor("name")` 装饰器注册自定义网络

### 2.4 缓冲区 (`seenerl/buffers/`)

- `ReplayBuffer`: Off-policy，预分配 numpy 数组，支持 `save()` / `load()`
- `RolloutBuffer`: On-policy，GAE 计算 + mini-batch 生成器

### 2.5 训练器 (`seenerl/trainers/`)

- `OffPolicyTrainer`: 逐步交互 → 缓冲区采样 → 网络更新
- `OnPolicyTrainer`: 收集 rollout → 计算 GAE → 多 epoch 训练 → 丢弃数据

### 2.6 评估器 (`seenerl/evaluator.py`)

从训练器解耦的独立评估模块，返回 `{avg_reward, std_reward, min_reward, max_reward}`。

### 2.7 Checkpoint管理 (`seenerl/checkpoint.py`)

多策略保存: `latest`, `best`, `interval_steps`, `interval_epochs`。  
保存内容: 模型参数 + 优化器状态 + buffer数据 + 训练进度。

### 2.8 日志 (`seenerl/logger.py`)

- Console: `[TRAIN]` / `[EVAL]` / `[INFO]` 前缀格式化输出
- TensorBoard: 标量曲线
- wandb: 可选集成，通过 `logger.use_wandb: true` 启用

### 2.9 渲染 (`render/`)

独立子模块，加载 checkpoint 后可视化运行：支持人类渲染模式和视频录制。

---

## 3. 如何添加新算法

1. 创建 `seenerl/algorithms/your_algo.py`
2. 继承 `BaseAlgorithm`
3. 实现四个核心方法
4. 创建 `configs/your_algo.yaml`
5. 在 `train.py` 中添加分发逻辑

## 4. 如何自定义网络

```python
# seenerl/networks/my_vla.py
from seenerl.networks.base import BaseActor
from seenerl.networks.registry import register_actor

@register_actor("vla")
class VLAActor(BaseActor):
    """Visual-Language-Action model for robotics."""
    def __init__(self, ...):
        super().__init__()
        # 构建VLA网络
    
    def forward(self, state):
        ...
    
    def sample(self, state):
        ...
```

然后在算法中使用 `build_actor("vla", **kwargs)` 构建。

---

## 5. 环境要求

- Python ≥ 3.8
- PyTorch ≥ 2.0
- Gymnasium ≥ 0.29（含 MuJoCo）
- 可选: wandb
