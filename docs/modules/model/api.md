# Model API

## Public APIs

- `ModelFactory.build(spec, backend)`
- `Model.forward_act(observation_batch, policy_state=None)`
- `Model.forward_train(train_request)`
- `Model.save_checkpoint(path, metadata)`
- `Model.load_checkpoint(path)`

## 接口语义

- `build`：根据 spec 返回后端实例，不暴露后端细节给上层
- `forward_act`：供 sampler/evaluator 使用，返回 `ActionOutput`
- `forward_train`：供 trainer/learner 使用，返回结构化 `TrainOutputs`
- `TrainOutputs` 最少分为 `policy`、`value`、`q`、`target`、`aux` 五个可选子域
- `save/load_checkpoint`：恢复的是训练态组件集合，必须与 `CheckpointManifest` 对齐

## `ActionOutput`

- 必带：`action`、`policy_version`
- 可选：`log_prob`、`value_estimate`、`actor_state`、`diagnostics`

## `TrainOutputs`

- `policy`：分布参数、deterministic action 或 actor diagnostics
- `value`：state value 或 value diagnostics
- `q`：online Q / twin Q
- `target`：target network 相关中间量
- `aux`：entropy、target smoothing、mask、debug stats 等算法特定补充量

## 错误语义

- backend 不存在
- spec 与 observation 不匹配
- `TrainOutputs` 缺失当前算法所需子域
- checkpoint 元数据缺失或与 backend 不匹配
