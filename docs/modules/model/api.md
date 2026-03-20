# Model API

## Public APIs

- `ModelFactory.build(spec, backend)`
- `Model.forward_act(observation_batch, policy_state=None)`
- `Model.forward_train(batch)`
- `Model.save_checkpoint(path, metadata)`
- `Model.load_checkpoint(path)`

## 接口语义

- `build`：根据 spec 返回后端实例，不暴露后端细节给上层
- `forward_act`：供 sampler/evaluator 使用，返回动作、分布信息、可选状态
- `forward_train`：供 trainer/algorithm 使用，返回训练需要的中间量
- `save/load_checkpoint`：只要求恢复流程一致，不要求 torch/jax 权重直接互通

## 错误语义

- backend 不存在
- spec 与 observation 不匹配
- checkpoint 元数据缺失
