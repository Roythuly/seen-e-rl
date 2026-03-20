# Trainer Failure Modes

## 常见故障

- rollout batch 字段不完整
- replay buffer 数据不足或陈旧
- 更新后策略未正确发布
- 更新成功但 checkpoint 未落盘
- on/off policy 调度策略与算法不匹配

## 处理原则

- 调度错误优先在 `RuntimeLoop` 暴露
- 数据缺失优先在 batch 装配边界暴露
- publish 与 checkpoint 分离后，错误必须能区分是 learner、publish 还是 checkpoint 链路
