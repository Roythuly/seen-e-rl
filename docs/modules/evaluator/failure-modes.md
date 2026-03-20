# Evaluator Failure Modes

## 常见故障

- checkpoint 无法加载
- evaluation env 与训练 env 配置不一致
- 多 seed 结果缺失

## 处理原则

- 评测失败应明确是 checkpoint、env 还是 runtime 问题
- evaluator 失败不应污染训练状态
