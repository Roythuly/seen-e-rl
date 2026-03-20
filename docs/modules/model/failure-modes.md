# Model Failure Modes

## 常见故障

- `ModelSpec` 与 observation shape 不兼容
- 目标 backend 未安装或不可用
- image / multimodal encoder 配置缺失
- `forward_train(...)` 未返回当前算法所需子域
- checkpoint 与 backend 不匹配

## 处理原则

- 失败尽早暴露
- 错误信息明确指出是 spec、backend、train-output 还是 checkpoint 问题
