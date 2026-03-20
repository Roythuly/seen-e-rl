# Info Failure Modes

## 常见故障

- 外部 sink 不可用
- metric schema 漂移
- 日志量过大

## 处理原则

- 外部 sink 失败不应阻塞训练主流程
- 结构化日志作为最低保真兜底路径
