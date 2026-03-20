# Info Failure Modes

## 常见故障

- 外部 sink 不可用
- metric schema 漂移
- 日志量过大
- 同一训练事件缺少统一关联键

## 处理原则

- 外部 sink 失败不应阻塞训练主流程
- 结构化日志作为最低保真兜底路径
- schema 漂移优先在 `MetricEvent` 校验边界暴露
