# Sampler Failure Modes

## 常见故障

- env reset / step 失败
- observation 无法标准化
- actor 返回非法动作
- 采样使用过旧策略版本

## 处理原则

- 尽量把故障定位到 env、adapter、actor 三个层次之一
- 对 policy version drift 提供监控而不是静默忽略
