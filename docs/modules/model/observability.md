# Model Observability

## 关注指标

- backend 选择与可用性
- 参数规模
- forward latency
- `forward_train` 子域缺失率
- checkpoint save/load 成功率

## 事件

- `model_built`
- `model_forward_failed`
- `model_train_forward_failed`
- `checkpoint_saved`
- `checkpoint_loaded`
