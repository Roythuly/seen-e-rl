# Info API

## Public APIs

- `InfoHub.record(event)`
- `MetricSink.write(metric_event)`
- `HealthSink.write(health_event)`
- `CheckpointSink.write(checkpoint_event)`

## 接口语义

- 所有模块都只通过统一 event 入口写观测数据
- sink 可以多路并行，但业务模块不直接依赖具体 sink
