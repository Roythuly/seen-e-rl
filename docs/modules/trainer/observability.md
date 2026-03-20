# Trainer Observability

## 关注指标

- update latency
- learner throughput
- buffer fill ratio
- sample age
- stale policy ratio
- actor publish lag
- checkpoint save latency

## 事件

- `update_started`
- `update_finished`
- `buffer_write`
- `buffer_sample`
- `policy_published`
- `checkpoint_saved`
