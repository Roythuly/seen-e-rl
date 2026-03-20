# Sampler Observability

## 关注指标

- env step latency
- rollout length
- reward summary
- actor inference latency
- policy version drift
- action output completeness

## 事件

- `rollout_started`
- `rollout_finished`
- `trajectory_built`
- `replay_record_written`
- `actor_action_failed`
- `env_reset_failed`
