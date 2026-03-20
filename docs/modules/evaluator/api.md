# Evaluator API

## Public APIs

- `Evaluator.evaluate(checkpoint, seeds, env_spec)`
- `CheckpointSelector.select(policy_version=None)`
- `EvalReportWriter.write(report)`

## 接口语义

- evaluator 必须与训练主循环解耦
- 评测应支持多 seed、多 episode
- 输出应既能给人看，也能给 `info` 消费
