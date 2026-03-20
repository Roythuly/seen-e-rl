# TD3

## 依赖字段

- 与 `SAC` 相同的 `ReplayBatch` 基本字段

## 依赖 model outputs

- deterministic actor output
- twin critic values
- target policy smoothing 所需动作输出

## 调度方式

- 通过 replay buffer 驱动
- critic 更新频率高于 actor 更新频率
- 策略发布时间应与 actor 更新节奏一致

## 接 `info` / `evaluator`

- 重点记录 actor/critic 更新比、target update 统计、buffer 状态
