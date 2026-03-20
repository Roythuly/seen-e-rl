# SAC

## 依赖字段

- `ReplayBatch.observations`
- `ReplayBatch.actions`
- `ReplayBatch.rewards`
- `ReplayBatch.next_observations`
- `ReplayBatch.terminated`

## 依赖 model outputs

- actor distribution params
- twin critic values
- target critic values

## 调度方式

- `sampler` 与环境交互并写入 `ReplayBuffer`
- `trainer` 从 buffer 采样做更新
- 更新完成后可立即发布新策略给 `ActorHandle`
- 支持“每步交互 + 若干次训练”的同步节奏

## 接 `info` / `evaluator`

- 重点记录 buffer fill ratio、sample age、critic loss、alpha 相关指标
- evaluator 默认按最新稳定 checkpoint 评测
