# Configs README

本目录存放最小实验配置模板。`experiment/` 是顶层装配入口，其余目录提供模块级子配置示例。

## 子目录

- `algo/`
- `backend/`
- `env/`
- `model/`
- `sampler/`
- `buffer/`
- `trainer/`
- `eval/`
- `experiment/`

## 约束

- `experiment/example.yaml` 必须显式拼出 `backend`、`env`、`model`、`algo`、`sampler`、`trainer`、`buffer`、`eval`
- 当前默认事实源是 `Torch-first`
- `JAX` 作为兼容目标保留在 backend/model contract 中，但不要求与 `Torch` 同轮完全等权
