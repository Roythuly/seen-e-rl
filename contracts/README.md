# Contracts README

`contracts/` 存放跨模块公共契约的版本化占位 schema。

## 规则

- 当前统一使用 `v0/`
- 首版 schema 为 placeholder，不追求字段一次定完
- 每个 schema 必须至少包含：
  - `$schema`
  - `$id`
  - `title`
  - `description`
  - `type`
  - `x-status: placeholder`

## 演进原则

- 默认 add-only
- 破坏性变更先写 RFC
- 文档先于实现更新
