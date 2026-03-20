# Contracts README

`contracts/` 存放跨模块公共契约的版本化 minimum schema。

## 规则

- 当前统一使用 `v0/`
- 首版 schema 追求最小可实现，不追求字段一次定完
- 每个 schema 必须至少包含：
  - `$schema`
  - `$id`
  - `title`
  - `description`
  - `type`
  - `x-status`
  - `required`

## 演进原则

- 默认 add-only
- 破坏性变更先写 RFC
- 文档先于实现更新
- 算法文档、schema、配置模板三者必须互相对齐
