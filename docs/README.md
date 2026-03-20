# Docs README

`docs/` 是本仓库的事实源。

## 目录规则

- `architecture/`：系统级当前态设计
  其中 `execution-model.md` 用于明确 `RuntimeLoop`、runtime schedule、publish schedule 与未来训练/推理拆分的职责边界
- `prd/`：里程碑目标与验收标准
- `modules/`：模块级实现规范
- `algorithms/`：算法装配说明
- `rfcs/`：跨模块设计演进
- `adrs/`：关键决策记录
- `runbooks/`：运行与事故处置模板
- `releases/`：冻结版本快照入口

## 写作规则

- 文档主体使用中文
- 文件名、接口名、类型名使用英文
- 交付导向，避免写成空泛调研笔记
- 模块文档必须覆盖定位、接口、contracts、可观测性、故障模式、待办
- 算法文档必须区分 sampler 产出字段与 trainer/learner 派生字段
- runtime 相关文档必须同时说明同步研究闭环与未来异步拆分如何共享同一套上层 contract

## 与外部文档的关系

飞书、讨论串、白板等都只作为输入。最终定稿必须回写到仓库。
