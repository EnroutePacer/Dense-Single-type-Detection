# EXP-EVAL-002：Invalid-anchor Top-K 修复对照

状态：planned / not_run。

目的：用同一个 checkpoint、验证集和 maxDets=300，单独量化 invalid boundary anchor 在旧 selector 中占用 query 槽位的影响，防止把该公共修复归因于 SGQS。

配对版本：

- 旧 selector：`e0a2353`；
- masked selector：`fd0c110`，关闭 SGQS supervisor。

两次评估必须使用同一 checkpoint SHA-256、输入、预处理、query 数 (K=640)、PostProcessor 和 evaluator。除 COCO 指标外，记录每图 invalid Top-K 数、有效候选数、层级来源和输出差异。

本实验完成前，历史 0.864248 不能作为 SGQS 的直接 control。
