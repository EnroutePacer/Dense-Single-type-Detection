# 术语

- **dense proposal**：对全部 encoder memory 位置导出的候选框；是否属于标准前向的一部分需在具体脚本中说明。
- **Top-K query**：按当前 selector 进入 decoder 的初始化 query。
- **unique-GT coverage**：在一对一约束下可被候选集合覆盖的 GT 数量/比例。
- **selection starvation rate (SSR)**：dense 可覆盖但 Top-K 后不可覆盖的 GT 条件概率。
- **survival gap**：已被上游阶段覆盖的 GT 在 decoder 或输出阶段失去覆盖/置信度。
- **predicted-small**：按预测框面积构造的代理标签，不等同于 GT-small。
- **validity**：实验对声称问题是否有效；与运行是否完成是两个字段。
