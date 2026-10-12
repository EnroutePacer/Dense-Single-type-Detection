# 论文框架草案

## 工作题目

**Beyond Global Top-K: Coverage-Aware Query Initialization for Dense Single-Class Detection**

中文：**超越全局 Top-K：面向密集单类别检测的覆盖感知 Query 初始化**

题目与方法均为工作版本；在带 GT 诊断完成前，不把 coverage-aware selector 写成已成立贡献。

## 叙事链

```text
密集单类检测并不等于简单检测
-> 固定 query 预算下审计 dense generation、Top-K selection、decoder survival 与 output
-> 量化 unique-GT coverage 与 selection starvation rate
-> 用简单 calibration/quota 和 GT oracle 确认可改进空间
-> 只针对被证实的阶段提出方法
-> 相同 epoch/steps 验证最终 AP/AR、分桶收益和部署代价
```

## 章节

1. Introduction：单类低类别熵与高实例竞争并存。
2. Related Work：DETR O2O/O2M、DEIM Dense O2O、query selection、拥挤/小目标、质量校准。
3. Stage-wise Diagnosis：GT 密度、dense recall、Top-K SSR、decoder trajectory、output 截断。
4. Method：仅放由诊断选中的一个主组件。
5. Experiments：相同协议主表、同 epoch 曲线、分桶、消融、第二数据集、效率。
6. Limitations：固定 K、标注质量、表格基座和数据许可。

## 当前可写材料

- `maxDets` 对密集检测评估的系统性影响；
- 无 GT query 先导观察，明确标注为动机而非结论；
- 配置身份审计和无效实验；
- 固定 CPU latency 协议。

当前材料不足以支撑方法论文主张。
