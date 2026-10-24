# 当前研究路线图

2026-10-24 用户暂停旧根因探针。本路线由 [M-003完整设计](../../methods/M-003_sgqs_v2/README.md) 和 [EXP-TRAIN-008](../../experiments/EXP-TRAIN-008_sgqs-v2-coverage-allocation/README.md) 定义；此前单因素路线保留于Git历史与EXP-TRAIN-007，不继续默认执行。

1. 完整设计后实现：保留几何辅助、修正分类目标和归一化、统一升温，加入受GT身份约束的候选分配；完成梯度、门控、DDP归一化、真实前后向、@640与实际K执行检查。
2. 最小主比较：同epoch27初始化跑b0/support/v2，七轮seed0；原SGQS可选参照，不再扩大六臂根因矩阵。
3. 固定K=640完整val：每轮raw/EMA，主@640、附@300，严格local-small/COCO各尺度与固定GT>=301高密度场景。检查实际pair支持与对support的净改善。
4. 同checkpoint实际K=640/512/384/256：重跑Decoder、记录当前后端同步forward时间，比较等K指标和可接受性能损失下的最小K。不部署未经验证的动态阈值。
5. 若机制与指标均有支持，再进行多seed完整训练、独立固定难例集和目标部署后端测量。若无安全pair或无指标改善，如实收敛结论，不堆叠更多loss兜底。

当前只提交可运行代码与协议，不自动启动GPU训练。
