# SGQS v2：方法配置与实验配方分离

[method.yml](method.yml) 仅定义 SGQS v2 附加方法；[train.yml](train.yml) 是组合模型、数据变换与方法的通用配置入口。训练轮数、数据路径、batch、seed、输入大小、query 数、AMP/EMA、验证预算、保存间隔和升温起点由命令行显式决定，见 [SERVER_COMMANDS](SERVER_COMMANDS.md)。底层不要求某个实验编号或初始化 epoch。

方法保留受保护 owner 的几何辅助，以 IoU 为 dense 分类目标，统一升温；用实例身份约束的分配替代旧 survival，不包含 EFGL 或 Top-300 admission loss。冗余中心/IoS、接收者和无接收者溢出规则见 [M-003](../../article/methods/M-003_sgqs_v2/README.md)。架构和推理头不因此改变。

运行、统计与恢复契约见 [框架 README](../../engine/sgqs_v2/README.md)。每次保存 resolved.yml 和实际 CLI，输入、源码和 checkpoint 身份可追溯。需要更改恢复参数时，从完整 checkpoint 创建新的 resume 运行；同一次中断的 recover 则保持原配置。

## 保留的历史固定实验配置

以下是 EXP-TRAIN-008 的配方，而非框架限制。八卡/640/epoch27→34 示例单独放在 [recipes/EXP-TRAIN-008.md](recipes/EXP-TRAIN-008.md)。common.yml 与 base_v2.yml 仅服务这些历史配方；通用入口不读取 arm 表或冻结输入约束。

| 配置 | 作用 |
|---|---|
| [b0_e28_r7.yml](b0_e28_r7.yml) | 基线配方 |
| [support_e28_r7.yml](support_e28_r7.yml) | v2 去掉 allocation 的对照 |
| [v2_e28_r7.yml](v2_e28_r7.yml) | 完整 v2 配方 |
| [sgqs_e28_r7.yml](sgqs_e28_r7.yml) | 保留旧语义的 v1 参照 |

历史实验继续遵守其预注册数据、预算、small 标准和对照设置。通用化允许另开新实验，不意味着可以把不同协议结果混在一起。单 seed 七轮仍只是筛选，当前没有新真实训练或性能结论。
