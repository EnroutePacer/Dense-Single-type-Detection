# Source coverage 修订与固定服务器环境兼容

日期：2026-10-28。基于 ee21f2f，保留远程通用训练/恢复框架、Decoder IoS 冗余与 overflow 设计。

## 用户已审核的 100 图证据

来自 `final_100_hd_v2` 对应的 100 图、14,540 GT（本地 small 6,575），使用完整 Encoder CSV 的未裁剪框。统计由用户人工审查并确认，作为本次代码修订的输入；这里不冒充本机重新执行的审计。机器可读记录见 [manual_audit.json](manual_audit.json)。

| 模型/分组 | 旧 IoU≥.5 覆盖 | 唯一 native 中心 + Encoder IoS≥.8 覆盖 | 旧覆盖但新缺失 |
|---|---:|---:|---:|
| B0 全部 | 14,530 / 14,540 | 13,661 / 14,540 | 871 |
| B0 small | 6,566 / 6,575 | 5,702 / 6,575 | 865 |
| SGQS 全部 | 14,451 / 14,540 | 14,405 / 14,540 | 50 |
| SGQS small | 6,486 / 6,575 | 6,440 / 6,575 | 50 |

报告中 B0 缺失879：无中心803、有中心但IoS不足76；SGQS缺失135：89/46。结论是旧指标无法衡量原始候选来源是否服务于该GT，不能把“回归框达到IoU”当成“来源覆盖”。该证据不等于新版训练或剪枝性能已经验证。

## 训练修订

同一有效候选满足唯一回归前中心归属与 Encoder IoS≥.8 才是合格来源，IoS=交集/较小面积。coverage 和外部最高分 receiver 使用同一条件，不用 Decoder 框或回归后中心代替。其他 GT 的 IoS 与面积比只记录，不加面积硬门槛。IoU继续承担跨实例安全检查。

每个已覆盖 GT 先固定一个合格来源代表，优先已有可靠 Decoder 代表，否则取合格来源最高 Encoder 分；该候选不受 transfer/overflow 降分，防止只保护Decoder代表却误降唯一合格source。无接收者时不伪造排序样本。

日志口径为 `v2_source_coverage_version=2`，缺失互斥拆成完全无中心、只有歧义中心、有唯一中心但 Encoder IoS不足，含本地small分组。全dense无合格来源与无唯一中心另记。面积比、跨GT IoS以及旧IoU对照保留分子分母。

有中心低IoS但外部存在合格来源时，排序可以换入已有好来源；全dense无合格来源时，现有owner/base回归继续工作，不给低质量来源新增救援升分。没有新增“保证覆盖所有几何失败”的回归项。

## 服务器报错与兼容

原始附件逐字节保存在 [server_console.log](server_console.log)，SHA256登记于manual_audit.json。致命错误为 `ModuleNotFoundError: No module named 'faster_coco_eval'`，发生于训练入口导入阶段；此前TensorFlow factory/TensorRT消息不是此次退出根因。

数据集、训练COCO evaluator、训练完整raw/EMA验证与活跃离线评价共用自动后端：已有faster-coco-eval → 已有pycocotools → 仓库内置bbox COCO参考实现。无pip、编译、联网或sys.modules包伪装。内置实现保留上游COCO匹配/累积代码和许可证，NumPy实现bbox IoU（含crowd分母），不伪装支持segmentation/RLE。参考后端更慢，maxDets300/640和严格一对一local-small口径保持；后端标识进入评价结果/矩阵缓存身份。

旧版PyTorch缺少`torch.amp.GradScaler`时，CUDA训练回退既有`torch.cuda.amp.GradScaler` API，保留AMP配置和状态。没有修改训练服务器环境。

同步完整源码后使用原CLI参数。此前已成功运行的源码身份仍受保护，改变方法后应resume到新目录，不绕过recover校验。具体命令见 [SERVER_COMMANDS](../../../configs/sgqs_v2/SERVER_COMMANDS.md)。检查记录见 [validation.json](validation.json)。
