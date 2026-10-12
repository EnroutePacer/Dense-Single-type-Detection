# Data

本目录只提交数据说明、split、统计和 manifest，不提交私有图片或标注原件。

每个 `DATA-*` manifest 至少包含：数据集名称/版本、来源 URL、许可、train/val/test 划分规则、图像数、标注数、annotation SHA-256、类别映射、去重与清洗说明，以及可复算统计脚本。

当前阻塞项：13 张 query 诊断图没有配套 GT JSON；完整训练/验证标注尚未登记哈希。取得这些资产前，不能计算真实 selection starvation rate。
