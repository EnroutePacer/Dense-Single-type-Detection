# EXP-DIAG-002：DEIM 与 PaddleX-L 定性比较

相同 13 图、`score>0.5` 下，PaddleX-L 的可视化覆盖在部分密集表格上更完整，DEIM 的高重叠预测代理更多。因为没有 GT，这只能作为失败模式线索，不能证明 precision/recall，也不能证明 PaddleX 使用了表格专用损失。

PaddleX wired/wireless 使用各自领域权重，DEIM 使用同一 checkpoint；因此定性差异还混合了训练数据、训练成熟度、结构深度和校准等因素。历史记录见 [legacy_record.md](legacy_record.md)。
