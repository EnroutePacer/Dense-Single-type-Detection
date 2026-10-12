# Tests

优先补三类测试：

1. launcher 尊重外部 `CONFIG_FILE`，resolved config 的 Resize/loss/query 与预期一致；
2. selector 在所有输入上保持固定 K、索引和层级映射正确；
3. 合成 GT 上 generation/selection/decoder/output 归因指标可手算复核。

研究代码只有在相应单测存在后才能进入主实验。
