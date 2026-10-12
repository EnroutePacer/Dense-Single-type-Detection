# ADR-001：保留 DEIM 原生根布局

状态：accepted  
日期：2026-10-10

## 决策

活跃代码保留 `train.py`、`engine/`、`configs/`、`tools/` 在仓库根，不包装进额外 `src/DEIM/`。

## 原因

现有 Python import、YAML include、shell launcher 和服务器命令都以 DEIM 根目录为基准。额外嵌套会扩大路径改写范围，并增加配置未加载、相对路径失效和复现实验偏移的风险。

历史服务器快照不作为活跃源码；它继续保留为不可变本地证据。以后论文实现只在根目录活跃代码演进，并由 Git commit/diff 与 source manifest 记录。
