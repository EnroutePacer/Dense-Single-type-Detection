# 源码与模型来源

## 服务器训练快照

`server_DEIM_20261010_182837/DEIM/` 是从服务器归档解出的源码、配置、工具、训练日志、TensorBoard 和 evaluator 数据。原始不可变归档位于 `../99_archive/DEIM_code_logs_20261010_182837.tar.gz`。

归档：

- 大小：209,909,671 bytes
- SHA-256：`488045E677E3B5A4ED81CF6EDB3EEA9270CA5C3518512AC20D86DDD6DCEB0ADE`
- 成员数：183
- 不包含 `.git`、`checkpoint0027.pth` 或各探针的 `probe.pth`

## 第一版 S 模型配置快照

`local_deim_s_baseline/` 保存本机现存的第一版 S 训练入口、`workspace/config/train.yml` 等文件，以及其 include 链中的关键 YAML。

该配置从 `deim_hgnetv2_s_coco.yml` 继承完整 DEIM 配方。注意 `train.yml` 中 `mosaic_prob: 0.0` 被写在 `transforms.policy` 下，而官方 `base/deim.yml` 的键位于 `transforms.mosaic_prob`；在没有解析后 cfg 的原始日志时，不能据此断言 Mosaic 已被关闭。MixUp 的覆盖位置则是 `collate_fn.mixup_prob: 0.0`。

第一版 S 的第三 epoch mAP≈0.86 目前是用户报告值；原训练日志和 checkpoint 尚未进入本目录。

## 推理模型

`inference_models_20261010/` 保存 CPU 可视化与 latency 实验使用的实际模型。

| 模型 | 文件 | SHA-256 |
|---|---|---|
| DEIM checkpoint | `DEIM/last.pth` | `81413C4638EB2CE76F175207AD01A844C6D88FDC62BC10D5E68EA49B2893FAAC` |
| PaddleX wired 参数 | `PaddleX/RT-DETR-L_wired_table_cell_det_infer/inference.pdiparams` | `608282D3206A62A4C5AA0E0E9FC3484CA504A500220049635640BFDB5E724906` |
| PaddleX wireless 参数 | `PaddleX/RT-DETR-L_wireless_table_cell_det_infer/inference.pdiparams` | `6AAAC2A3026A2324F7A663B0C3270F90D483FB9C453E63ADB62C6EDC2D3E8BC7` |

## 论文

`papers/` 保存 DEIM arXiv v3 原文及本地中文翻译：

- `2412.04234v3.pdf`：SHA-256 `DB188C17743F2EB443F9912047A77037C4736D5C800EC67E66E1FE02A04DA469`
- `2412.04234v3_中文翻译.pdf`：SHA-256 `6AD984BF4F99C6D1934465B5E14E048C969DDEB404CE694A636760ED70C4B72D`

## 尚缺的训练复现资产

- 服务器 `output/dfine_hgnetv2_m_table_cell/checkpoint0027.pth`；
- 每个探针的 `probe.pth`；
- 服务器仓库 commit ID 或完整 diff；
- 数据集标注 JSON 与哈希；
- Conda/pip freeze、PyTorch、CUDA、cuDNN 和 GPU 信息；
- 每次实验原始启动命令的不可变副本。

