# 公开发布检查表

- [ ] 确认上游 DEIM URL、commit、许可证和本地 patch。
- [ ] 确认作者、机构、仓库 URL，并生成正式 `CITATION.cff`。
- [ ] 参数化服务器专用 Conda 路径和不可移植模型路径。
- [ ] 数据、输入图片、派生图、权重和论文 PDF 均完成许可审计。
- [ ] Git 索引无权重、原始数据、tar、event、raw log、缓存和大 CSV。
- [ ] 全库无本机/服务器绝对路径、用户名、主机名和凭据。
- [ ] `public_release_manifest.sha256` 在干净 clone 复验通过。
- [ ] Markdown 链接、Python AST、YAML、JSON、CSV 与 shell 语法通过。
- [ ] canonical config 能解析，最小 CPU smoke test 通过。
- [ ] claims matrix 中没有把 `target` 写成 `supported`。
