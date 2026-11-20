# 历史归档 · 原 v1.0 / V6

| 项目 | 记录 |
|---|---|
| 冻结日期 | 2026-11-19 |
| 原发布提交 | 5bbb98bf84ee098f267d3cad0a3456b5325db6bb |
| 历史标签 | legacy-v6-baseline-2026-11-19 |
| 发布批次 | run_20261119_214237_773592 |
| 原 V6 对照批次 | run_20261119_212334_529951 |
| 批次规模 | 14 株、90 片叶、325,818 个原始点 |
| GitHub 历史发布记录 | 2026-11-20 查询时有旧 v1.0 标签，Releases API 返回空列表 |

旧流程使用稳健茎秆中心线和截面表面模型，比较叶片中线与叶基候选方法，按质量规则输出几何及表型。90 片叶均提取可见中线，68 片达到中高等级，87 片有叶基候选，5 片通过正式叶基规则。正式叶茎夹角和曲率各覆盖 2 片，均来自同一株。[历史发布核验](release-results/release_verification.json) · [失败分析](release-results/batch_failure_analysis.md)

项目从新 V1.0 起以全局欧氏最近邻作为叶基初始估计。V6 冻结为独立的历史基线，原几何流程、质量等级和历史输出继续按当年的方法解释。

## 归档来源和可用性

- [git-baseline](git-baseline/) 是原 Git 提交的完整归档，包含 README、运行脚本、核心算法、测试和原项目总览。
- [release-results](release-results/) 保存旧发布批次可公开的 CSV、Markdown、PNG 及核验记录。可从 [历史三维图集](release-results/index.html) 查看逐株图片。
- [pre-release-v6](pre-release-v6/) 保存整理前的原 V6 说明和入口脚本。完整历史运行环境见本机整理前 ZIP 和成员副本。
- 本机 local-only/working-copy/ 逐字节保留原结果目录 results/v1.0、后续复算批次 results/runs、工作日志 logs、README.local.md。历史 JSON 中包含机器路径的原件也在此目录。为留存迁移开始时的工作区状态，三份未跟踪的 V1.1 审计脚本另存于 local-only/pre-existing-untracked/。
- 本机 local-only/snapshot/ 保留 2026-11-19 整理前原 ZIP。已验证其 SHA-256 为 e1b6174e39805928a0bf9c29ad50c0f093809858b8e6e4ab2ad50409ac200e40，内部原清单的 994 个成员全部匹配。local-only/snapshot-members/ 另保留原 V6 的源码、测试、版本说明、运行结果及 ZIP 原清单。
- 已检查两个工作区及可访问的下载目录，未找到名称为 maize_v1.0_release 的独立发布包。GitHub 旧标签存在，查询时没有旧 Release 说明或附件可归档。原始数据保留于项目数据目录。

## 复现及校验

在 git-baseline/ 中运行以下命令，以外部输入目录提供历史 TXT 和 CSV。整理前 V6 的复现入口另见原 [README_v6.md](pre-release-v6/README_v6.md)。

~~~shell
python -m pip install -r requirements.txt
python run_analysis.py --version
python -m unittest discover -s tests -v
python run_analysis.py --batch --data-dir path/to/csv --txt-dir path/to/txt
~~~

在归档目录中核验可共享文件。本机额外运行第二条命令核验完整本地归档。校验清单对各自列出的文件逐一记录原始字节的 SHA-256。

~~~shell
sha256sum -c SHA256SUMS
sha256sum -c local-only/SHA256SUMS
~~~
