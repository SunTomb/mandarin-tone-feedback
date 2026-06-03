# 面向普通话声调学习的可解释 AI 反馈系统

本仓库是中国科学技术大学语言科学课程项目 **“面向普通话声调学习的可解释 AI 反馈系统”** 的最终归档版本。项目结合语言科学论文、可运行 Web 原型、后端音频分析服务、离线实验脚本和真实实验输出，目标是展示一个面向普通话声调学习的轻量、可解释 AI 反馈系统。

系统支持上传或录制孤立单字/短词音频，提取并归一化基频（F0）轮廓，将学习者曲线与目标五度调值曲线对比，并返回与声学偏差对应的规则化反馈。在线 demo 中的预测声调仅作为参考，系统重点不是自动判分，而是通过曲线可视化和可解释反馈帮助学习者理解声调产出问题。

## 最终交付物

- 最终中文课程论文：
  - LaTeX 源文件：[`paper/final_report_cn.tex`](paper/final_report_cn.tex)
  - 编译后的 PDF：[`paper/final_report_cn.pdf`](paper/final_report_cn.pdf)
  - GitHub 可读 Markdown 版：[`paper/final_report_cn.md`](paper/final_report_cn.md)
- 论文图像和 Web demo 截图：[`paper/figures/`](paper/figures/)
- 从实验室服务器下载回本地的实验输出：[`outputs/`](outputs/)
- 关键服务器训练日志：[`logs/`](logs/)
- 服务器训练与复现实验说明：[`docs/server_training_runbook.md`](docs/server_training_runbook.md)

## 项目范围

第一版系统聚焦受控普通话声调反馈：

- 孤立音节或短词；
- 普通话四个词汇声调；
- F0 轮廓提取与归一化；
- 目标五度调值曲线比较；
- 规则化、可解释学习者反馈；
- 离线比较声学特征、wav2vec2 预训练表征和融合模型。

本项目不包含完整自动语音识别、句子级发音评分、用户账户、课堂管理面板、长期练习记录，也不让网页 demo 依赖实时 GPU 推理。

## 系统结构

```text
用户音频 / 数据集样本
        ↓
音频读取 + 静音裁剪
        ↓
F0 提取 + 轮廓归一化
        ↓
在线 demo 的轻量规则预测
        ↓
目标声调曲线比较
        ↓
可解释反馈 + 前端曲线图
```

主要目录：

- [`backend/`](backend/)：FastAPI 后端，负责音频读取、F0 提取、参考声调预测和反馈生成。
- [`frontend/`](frontend/)：React + Vite 前端，负责录音/上传、目标声调选择、F0 曲线展示和反馈卡片。
- [`experiments/`](experiments/)：离线实验脚本，包括声学基线、冻结 wav2vec2 表征、声学 + wav2vec2 融合、自录数据 manifest 和质量过滤。
- [`outputs/`](outputs/)：已归档的实验输出，包括指标、混淆矩阵、处理后特征和轻量分类器。
- [`logs/`](logs/)：服务器端关键训练日志。
- [`paper/`](paper/)：最终论文、Markdown 报告、审稿建议、结果摘要和图像。
- [`docs/`](docs/)：服务器训练 runbook 和项目说明。

## Web demo 运行方式

后端：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "backend[test,ml]"
uvicorn backend.app.main:app --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

后端测试：

```bash
pytest backend/tests -q
```

前端构建与测试：

```bash
cd frontend
npm run build
npm test
```

在线 demo 使用轻量 F0 规则生成参考预测和反馈，不在线运行 wav2vec2 或融合模型。

## 离线实验

代表性命令：

```bash
python experiments/train_acoustic_baseline.py --config experiments/configs/acoustic_baseline.yaml
python experiments/train_representation_model.py --config experiments/configs/representation_model.yaml
python experiments/train_fusion_model.py --config experiments/configs/fusion_model.yaml
python experiments/analyze_representations.py --config experiments/configs/representation_model.yaml
pytest experiments/tests -q
```

已归档的服务器输出位于 [`outputs/`](outputs/)：

- `outputs/thchs_slices_pilot/`：THCHS 派生切片公开语料基线。
- `outputs/self_acoustic_baseline/`：自录数据声学基线。
- `outputs/self_representation_wav2vec2/`：自录数据冻结 wav2vec2 表征基线。
- `outputs/self_fusion_wav2vec2/`：自录数据声学 + wav2vec2 融合模型。
- `outputs/self_filtered_lenient_*`：宽松质量过滤后的自录数据实验。
- `outputs/self_loso_lenient_acoustic_speaker*/`：留一说话人声学基线。
- `outputs/self_quality/`：自录数据质量报告和论文结果摘要。

## 主要实验结果

下表所有数值均来自本仓库已归档的真实实验输出。

| 数据/模型 | 准确率 | 宏平均 F1 |
|---|---:|---:|
| THCHS raw acoustic | 0.2250 | 0.1983 |
| THCHS raw wav2vec2 | 0.3125 | 0.3149 |
| THCHS raw fusion | 0.2625 | 0.2599 |
| Self raw acoustic | 0.3639 | 0.3198 |
| Self raw wav2vec2 | 0.3389 | 0.3211 |
| Self raw fusion | 0.3556 | 0.3373 |
| Self filtered acoustic | 0.3761 | 0.3275 |
| Self filtered wav2vec2 | 0.3644 | 0.3354 |
| **Self filtered fusion** | **0.3965** | **0.3722** |

过滤后自录数据的留一说话人声学基线：

| 测试说话人 | 准确率 | 宏平均 F1 |
|---|---:|---:|
| speaker01 | 0.5156 | 0.5028 |
| speaker02 | 0.3276 | 0.3186 |
| speaker03 | 0.3761 | 0.3275 |

这些结果应理解为小规模课程项目证据，而不是生产级普通话声调自动评分基准。因此，网页原型只把预测声调作为参考，核心展示 F0 曲线、目标轮廓比较和规则化反馈。

## 数据与隐私说明

- 自录孤立单字数据是本项目论文的主要实验路径。
- 本地录音目录语义为 `data/self/{speaker}-{tone}-{repetition}`。
- 原始个人录音文件没有提交到公开仓库。
- 已归档的 `outputs/` 包含复查论文结果所需的处理后特征、指标、混淆矩阵、轻量分类器和质量报告。
- THCHS 派生切片仅作为公开语料基线和局限分析材料，不作为最终学习者反馈系统的理想数据来源。

## 论文与引用

最终课程论文：

- PDF：[`paper/final_report_cn.pdf`](paper/final_report_cn.pdf)
- Markdown：[`paper/final_report_cn.md`](paper/final_report_cn.md)

论文中使用的项目引用：

> 叶盛豪. 面向普通话声调学习的可解释 AI 反馈系统项目[EB/OL]. <https://github.com/SunTomb/mandarin-tone-feedback>.

## 归档状态

本仓库已作为课程提交版本归档，包含：

- 可运行后端与前端原型；
- 最终中文 LaTeX 论文与 PDF；
- GitHub 可读 Markdown 论文报告；
- 从 `/NAS/yesh/mandarin-tone-feedback` 下载回本地的服务器实验输出与训练日志；
- 不包含原始个人录音、本地虚拟环境、大模型缓存或临时演示截图。
