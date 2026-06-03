# 《面向普通话声调学习的可解释 AI 反馈系统》审稿式修改建议

## 总体评价

这篇论文已经具备课程项目的核心条件：有明确语言科学问题意识、有可运行 Web demo、有自建数据与真实实验结果，并且多处主动限制了“分类准确率”的解释范围。整体方向是可提交的，但现在仍有几个会被严格评审追问的问题：**实验划分与验证集使用不够透明、主数据结果缺少关键图证据、模型结果解释略偏因果化、参考文献和格式还不够规范**。

我核对了论文与代码，主指标与 `paper/self_experiment_results.md`、`paper/self_experiment_summary.json` 一致，没有发现把建议实验写成已完成实验的明显问题。但本地仓库没有 `outputs/` 目录，论文中关于 tone4 混淆的更细证据目前只能从摘要记录间接支持，提交前应补充真实混淆矩阵或分类报告文件。

---

## 主要问题

### 1. 必须修改：实验划分与验证集使用不清楚

论文在 `paper/final_report_cn.tex:120-122` 写了 train/val/test 数量，但当前训练脚本实际只用 train 和 test：

- `experiments/train_acoustic_baseline.py:48-53`
- `experiments/train_representation_model.py:71-79`
- `experiments/train_fusion_model.py:35-45`

验证集没有用于调参、早停或模型选择。现在论文容易让读者误以为验证集参与了模型开发流程。建议在“模型设置与可复现性”中明确写：

> 当前基线实验采用预设说话人独立划分，训练仅使用 train split，测试在 test split 上报告；val split 在本版本中保留为未来阈值/超参数选择集合，未参与当前表格结果的模型选择。

否则评审会问：既然有验证集，为什么没有说明其用途？

---

### 2. 必须修改：主数据路径缺少主图，THCHS 图反而更突出

论文强调主数据路径是自建孤立单字数据，但目前正文图包括：

- Web demo 截图：`paper/final_report_cn.tex:101-109`
- THCHS F0 示例：`paper/final_report_cn.tex:208-216`

而自录数据没有 F0 示例图、混淆矩阵图或质量过滤图。这样会造成叙事不平衡：读者看到的声学图主要来自“局限分析”的 THCHS，而不是主数据。

建议至少替换或新增一个图：

- **图 2：自录数据四声 F0 典型曲线**，每声选 2–3 条过滤后样本，叠加目标五度曲线。
- 或 **图 2：filtered self fusion confusion matrix**，用真实输出生成，支持 `tone4` 与 `tone3` 混淆的说法。
- THCHS 图可以保留，但标题应更明确为“连续语音派生切片局限示例”。

---

### 3. 必须修改：tone4 弱项目前证据不足

论文在 `paper/final_report_cn.tex:217-218` 写到 tone4 弱、与 tone3 混淆，这是合理解释，但正文没有展示 per-class F1 或混淆矩阵。`paper/self_experiment_results.md:25` 也只是一条 notes，不是完整证据。

建议二选一：

1. **补充真实混淆矩阵或 per-class F1 表**；
2. 如果来不及补图，就把语气降级：

原意：

> tone4 仍是较弱类别，并且不少样本与 tone3 混淆。

建议改为：

> 根据实验记录中的分类摘要，tone4 是需要重点检查的类别，部分错误与 tone3 相关；由于本文未展开逐类误差统计，后续应以混淆矩阵和人工复核进一步确认这一模式。

如果已有真实 confusion matrix，请直接放入论文，不要只口头描述。

---

### 4. 强烈建议修改：过滤提升的解释略有因果化

`paper/final_report_cn.tex:184-185` 写“说明去除低质量录音对声调分类有帮助”。这个结论方向合理，但严格说目前只有一个过滤策略、一个固定划分、一个随机种子，不能排除过滤改变样本分布、移除困难样本等因素。

建议改成：

> 在当前固定划分和 lenient 过滤策略下，过滤后最佳融合模型的 macro-F1 从 0.3373 提升到 0.3722，提示低质量录音可能影响声调分类；但由于过滤也改变了调类和说话人分布，这一结果不能单独解释为模型能力提升。

这会显著提高诚实度。

---

### 5. 强烈建议修改：Web demo 与离线模型关系还可更形式化

代码显示 Web 后端是轻量规则预测：

- `backend/app/inference.py:35-58` 使用 F0 曲线启发式规则预测声调。
- `backend/app/feedback.py:24-42` 使用目标声调条件下的规则反馈。
- `backend/app/main.py:37-54` `/api/analyze` 返回预测、目标曲线、用户曲线和反馈。
- 前端明确展示“预测仅供参考”：`frontend/src/components/FeedbackCard.tsx:18-25`。

论文已经说了“不依赖实时 GPU 推理”，但建议在系统设计中增加一个小表：

| 模块 | 在线 demo 使用 | 离线实验使用 |
|---|---|---|
| F0 提取 | 是 | 是 |
| 规则预测 | 是 | 否/仅 demo |
| wav2vec2 表征 | 否 | 是 |
| 融合模型 | 否 | 是 |
| 规则反馈 | 是 | 可用于分析 |

这样能避免评审误解“网页里跑了 wav2vec2 融合模型”。

---

## 逐节修改建议

### 摘要

`paper/final_report_cn.tex:57` 信息密度很高，优点是完整，问题是像把全篇压进一段。建议压缩工程细节，突出三件事：

1. 语言科学问题：调类 vs 调值，F0 轮廓反馈。
2. 系统贡献：F0 可视化、目标五度曲线、规则反馈。
3. 实验事实：1080→953，best accuracy/macro-F1，speaker 差异。

建议把“高于声学基线和 wav2vec2 表征模型”改为：

> 在该固定划分下高于单独声学特征和单独 wav2vec2 表征路径。

避免读者理解为普遍结论。

---

### 引言

`paper/final_report_cn.tex:63-72` 逻辑较好，但最后研究问题略宽：“预训练语音表征能否结合传统 F0 声学特征，为普通话声调学习提供可解释反馈”。严格说，预训练表征并没有直接生成反馈，反馈主要来自 F0 规则。

建议改成更准确的三分问题：

> 本文考察两个层面的问题：第一，F0 曲线和五度调值比较能否构成学习者可理解的声调反馈界面；第二，在离线声调分类中，显式声学特征、冻结 wav2vec2 表征及其融合在小规模孤立单字数据上表现如何；第三，这些实验结果如何限制在线 demo 中“预测仅供参考”的设计边界。

这样更贴合代码和结果。

---

### 语言学背景与相关工作

这一节是当前论文的强项。`paper/final_report_cn.tex:75-89` 已经体现调类、调值、F0、CAPT。

建议增强两点：

1. **三声变体**：你现在写“三声常表现为低凹或转折”，但需要说明在孤立单字中可以期待低凹/降升，在连续语音中三声常有变体，因此 THCHS 切片更难。
2. **五度调值是近似目标，不是真实个体 F0 模板**：避免被质疑“所有人都应发成 55/35/214/51”。

可在 `paper/final_report_cn.tex:79-83` 后加一句：

> 因此，本文中的目标五度曲线只作为教学型相对轮廓，而不是对每个说话人真实 Hz 轨迹的强约束。

---

### 系统设计与开源实现

`paper/final_report_cn.tex:91-99` 基本可信，代码支持如下：

- 静音裁剪：`backend/app/audio.py:7-18`
- 16 kHz 重采样：`backend/app/audio.py:19-23`
- `librosa.pyin` F0：`backend/app/features.py:77-87`
- 小半音范围保持平稳曲线：`backend/app/features.py:4-15`
- 规则反馈：`backend/app/feedback.py:24-42`
- 前端“预测仅参考”：`frontend/src/components/FeedbackCard.tsx:18-25`

但 `paper/final_report_cn.tex:83-84` 说“并计算起点、中点、终点、范围、斜率、时长和有声比例等摘要特征”。这在实验特征中成立，但 Web API 响应并不展示这些全部摘要；`backend/app/features.py:25-47` 的 `summarize_contour` 也没有在 `/api/analyze` 中返回。

建议改为：

> 离线实验计算起点、中点、终点、范围、斜率、时长和有声比例等摘要特征；在线 demo 则主要返回归一化曲线、参考预测、目标曲线和反馈项。

---

### 数据构建与实验方法

这是需要最严谨修改的一节。

#### 自录数据

`paper/final_report_cn.tex:120-122` 的样本数量与结果文件一致，可信。但应补充：

- 主实验 test speaker 是 `speaker03`，来自 manifest 脚本默认设置：`experiments/self_recording_manifest.py:59-75`。
- 留一说话人是后续单独生成的 LOSO manifest：`experiments/self_recording_quality.py:86-97`、`155-157`。

建议写：

> 主结果采用 speaker03 作为 held-out test speaker，非测试说话人的第 3 次重复作为 val，其余为 train；此外，为检查说话人差异，另行构造三个 leave-one-speaker-out acoustic baseline。

否则读者会把主表和 LOSO 表的关系看混。

#### 模型设置

`paper/final_report_cn.tex:152-155` 需要更具体：

- acoustic baseline 默认是 RandomForestClassifier，200 trees，class_weight balanced：`experiments/train_acoustic_baseline.py:23-29`
- representation 是 `facebook/wav2vec2-base`，mean pooling，LogisticRegression：`experiments/train_representation_model.py:20-27`、`74-77`
- fusion 是 acoustic + representation 拼接后 LogisticRegression：`experiments/train_fusion_model.py:34-45`

建议在文中补充这些超参数。课程论文不需要很长，但这些是复现性的最低信息。

---

### 实验结果与分析

`paper/final_report_cn.tex:158-186` 的表格数值与 `paper/self_experiment_results.md:1-19` 一致，这是优点。

需要调整的是解释强度：

- “wav2vec2 冻结表征优于...”应限定为 THCHS raw 的这个 pilot 设置。
- “自录孤立单字数据整体高于 THCHS 基线”可以保留，但要指出数据条件不同，不能视为公平同源比较。
- “融合模型取得最高结果”可以保留，但要说性能仍接近弱基线，不能证明模型已可可靠判分。

建议增加一句：

> 表中 THCHS 与 self 数据的比较主要反映任务条件差异，而不是语料之间的模型优劣；两者切分方式、录音场景和标签来源不同。

---

### 扩展实验设计

`paper/final_report_cn.tex:219-226` 很诚实，明确“后续实验”，这是好的。但如果 3–9 页篇幅紧张，这一节可以压缩为“局限与未来工作”的一段，不必单独成节。

建议改名为：

> 局限与扩展方向

原因：课程论文中单独“扩展实验设计”容易显得当前实验不足，而“局限与扩展方向”更符合论文收束节奏。

---

### 系统原型与课程关联

`paper/final_report_cn.tex:227-232` 这一节内容很好，但位置略晚。它其实是论文的核心贡献，应考虑前移到系统设计之后，或者与系统设计合并。现在读者读完实验才看到“课程关联”，节奏有些后置。

建议结构改为：

1. 引言
2. 语言学背景与相关工作
3. 系统设计与反馈规则
4. 数据与实验方法
5. 实验结果与误差分析
6. 局限、课程关联与未来工作
7. 结论

---

### 结论

`paper/final_report_cn.tex:235-236` 总体稳健。建议删除或弱化“质量过滤提升了最佳融合模型的宏平均 F1 值”中的因果语气，改为：

> 在当前过滤策略下，最佳融合模型的 macro-F1 高于未过滤设置。

---

## 建议增加或替换的参考文献

当前参考文献覆盖了 Chao、Gandour、Wang、CAPT、wav2vec2、HuBERT、THCHS、librosa、sklearn，基本够用。但若想更像语言科学/计算语音学论文，建议增加 3–5 篇，不要贪多。

### 普通话声调与二语学习

1. **Leather, J. (1990)**  
   Second-language speech research on Mandarin tone learning 常被引用，可支持“二语声调习得困难”。

2. **Wang, Jongman & Sereno 相关后续研究**  
   你已引用 Wang et al. 1999，可以再补一篇关于 perception-production 或 training generalization 的后续文献。

3. **Shen, X. S. 关于 Mandarin tone production/perception 的研究**  
   可支持普通话声调声学实现和学习者差异。

### CAPT / 发音反馈

4. **Cucchiarini, Neri, Strik 相关 CAPT review**  
   你已有 Neri 2002 和 Eskenazi 2009，若篇幅允许，可加一篇综述型文献，增强“反馈必须可理解”的依据。

### 预训练语音表征

5. **wav2vec2 + HuBERT 已够**  
   论文没有实际做 HuBERT/Whisper，不建议再大幅增加模型文献。可以保留 HuBERT 作为 related work，但不要让读者以为本文实验比较了 HuBERT。

### 工具文献

6. `librosa` 与 `scikit-learn` 引用可以保留，但格式要按 NCMMSC/GB/T 风格统一。`repo` 作为参考文献是否允许，要看课程模板；如果不允许，可把 GitHub 链接放在正文脚注或“代码可用性”句子中，不放参考文献表。

---

## 建议增加的实验或图

### 必须优先补的图

1. **自录过滤后混淆矩阵**  
   支持 tone4 弱项和 tone3 混淆。必须来自真实 `fusion_confusion_matrix.csv` 或重新运行生成。

2. **自录四声 F0 典型曲线图**  
   每声 2–3 条真实样本，叠加目标五度曲线。这个图最能支撑“语言科学解释反馈”的主贡献。

### 强烈建议补的表

3. **per-class precision/recall/F1 表**  
   如果篇幅有限，不一定放完整表，可在正文写关键类别，例如 tone4 recall/F1。前提是来自真实输出。

### 有时间再做

4. **质量过滤前后分布图**  
   展示各 tone/speaker 保留数量，解释 tone4 数据减少。

5. **LOSO bar chart**  
   将 speaker01/02/03 macro-F1 可视化，更直观说明说话人差异。

### 不建议临时新增、避免提交风险

6. 不建议现在临时做 HuBERT/Whisper 横向比较，除非已经有稳定输出。否则容易引入未验证结果和篇幅膨胀。

7. 不建议临时做人工评价，除非能清楚记录评分者、样本数、评分标准。否则审稿风险高于收益。

---

## 代码与论文一致性检查结论

### 支持论文说法的实现

- 后端支持音频读取、静音裁剪、重采样：`backend/app/audio.py:14-23`
- 后端支持 `librosa.pyin` F0 提取：`backend/app/features.py:77-87`
- 支持小范围 F0 不强行拉伸：`backend/app/features.py:4-15`
- 支持规则预测：`backend/app/inference.py:35-58`
- 支持目标五度曲线：`backend/app/inference.py:5-10`
- 支持规则反馈：`backend/app/feedback.py:24-42`
- 前端明确“预测仅供参考”：`frontend/src/components/FeedbackCard.tsx:18-25`
- 前端显示用户曲线与目标曲线：`frontend/src/components/ToneChart.tsx:3-22`
- 实验脚本支持 acoustic / wav2vec2 / fusion 三路径：
  - `experiments/train_acoustic_baseline.py:37-82`
  - `experiments/train_representation_model.py:60-95`
  - `experiments/train_fusion_model.py:21-55`

### 论文需要避免的潜在误读

1. 不要暗示 Web demo 使用离线 fusion 模型在线推理。代码中在线预测是规则启发式。
2. 不要暗示验证集参与了模型选择。当前脚本未使用 val。
3. 不要暗示 HuBERT/Whisper 已经实验。当前只是扩展方向。
4. 不要暗示 tone4 混淆已经用图表证明，除非补真实混淆矩阵。

---

## 最终提交前检查清单

### 必须完成

- [ ] 在方法节说明 val split 当前未用于模型选择，主结果使用预设 speaker-independent test split。
- [ ] 明确主实验 test speaker 与 LOSO 实验的关系。
- [ ] 将过滤提升的结论改成“当前固定设置下观察到”，避免因果过强。
- [ ] 补充或降级 tone4 与 tone3 混淆的说法。
- [ ] 增加至少一个自录数据相关图，优先混淆矩阵或四声 F0 示例。
- [ ] 检查参考文献格式是否符合 NCMMSC/中文会议模板。
- [ ] 确认全文页数仍在 3–9 页内；英文摘要放在参考文献后的一栏格式可能需要按模板调整。

### 强烈建议完成

- [ ] 在系统设计中加“在线 demo vs 离线实验”对照表。
- [ ] 在模型设置中补充 classifier、pooling、random_state、class_weight 等最小复现信息。
- [ ] 把“扩展实验设计”改为“局限与扩展方向”，减少像未完成工作清单的感觉。
- [ ] 在语言学背景中补一句五度曲线是教学近似，不是个体 Hz 模板。

### 有时间再增强

- [ ] 加 per-class F1 或 filtered fusion confusion matrix。
- [ ] 加质量过滤前后 tone/speaker 分布图。
- [ ] 加 LOSO macro-F1 bar chart。
- [ ] 补 2–3 篇二语普通话声调学习或 CAPT 综述文献。

---

## 最终判断

这篇论文已经有可提交基础，但还不应以“模型有效”作为卖点；最佳策略是把它打磨成一个诚实的语言科学 + 可解释 demo + 小规模实证项目。只要补清楚划分、图证据和结论边界，评审风险会明显降低。
