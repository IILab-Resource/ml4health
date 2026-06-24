# 模块 3：训练评估与结果分析

> 本模块是案例教程 9「机器学习建模 — 7 模型对比」的第四部分，承接模块 2（七个模型定义）。在 7 个模型定义就绪后，本模块运行训练和评估，得到 7 个模型在 5 个维度（AUC、Recall、Brier、Time、σ）上的完整结果，并深入分析核心发现。
>
> 本模块最核心的知识点有五个：**一是 LightGBM 是性能冠军**（AUC=0.9423，Brier=0.0974 最低）；**二是 KNN 的 Recall 最低**（0.8335），但不平衡程度决定 KNN 是否失效；**三是 SVM 与 RF 的 Recall 最高**（0.8984 并列）；**四是 LightGBM 性能最好但最慢**（64.64s，是 RF 的 35 倍）；**五是线性模型（LR/SVM）跨折方差最大**（σ=0.0033）。这五个发现共同回答了"7 个模型在同一数据上表现如何"。

***

## 学习目标

学完本模块后，你将能够：

1. **理解训练评估循环的代码结构**：如何用 `for name, model, scale_needed in models` 遍历 7 个模型，调用 `evaluate_model_cv` 评估。
2. **掌握完整对比表的解读**：7 个模型在 AUC、σ、Recall、Brier、Time 五个维度的排名。
3. **理解 LightGBM 性能冠军的原因**：AUC=0.9423 最高，Brier=0.0974 最低，梯度提升 + Leaf-wise 的优势。
4. **理解 KNN Recall 最低的原因**：Recall=0.8335，但不平衡程度决定 KNN 是否失效。
5. **理解 SVM 与 RF Recall 最高的原因**：class\_weight='balanced' 加权后，对存活类敏感性最好。
6. **理解 LightGBM 性能最好但最慢的权衡**：64.64s 是 RF 的 35 倍，但 AUC 差距仅 0.003。
7. **理解线性模型最不稳定的原因**：σ=0.0033 是 KNN 的近 3 倍。
8. **掌握稳定性与性能的二维分析**：第一象限（高 AUC + 高稳定）集中了 LGBM/XGB/RF。
9. **理解模型选择指南**：可解释性优先、性能优先、稳定性优先分别选什么模型。
10. **理解医疗 AI 论文的模型推荐**：基线模型、主模型、性价比之选、探索性分析。

***

## 一、训练评估循环

```python
# ============================================================================
# 3. 训练 & 评估所有模型
# ============================================================================
print("\n" + "=" * 70)
print("开始训练 & CV 评估...")
print("=" * 70)

all_results = []

for name, model, scale_needed in models:
    print(f"\n  [{name}]  Stratified 5-Fold CV...", end=' ')
    result = evaluate_model_cv(X, y, name, model, scale_needed)
    all_results.append(result)
    print(f"done.")
    print(f"    AUC = {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
    print(f"    Recall = {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
    print(f"    Brier = {result['brier_mean']:.4f} ± {result['brier_std']:.4f}")
    print(f"    Time = {result['time_total']:.2f}s ({result['time_mean']:.2f}s/fold)")
```

### 1.1 `all_results = []` — 初始化结果列表

`all_results` 列表存储所有模型的评估结果，每个元素是 `evaluate_model_cv` 返回的字典。

### 1.2 `for name, model, scale_needed in models:` — 遍历 7 个模型

`models` 列表的每个元素是 `(name, model, scale_needed)` 三元组（模块 2 定义）。这里用元组解包同时获取三个值。

### 1.3 `print(f"\n  [{name}]  Stratified 5-Fold CV...", end=' ')` — 进度提示

`end=' '` 让光标停在行末，不换行。等评估完成后用 `print(f"done.")` 接着打印"done."，形成"\[模型名] Stratified 5-Fold CV... done."的一行输出。

### 1.4 `result = evaluate_model_cv(X, y, name, model, scale_needed)` — 调用评估函数

调用模块 1 定义的 `evaluate_model_cv` 函数，传入特征矩阵、标签、模型名、模型实例、是否标准化。返回包含 AUC/Recall/Brier/Time 等字段的字典。

### 1.5 `all_results.append(result)` — 追加结果

把当前模型的评估结果追加到 `all_results` 列表。

### 1.6 打印每折指标

```python
print(f"    AUC = {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")
print(f"    Recall = {result['recall_mean']:.4f} ± {result['recall_std']:.4f}")
print(f"    Brier = {result['brier_mean']:.4f} ± {result['brier_std']:.4f}")
print(f"    Time = {result['time_total']:.2f}s ({result['time_mean']:.2f}s/fold)")
```

- `:.4f`：保留 4 位小数。
- `:.2f`：保留 2 位小数。
- `time_total`：5 折总耗时。
- `time_mean`：平均每折耗时。

***

## 二、实际运行结果（与 results/ 文件一致）

根据 `results/15_model_comparison.csv` 和 `results/15_model_comparison_summary.txt`，7 个模型的实际运行结果如下：

### 2.1 完整对比表

| 排名    | 模型                  | AUC        | σ(AUC)     | Recall     | Brier      | 时间(s) |
| ----- | ------------------- | ---------- | ---------- | ---------- | ---------- | ----- |
| **1** | **LightGBM**        | **0.9423** | 0.0016     | 0.8899     | **0.0974** | 64.64 |
| 2     | XGBoost             | 0.9415     | 0.0019     | 0.8863     | 0.0982     | 22.79 |
| 3     | Random Forest       | 0.9391     | 0.0022     | **0.8984** | 0.1002     | 1.83  |
| 4     | Decision Tree       | 0.9149     | 0.0018     | 0.8869     | 0.1145     | 0.12  |
| 5     | KNN (k=15)          | 0.9111     | **0.0012** | 0.8335     | 0.1185     | 0.06  |
| 6     | SVM (Linear)        | 0.8946     | 0.0033     | **0.8984** | 0.1347     | 0.18  |
| 7     | Logistic Regression | 0.8936     | 0.0033     | 0.8786     | 0.1310     | 0.06  |

### 2.2 性能排名（按 AUC）

```
1. LightGBM                  AUC=0.9423 ± 0.0016
2. XGBoost                   AUC=0.9415 ± 0.0019
3. Random Forest             AUC=0.9391 ± 0.0022
4. Decision Tree             AUC=0.9149 ± 0.0018
5. KNN (k=15)                AUC=0.9111 ± 0.0012
6. SVM (Linear)              AUC=0.8946 ± 0.0033
7. Logistic Regression       AUC=0.8936 ± 0.0033
```

### 2.3 稳定性排名（按 CV 标准差，越低越稳定）

```
1. KNN (k=15)                σ=0.0012 (CV=0.14%)
2. LightGBM                  σ=0.0016 (CV=0.17%)
3. Decision Tree             σ=0.0018 (CV=0.20%)
4. XGBoost                   σ=0.0019 (CV=0.20%)
5. Random Forest             σ=0.0022 (CV=0.24%)
6. Logistic Regression       σ=0.0033 (CV=0.37%)
7. SVM (Linear)              σ=0.0033 (CV=0.37%)
```

### 2.4 速度排名（按总耗时）

```
1. Logistic Regression       0.06s
2. KNN (k=15)                0.06s
3. Decision Tree             0.12s
4. SVM (Linear)              0.18s
5. Random Forest             1.83s
6. XGBoost                   22.79s
7. LightGBM                  64.64s
```

***

## 三、核心发现分析

### 3.1 发现 1：LightGBM 是性能冠军（AUC=0.9423）

> 💡💡💡 **重点发现：LightGBM 性能冠军**
>
> LightGBM 的 AUC=0.9423（最高），且 Brier=0.0974（最低，概率校准最好）。
>
> **前三名都是集成树模型**：
>
> - LightGBM: 0.9423
> - XGBoost: 0.9415
> - Random Forest: 0.9391
>
> 三者 AUC 差距很小（< 0.003），说明在该数据集上**梯度提升与 Bagging 都能很好地捕捉非线性关系**。
>
> **为什么 LightGBM 最强？**
>
> 1. 梯度提升：每棵树纠正前一棵的错误，逐步降低偏差。
> 2. Leaf-wise 分裂：比 Level-wise 收敛更快，增益更高。
> 3. 二阶优化：用梯度和 Hessian（二阶导数），比一阶梯度更精确。
> 4. 正则化：L1/L2 正则化防止过拟合。
>
> **Brier 最低的意义**：LightGBM 不仅排序好（AUC 高），概率估计也最准（Brier 低），这对临床决策至关重要——"存活概率 80%"和"存活概率 99%"对治疗决策影响不同。

### 3.2 发现 2：KNN 的 Recall 最低（0.8335）

> 💡 **重点发现：KNN Recall 最低但不失效**
>
> KNN 的 Recall=0.8335 在七个模型中最低，但已远非"灾难"级别。
>
> **原因分析**：
>
> ```
> 轻度不平衡 + class_weight 无效（KNN 不支持）
>         ↓
> KNN 仍能识别多数存活病例
>         ↓
> Recall ≈ 0.83 = 低于树模型集成，但不再失效
> ```
>
> **不平衡程度是决定 KNN 是否失效的关键**：
>
> - 严重不平衡下（如 1:20）：KNN 的近邻中几乎全是多数类，Recall 接近 0。
> - 轻度不平衡下（本数据集 41:59）：KNN 的近邻中有足够的正类样本，Recall=0.8335。
>
> 这说明**不平衡程度是决定 KNN 是否失效的关键**——在严重不平衡下 KNN 几乎检测不到少数类，而在轻度不平衡下仍可工作。

### 3.3 发现 3：SVM 与 Random Forest 的 Recall 最高（0.8984）

> 💡 **重点发现：SVM 和 RF 的 Recall 并列最高**
>
> 经过 `class_weight='balanced'` 加权后，SVM 和 RF 对存活类表现出最好的敏感性（Recall=0.8984，并列最高）。Logistic Regression（0.8786）紧随其后。
>
> **临床意义**：
> 在医学场景中高 Recall 非常有价值——**假阴性（漏诊存活患者）的代价可能高于假阳性**。如果一个可能存活的患者被预测为死亡，可能放弃治疗机会；而一个可能死亡的患者被预测为存活，只是多消耗一些医疗资源。
>
> **为什么 SVM 和 RF 的 Recall 最高？**
>
> 1. `class_weight='balanced'` 让模型更关注少数类（VIVO）。
> 2. SVM 的间隔最大化对边界样本敏感，加权后更倾向于把边界样本判为 VIVO。
> 3. RF 的 200 棵树投票，加权后更多树倾向于预测 VIVO。
>
> **注意**：Recall 高不等于 AUC 高。SVM 的 Recall=0.8984（并列最高），但 AUC=0.8946（倒数第二）。这是因为 SVM 在高 Recall 时可能牺牲了 Precision（把很多 MORTO 也预测为 VIVO）。

### 3.4 发现 4：LightGBM 性能最好但最慢（64.64s）

> 💡 **重点发现：性能 vs 速度的权衡**
>
> LightGBM 的 AUC=0.9423 排第一，但训练时间是 RF（1.83s）的 35 倍。XGBoost（22.79s）次之。
>
> **性价比分析**：
>
> | 模型            | AUC    | 时间(s) | 性价比（AUC/时间） |
> | ------------- | ------ | ----- | ----------- |
> | LightGBM      | 0.9423 | 64.64 | 0.0146      |
> | XGBoost       | 0.9415 | 22.79 | 0.0413      |
> | Random Forest | 0.9391 | 1.83  | **0.5132**  |
>
> RF 的性价比远高于 LGBM/XGB！
>
> **通常在更大数据集中梯度提升模型的优势会更明显**，但本数据集上三者（LightGBM/XGBoost/RF）AUC 差距很小（< 0.003），**RF 在性价比上更有优势**。
>
> **模型选择建议**：
>
> - 如果追求极致性能：LightGBM
> - 如果追求性价比：Random Forest
> - 如果需要快速迭代：Logistic Regression（0.06s）

### 3.5 发现 5：Logistic Regression 与 SVM 最不稳定（σ=0.0033）

> 💡 **重点发现：线性模型最不稳定**
>
> 线性模型的 AUC 跨折方差（σ=0.0033）是 KNN（0.0012）的近 3 倍。
>
> **稳定性排名**：
>
> 1. KNN: σ=0.0012（最稳定）
> 2. LightGBM: σ=0.0016
> 3. Decision Tree: σ=0.0018
> 4. XGBoost: σ=0.0019
> 5. Random Forest: σ=0.0022
> 6. LR/SVM: σ=0.0033（最不稳定）
>
> **为什么线性模型最不稳定？**
>
> 线性模型只能学习线性关系，而本数据集的特征与目标的关系是非线性的（如 Code.Profession 是分类变量被编码为数值，线性模型会学到错误的"线性关系"）。不同折的数据分布略有差异，线性模型的决策边界对这些差异更敏感，导致跨折方差大。
>
> **反直觉发现**：Decision Tree（σ=0.0018）反而较稳定。这与旧印象不同——在本数据集上单棵树的方差并不突出，可能因为深度限制（max\_depth=10）起到了正则化作用。

***

## 四、稳定性与性能的二维分析

### 4.1 二维分布图

```
AUC (性能, 越高越好)
    ↑
0.945 │  ● LGBM  ● XGB
0.940 │              ● RF
0.935 │
0.915 │                    ● DT
0.910 │                          ● KNN
0.895 │  ● LR   ● SVM
0.890 │
     ───────────────────────────────→ σ (稳定性, 越低越好)
      0.001   0.002   0.003

     ● 第一象限 (高AUC + 高稳定) 集中了 LGBM/XGB/RF
     ● KNN 最稳定 (σ=0.0012) 但 AUC 最低
     ● LR 和 SVM AUC 最低且最不稳定 (σ=0.0033)
```

### 4.2 四象限解读

> 💡 **重点概念：稳定性与性能的二维分析**
>
> 把 7 个模型放在"AUC vs σ"的二维平面上：
>
> **第一象限（高 AUC + 高稳定，左上）**：
>
> - LightGBM（AUC=0.9423, σ=0.0016）
> - XGBoost（AUC=0.9415, σ=0.0019）
> - Random Forest（AUC=0.9391, σ=0.0022）
> - **这是最理想的区域，集成树模型占据这里。**
>
> **第二象限（低 AUC + 高稳定，左下）**：
>
> - KNN（AUC=0.9111, σ=0.0012）
> - Decision Tree（AUC=0.9149, σ=0.0018）
> - **稳定但性能一般。**
>
> **第四象限（低 AUC + 低稳定，右下）**：
>
> - LR（AUC=0.8936, σ=0.0033）
> - SVM（AUC=0.8946, σ=0.0033）
> - **最不理想的区域，线性模型在这里。**
>
> **结论**：集成树模型（LGBM/XGB/RF）在性能和稳定性上都优于线性模型。

***

## 五、Recall vs AUC 的权衡

### 5.1 Recall vs AUC 分布图

```
回忆: Recall (VIVO 检出率)

0.90 ┤              ● SVM ● RF
0.89 ┤ ● LGBM
0.88 ┤ ● XGB  ● DT
0.87 ┤ ● LR
0.83 ┤ ● KNN
     └───────────────────────→ AUC
      0.89  0.91  0.93  0.95

● SVM 和 RF 的 Recall 最高（0.8984）→ 适合"不漏诊"场景
● LightGBM/XGBoost 的 AUC 最高 → 整体排序最优
● KNN 两项都偏低 → 不再是"灾难"但仍是最弱
```

### 5.2 权衡解读

> 💡 **重点概念：Recall vs AUC 的权衡**
>
> **SVM 和 RF 的 Recall 最高（0.8984）**：
>
> - 适合"不漏诊"场景——宁可多预测一些 VIVO（假阳性），也不要漏掉真正的 VIVO。
> - 但 SVM 的 AUC 只有 0.8946（倒数第二），说明它的高 Recall 是以牺牲 Precision 为代价的。
>
> **LightGBM/XGBoost 的 AUC 最高**：
>
> - 整体排序最优——把 VIVO 排在 MORTO 前面的能力最强。
> - 但 Recall（0.8899/0.8863）略低于 SVM/RF，因为在 0.5 阈值下，它们更"保守"。
>
> **KNN 两项都偏低**：
>
> - AUC=0.9111（倒数第三），Recall=0.8335（最低）。
> - 不再是"灾难"（轻度不平衡下 KNN 仍能工作），但仍是最弱。
>
> **临床选择**：
>
> - 如果"不漏诊"最重要：SVM 或 RF（Recall 最高）
> - 如果"整体排序"最重要：LightGBM 或 XGBoost（AUC 最高）
> - 如果"概率校准"最重要：LightGBM（Brier 最低）

***

## 六、Brier Score 与不平衡程度的关系

### 6.1 Brier Score 对比

| 模型       | Brier  | 解读                           |
| -------- | ------ | ---------------------------- |
| KNN      | 0.1185 | 不再异常低 → 轻度不平衡下 KNN 不再"全预测死亡" |
| LR       | 0.1310 | 线性模型校准一般                     |
| RF       | 0.1002 | 明显优于 LR，概率估计更准确              |
| LightGBM | 0.0974 | 最低，校准最好                      |

### 6.2 Brier Score 解读

> 💡 **重点概念：Brier Score 与不平衡程度的关系**
>
> **教学要点**：在严重不平衡数据中 Brier Score 需要警惕——模型只预测多数类时 Brier 会假性偏低（因为多数类概率高，Brier 看起来好）。
>
> 本数据集为**轻度不平衡**，该陷阱不再明显，但**仍建议和 Recall 一起看**以确认模型对少数类的识别能力。
>
> **Brier 排名**：
>
> 1. LightGBM: 0.0974（最低，校准最好）
> 2. XGBoost: 0.0982
> 3. RF: 0.1002
> 4. DT: 0.1145
> 5. KNN: 0.1185
> 6. LR: 0.1310
> 7. SVM: 0.1347（最高，校准最差）
>
> **为什么 SVM 的 Brier 最高？**
>
> SVM 用 `decision_function` + Sigmoid 转换得到概率，这种转换不是严格校准的——SVM 的 decision\_function 没有概率语义，Sigmoid 只是把分数压到 \[0, 1]。所以 SVM 的概率估计最不准确。
>
> **为什么 LightGBM 的 Brier 最低？**
>
> LightGBM 优化的是 logloss（对数损失），logloss 和 Brier 都衡量概率质量。LightGBM 的二阶优化（用 Hessian）让概率估计更精确。

***

## 七、模型选择指南

### 7.1 当模型选择取决于目标

```
┌─────────────────────────────────────────────────────────┐
│                    你的目标是什么?                        │
└─────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  可解释性优先           性能优先             稳定性优先
         │                    │                    │
         ▼                    ▼                    ▼
  Logistic Regression    Random Forest        Random Forest
  Decision Tree          XGBoost              Logistic Regression
  (医生能看懂)           (模型竞赛首选)        (低方差)
```

### 7.2 医疗论文推荐

> 💡 **重点概念：医疗 AI 论文的模型推荐**
>
> 从本教程的结果出发，给医疗 AI 论文的建议：
>
> | 场景    | 推荐模型                    | 理由                  |
> | ----- | ----------------------- | ------------------- |
> | 基线模型  | Logistic Regression     | 可解释、速度快、Recall 高    |
> | 主模型   | **LightGBM / XGBoost**  | 性能冠军、AUC 最高         |
> | 性价比之选 | Random Forest           | AUC 接近冠军、训练快（1.83s） |
> | 探索性分析 | Decision Tree           | 可视化决策规则             |
> | 审稿人要求 | Stratified 5-Fold + 全报告 | 本教程的完整评估框架          |
>
> **典型医疗 AI 论文的模型配置**：
>
> 1. 用 LR 作为基线（可解释、Recall 高）
> 2. 用 LightGBM/XGBoost 作为主模型（AUC 最高）
> 3. 用 RF 作为对比（性价比）
> 4. 报告 AUC + Recall + Brier + 5 折 CV 标准差
> 5. 加 SHAP 解释（案例教程 15-24 会讨论）

***

## 八、如果样本量变化，排名会变吗？

> 💡 **重点概念：模型排名与样本量的关系**
>
> 模型排名不是固定的。它是样本量、特征数、信号噪声比的联合函数。
>
> | 样本量                          | 预测的排名变化                                       |
> | ---------------------------- | --------------------------------------------- |
> | n < 500                      | SVM 可能下降，LR 和 DT 更可靠                          |
> | 500 < n < 5,000              | RF 开始超越 LR                                    |
> | 5,000 < n < 50,000 **(本实验)** | LightGBM > XGBoost > RF > DT > KNN > SVM > LR |
> | n > 100,000                  | XGBoost/LightGBM 可能超越 RF                      |
>
> **本实验的 20,000 样本落在"5,000 < n < 50,000"区间**，展示了典型的"中等样本量"模型排名：集成树模型 > 单棵树 > KNN > 线性模型。
>
> **教学要点**：不要把本教程的排名当作"永恒真理"。在不同的数据集上，排名可能完全不同。这就是为什么需要"在同一数据上用同一评估框架比较多个模型"——本教程的核心方法论。

***

## 小贴士

1. **LightGBM 是性能冠军**：AUC=0.9423 最高，Brier=0.0974 最低，但训练 64.64s 最慢。
2. **Random Forest 是性价比之王**：AUC=0.9391（仅低 0.003），训练 1.83s（快 35 倍）。
3. **KNN 在轻度不平衡下不失效**：Recall=0.8335 最低但非灾难，不平衡程度决定 KNN 是否失效。
4. **SVM 和 RF 的 Recall 最高**：0.8984 并列，适合"不漏诊"场景。
5. **线性模型最不稳定**：σ=0.0033 是 KNN 的近 3 倍，因为线性模型对非线性关系敏感。
6. **Decision Tree 反直觉地稳定**：σ=0.0018，因为 max\_depth=10 起到了正则化作用。
7. **Brier Score 要和 Recall 一起看**：轻度不平衡下陷阱减弱，但仍建议联合查看。
8. **模型排名随样本量变化**：本实验的排名适用于 5,000–50,000 样本区间。
9. **五个维度联合解读**：AUC + Recall + Brier + Time + σ，缺一不可。
10. **没有"最好"的模型**：选择取决于可解释性 vs 性能 vs 稳定性 vs 速度的权衡。

***

## 常见问题

**Q1：LightGBM 的 AUC=0.9423 和 XGBoost 的 0.9415 差距只有 0.0008，这个差距显著吗？**

A：从统计角度看，0.0008 的差距可能不显著（两者的标准差分别是 0.0016 和 0.0019，差距小于 1 个标准差）。但从实践角度看，LightGBM 还在 Brier 上更低（0.0974 vs 0.0982），综合来看 LightGBM 略优。在论文中，应该报告两者并说明"差距不显著"，而不是宣称 LightGBM "显著优于" XGBoost。

**Q2：为什么 KNN 的 Recall 最低（0.8335），但 AUC 不低（0.9111）？**

A：因为 AUC 衡量的是"排序能力"，Recall 衡量的是"在 0.5 阈值下的检出率"。KNN 的排序能力不差（AUC=0.9111），但在 0.5 阈值下，它的预测偏向 MORTO（因为 KNN 不支持 class\_weight，没有加权）。如果降低阈值（如 0.3），KNN 的 Recall 会提高，但 Precision 会下降。

**Q3：SVM 的 Recall=0.8984（并列最高），但 AUC=0.8946（倒数第二），为什么？**

A：因为 SVM 在高 Recall 时牺牲了 Precision。`class_weight='balanced'` 让 SVM 倾向于把边界样本判为 VIVO，提高了 Recall，但也把很多 MORTO 误判为 VIVO（假阳性增加）。AUC 衡量整体排序，不受单一阈值影响，所以 SVM 的 AUC 不高。

**Q4：为什么 LightGBM 训练 64.64s，比 XGBoost 的 22.79s 慢？**

A：因为本教程只有 20,000 样本，LightGBM 的 GOSS 和 EFB 优化在小数据上开销超过收益。GOSS 需要计算所有样本的梯度并排序，EFB 需要分析特征互斥性，这些预处理在大数据上才划算。在百万级样本上，LightGBM 会比 XGBoost 快 2-10 倍。

**Q5：Decision Tree 的 σ=0.0018，比 RF 的 0.0022 还稳定，这正常吗？**

A：这是本数据集的反直觉发现。通常单棵树比方差比集成高，但本教程中 DT 的 max\_depth=10 起到了正则化作用，限制了树的复杂度，降低了方差。而 RF 的 max\_depth=12 更深，单棵树方差更大，但集成后应该降低——这里 RF 的 σ=0.0022 是集成后的，仍然略高于 DT，可能是因为 RF 的 bootstrap 采样引入了额外随机性。

**Q6：Brier Score 0.0974（LightGBM）算好还是差？**

A：Brier Score 的绝对值没有统一标准，取决于数据集难度。一般来说：

- Brier < 0.1：校准较好
- Brier 0.1–0.2：校准一般
- Brier > 0.2：校准较差

LightGBM 的 0.0974 刚好低于 0.1，校准较好。SVM 的 0.1347 最高，因为 Sigmoid 转换不是严格校准。

**Q7：如果我要写医疗 AI 论文，应该选哪个模型？**

A：根据教学文档的建议：

- **基线模型**：Logistic Regression（可解释、Recall 高）
- **主模型**：LightGBM 或 XGBoost（AUC 最高）
- **性价比对比**：Random Forest（训练快、AUC 接近）
- **评估框架**：Stratified 5-Fold + AUC + Recall + Brier + σ
- **解释**：加 SHAP（案例教程 15-24）

**Q8：为什么 LR 和 SVM 的 AUC 几乎相同（0.8936 vs 0.8946）？**

A：因为它们都是线性模型，决策边界都是线性的。在本数据集上，线性决策边界的性能上限就是约 0.894。两者的微小差异来自优化算法不同（LR 用拟牛顿法，SVM 用 liblinear）和损失函数不同（LR 用 logloss，SVM 用 hinge）。

**Q9：如果样本量变成 100 万，排名会怎么变？**

A：根据教学文档 5.6 节，n > 100,000 时 XGBoost/LightGBM 可能超越 RF（本教程 RF 已经接近）。在百万级样本上：

- LightGBM 会比 XGBoost 快 2-10 倍（GOSS + EFB 生效）
- 梯度提升模型的优势更明显
- LR/SVM 仍然是最快的基线
- KNN 的预测会变得极慢（要计算 100 万距离）

**Q10：本教程的排名能推广到其他数据集吗？**

A：不能直接推广。模型排名是样本量、特征数、信号噪声比的联合函数。本教程的排名适用于"中等样本量（5,000–50,000）、表格数据、轻度不平衡"的场景。在其他场景下：

- 图像数据：CNN 远优于树模型
- 文本数据：Transformer 远优于树模型
- 严重不平衡：需要重采样 + 调整 class\_weight
- 小样本（< 500）：LR 和 DT 更可靠

***

## 本模块小结

### 核心知识点回顾

1. **训练评估循环**：`for name, model, scale_needed in models` 遍历 7 个模型，调用 `evaluate_model_cv` 评估。
2. **完整对比表**（与 results/ 文件一致）：
   - LightGBM: AUC=0.9423, σ=0.0016, Recall=0.8899, Brier=0.0974, Time=64.64s
   - XGBoost: AUC=0.9415, σ=0.0019, Recall=0.8863, Brier=0.0982, Time=22.79s
   - RF: AUC=0.9391, σ=0.0022, Recall=0.8984, Brier=0.1002, Time=1.83s
   - DT: AUC=0.9149, σ=0.0018, Recall=0.8869, Brier=0.1145, Time=0.12s
   - KNN: AUC=0.9111, σ=0.0012, Recall=0.8335, Brier=0.1185, Time=0.06s
   - SVM: AUC=0.8946, σ=0.0033, Recall=0.8984, Brier=0.1347, Time=0.18s
   - LR: AUC=0.8936, σ=0.0033, Recall=0.8786, Brier=0.1310, Time=0.06s
3. **五大核心发现**：
   - **发现 1**：LightGBM 性能冠军（AUC=0.9423，Brier=0.0974 最低）
   - **发现 2**：KNN Recall 最低（0.8335），但不平衡程度决定 KNN 是否失效
   - **发现 3**：SVM 与 RF Recall 最高（0.8984 并列），适合"不漏诊"场景
   - **发现 4**：LightGBM 性能最好但最慢（64.64s，是 RF 的 35 倍）
   - **发现 5**：线性模型最不稳定（σ=0.0033，是 KNN 的近 3 倍）
4. **稳定性与性能二维分析**：
   - 第一象限（高 AUC + 高稳定）：LGBM/XGB/RF
   - 第二象限（低 AUC + 高稳定）：KNN/DT
   - 第四象限（低 AUC + 低稳定）：LR/SVM
5. **Recall vs AUC 权衡**：
   - SVM/RF：Recall 最高，适合"不漏诊"
   - LGBM/XGB：AUC 最高，整体排序最优
   - KNN：两项都偏低
6. **Brier Score 与不平衡程度**：
   - 轻度不平衡下陷阱减弱
   - LightGBM Brier 最低（0.0974）
   - SVM Brier 最高（0.1347，因为 Sigmoid 转换不校准）
7. **模型选择指南**：
   - 可解释性优先：LR / DT
   - 性能优先：XGBoost / LightGBM
   - 稳定性优先：RF / LR
   - 速度优先：LR
   - 医疗论文：LR（基线）+ LGBM/XGB（主模型）
8. **模型排名随样本量变化**：
   - 本实验（20,000 样本）：LGBM > XGB > RF > DT > KNN > SVM > LR
   - 排名不是固定的，是样本量、特征数、信号噪声比的联合函数

### 五维度对比图

```
维度          LR      SVM     KNN     DT      RF      XGB     LGBM
─────────────────────────────────────────────────────────────────
AUC(↑)      0.8936  0.8946  0.9111  0.9149  0.9391  0.9415  0.9423  ← LGBM 最高
σ(↓)        0.0033  0.0033  0.0012  0.0018  0.0022  0.0019  0.0016  ← KNN 最稳定
Recall(↑)   0.8786  0.8984  0.8335  0.8869  0.8984  0.8863  0.8899  ← SVM/RF 最高
Brier(↓)    0.1310  0.1347  0.1185  0.1145  0.1002  0.0982  0.0974  ← LGBM 最低
Time(↓)     0.06s   0.18s   0.06s   0.12s   1.83s   22.79s  64.64s  ← LR/KNN 最快
```

### 下一部分预告

本模块完成了训练评估和结果分析，得到了 7 个模型在 5 个维度上的完整结果。接下来的模块 4 将可视化这些结果：

**模块 4：可视化与结果保存**

- 生成 6 张图：性能条形图（12a）、稳定性雷达图（12b）、速度 vs 性能散点图（12c）、ROC 曲线（12d）、PR 曲线（12e）、5 折 AUC 波动图（12f）
- 保存 CSV 和 TXT 结果文件
- 理解每张图的设计意图和解读方法

让我们带着"如何用可视化展示 7 模型对比结果"的目标，进入模块 4 的可视化！

***

