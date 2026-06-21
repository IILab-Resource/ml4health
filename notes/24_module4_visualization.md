# 模块 4：可视化与结果解读

> 在前三个模块中，我们完成了数据加载、缺失值分析、四种插补方法的实现与建模评估。本模块将把所有数字结果转化为**可视化图表**，并通过图表深入解读不同插补方法对模型性能、特征分布的影响。
>
> 可视化是数据科学沟通的核心技能——一张好图胜过千言万语的数据表。

---

## 学习目标

完成本模块后，你将能够：

1. **掌握 matplotlib 子图布局**：使用 `plt.subplots()` 创建 1×3、2×2 等多子图布局，并理解 `zip(axes, ...)` 并行遍历的编程技巧。
2. **理解六类核心评估图表**：性能对比柱状图、ROC 曲线、校准曲线、训练集规模对比、分布影响图、PR 曲线的绘制原理与解读方法。
3. **解读校准曲线**——机器学习中最容易被忽视但极其重要的概念：理解"预测概率"与"实际频率"的关系，判断模型是高估还是低估风险。
4. **识别均值插补的"假峰现象"**：通过直方图叠加对比，直观看到均值插补如何破坏数据分布。
5. **综合分析实验结果**：理解为什么在缺失率较低（< 15%）时，不同插补方法对模型性能影响有限，以及如何根据实际场景选择合适方法。
6. **区分 ROC 曲线与 PR 曲线**的适用场景，理解为什么在不平衡数据集中 PR 曲线更具诊断价值。

---

## 本模块在整体流程中的位置

```
案例教程 3 流程:
┌─────────────────────────────────────────────────────────────┐
│ 模块 1: 数据加载与缺失值分析 (识别问题)                       │
│ 模块 2: 四种插补方法实现 (Mean / KNN / MICE / Complete Case)  │
│ 模块 3: 模型训练与评估指标计算 (AUC / Recall / Brier / PR-AUC) │
│ 模块 4: 可视化与结果解读 ← 你在这里                           │
└─────────────────────────────────────────────────────────────┘
```

本模块对应源代码 `src/03_preprocessing_imputation.py` 的第 290–582 行，共生成 **6 张可视化图表**（图 6a–6f）。

---

## 实验结果回顾

在深入可视化之前，先回顾四种方法的实际运行结果（这是所有图表的数据来源）：

| 方法 | AUC | Recall | Brier | PR-AUC | 训练样本 | 耗时 |
|------|-----|--------|-------|--------|----------|------|
| Complete Case | 0.8644 | 0.9240 | 0.1493 | 0.6879 | 47,270 | 0.0s |
| Mean Imputation | 0.8632 | 0.9236 | 0.1466 | 0.7256 | 56,000 | 0.0s |
| KNN Imputation | 0.8628 | 0.9237 | 0.1467 | 0.7247 | 56,000 | 19.5s |
| MICE Imputation | 0.8637 | 0.9223 | 0.1462 | 0.7263 | 56,000 | 0.1s |

**Age 特征方差对比**：原始 274.41，Mean 273.99，KNN ≈ 274，MICE ≈ 274

> 💡 **观察提示**：请注意 AUC 列四个数字非常接近（差距 < 0.002），但 PR-AUC 列差异更明显（Complete Case 0.6879 vs MICE 0.7263）。这种"分指标差异不同"的现象，正是本模块要解读的核心。

---

## 图 6a：性能对比柱状图

### 1.1 我们要解决什么问题？

我们有一张包含 4 个方法 × 4 个指标的表格，但表格数字密集，难以一眼看出"哪个方法在哪个指标上更好"。柱状图通过**高度对比**让差异可视化。

### 1.2 代码逐行解析

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
```

- `plt.subplots(1, 3)`：创建 **1 行 3 列**的子图网格，共 3 个子图并排显示。
- `figsize=(15, 5)`：整个画布宽 15 英寸、高 5 英寸。每个子图约 5×5 英寸，是正方形比例，便于柱状图展示。
- 返回值 `fig` 是整张画布，`axes` 是包含 3 个子图坐标轴对象的 NumPy 数组（长度为 3）。

```python
metrics_to_plot = [
    ('AUC', 'AUC (ROC)', 'higher is better', False),
    ('Recall', 'Recall (VIVO)', 'higher is better', False),
    ('Brier_Score', 'Brier Score', 'lower is better', True),
]
```

这是一个**列表 of 元组**，每个元组有 4 个元素：

| 位置 | 字段名 | 含义 | 示例值 |
|------|--------|------|--------|
| 0 | `metric` | 结果字典中的键名（用于取值） | `'AUC'` |
| 1 | `title` | 子图标题显示文本 | `'AUC (ROC)'` |
| 2 | `note` | 副标题提示优劣方向 | `'higher is better'` |
| 3 | `invert` | 是否需要反转 Y 轴（Brier 越低越好） | `False` / `True` |

> 💡 **设计哲学**：用元组列表+循环代替三段重复代码，是 DRY（Don't Repeat Yourself）原则的典型应用。如果将来要加 F1-score，只需在列表里加一行，循环代码完全不用改。

```python
for ax, (metric, title, note, invert) in zip(axes, metrics_to_plot):
```

- `zip(axes, metrics_to_plot)`：把 3 个子图轴对象和 3 个元组**配对**，生成 `[(ax0, 元组0), (ax1, 元组1), (ax2, 元组2)]`。
- `for ax, (metric, title, note, invert) in ...`：每次循环解包两个值——`ax` 是当前子图，`(metric, title, note, invert)` 是元组解包后的 4 个变量。
- 这种写法让"第 i 个子图画第 i 个指标"一目了然，避免下标 `axes[i]` 的混乱。

```python
methods_names = [r['Method'] for r in results]
values = [r[metric] for r in results]
colors = [r['Color'] for r in results]
```

从 `results` 列表（每个元素是一个方法的结果字典）中提取三个并行列表：
- `methods_names`：`['Complete Case', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']`
- `values`：当前指标对应的 4 个数值
- `colors`：每个方法固定的颜色编码（用于全图统一识别）

> 🎨 **颜色一致性原则**：Complete Case=灰 `#7f8c8d`、Mean=蓝 `#3498db`、KNN=橙 `#e67e22`、MICE=紫 `#9b59b6`。这套配色在所有 6 张图中保持一致，读者一眼就能识别方法。

```python
bars = ax.bar(methods_names, values, color=colors, edgecolor='white', width=0.6)
```

`ax.bar()` 参数详解：
- `methods_names`：X 轴标签（4 个方法名）。
- `values`：柱子高度（4 个指标值）。
- `color=colors`：每个柱子填充颜色（列表对应）。
- `edgecolor='white'`：柱子边缘描白色边框，让相邻柱子有视觉分隔。
- `width=0.6`：柱子宽度占分类间距的 60%（默认 0.8），稍窄显得更精致。
- 返回值 `bars` 是 BarContainer 对象，包含 4 个 Rectangle，后续用于在柱顶标注数值。

```python
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
```

这是**在柱顶标注数值**的关键代码：
- `bar.get_x() + bar.get_width() / 2`：柱子的水平中心点 X 坐标。
- `bar.get_height()`：柱子顶部 Y 坐标（即数值本身）。
- `f'{val:.4f}'`：格式化为 4 位小数（如 `0.8644`）。
- `ha='center'`（水平居中）、`va='bottom'`（文字底部对齐柱顶，文字在柱子上方）。
- `fontsize=10, fontweight='bold'`：字号 10、加粗，确保可读性。

```python
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

隐藏子图的上边框和右边框（"spines"是坐标轴边线），只保留左 Y 轴和下 X 轴。这是现代数据可视化的常见审美——减少视觉噪音。

```python
ax.set_xticklabels(methods_names, rotation=20, ha='right', fontsize=9)
```

X 轴标签旋转 20°、右对齐，避免长方法名（如 "MICE Imputation"）重叠。

### 1.3 图表解读

![性能对比柱状图](../img/06a_performance_comparison.png)

**从图中能读出什么？**

1. **AUC 子图（左）**：四个柱子高度几乎相同（0.8628–0.8644），肉眼难以区分。Complete Case 略高，但差距 < 0.002，统计上不显著。
2. **Recall 子图（中）**：同样几乎一致（0.9223–0.9240），说明四种方法对"找到存活患者"的能力相当。
3. **Brier Score 子图（右）**：标注"lower is better"。Complete Case 最高（0.1493，最差），三种插补方法都更低（0.1462–0.1467），MICE 最优。

> 📌 **关键洞察**：AUC 和 Recall 几乎看不出差异，但 Brier Score（校准度）能区分出 Complete Case 较差。这说明**单一指标会掩盖差异**，必须多维度评估。

---

## 图 6b：ROC 曲线

### 2.1 ROC 曲线基础回顾

ROC（Receiver Operating Characteristic）曲线是二分类模型的"黄金标准"评估图：
- **X 轴**：False Positive Rate (FPR) = 假阳性率 = FP / (FP + TN)
- **Y 轴**：True Positive Rate (TPR) = 真阳性率 = Recall = TP / (TP + FN)
- 曲线从 (0,0) 走到 (1,1)，**越靠近左上角越好**。
- AUC（曲线下面积）∈ [0, 1]，0.5 表示随机猜测，1.0 表示完美分类。

### 2.2 代码逐行解析

```python
fig, ax = plt.subplots(figsize=(9, 7))

for r in results:
    method = r['Method']
    color = r['Color']
    X_t, _, y_t, _ = imputed_datasets[method]
    lr = r['Model']
```

遍历 4 个方法的结果字典。`r['Model']` 存储了已训练好的 LogisticRegression 对象，可以直接调用 `predict_proba()` 而无需重新训练。

```python
    y_prob_roc = lr.predict_proba(X_test_imp if method != 'Complete Case'
                                   else imputed_datasets['Complete Case'][1])[:, 1]
    y_true_roc = (y_test_imp if method != 'Complete Case'
                  else imputed_datasets['Complete Case'][3])
```

**为什么 Complete Case 要单独处理？** 这是本图最关键的细节：

- 三种插补方法（Mean/KNN/MICE）的测试集是**完整测试集**（24,000 行），因为插补填补了所有缺失值。
- Complete Case 删除了测试集中含缺失的行，测试集只剩约 20,000 行。
- 如果用完整测试集评估 Complete Case 模型，模型无法处理 NaN 会报错；如果用删减后的测试集评估插补模型，则不公平。
- 因此必须为每个方法匹配它**自己处理过的测试集**：
  - `imputed_datasets['Complete Case'][1]` 是 Complete Case 删减后的 X_test
  - `imputed_datasets['Complete Case'][3]` 是对应的 y_test
  - `X_test_imp` 和 `y_test_imp` 是循环外保留的"最后一次插补"的测试集（三种插补方法结果相同，因为删除行数相同）

> ⚠️ **常见陷阱**：初学者常犯的错误是所有方法都用同一个测试集。这会导致 Complete Case 模型在含 NaN 的数据上崩溃，或者评估不公平。**评估数据必须与训练数据经过相同的预处理**。

```python
    fpr, tpr, _ = roc_curve(y_true_roc, y_prob_roc)
```

`roc_curve()` 返回三个值：
- `fpr`：假阳性率数组（从 0 到 1）
- `tpr`：真阳性率数组（从 0 到 1）
- `_`：阈值数组（thresholds），我们这里不用，用下划线忽略

它的工作原理：扫描所有可能的分类阈值（从 1.0 降到 0.0），在每个阈值下计算 (FPR, TPR) 点，连成曲线。

```python
    auc_val = r['AUC']
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f'{method} (AUC = {auc_val:.4f})')
```

绘制曲线：
- `color=color`：使用该方法对应的固定颜色。
- `linewidth=2`：线宽 2，确保在重叠时仍可辨识。
- `label=...`：图例文本，包含方法名和 AUC 数值（4 位小数）。

```python
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
```

**对角线参考线**——随机分类器的 ROC 曲线：
- `[0, 1], [0, 1]`：从 (0,0) 到 (1,1) 的直线。
- `'k--'`：`k` 表示黑色（black），`--` 表示虚线。
- `alpha=0.5`：半透明，不喧宾夺主。
- 这条线代表"随机猜测"（AUC=0.5），任何有效模型的曲线都应**位于该线上方**。

### 2.3 图表解读

![ROC曲线](../img/06b_roc_curves.png)

**从图中能读出什么？**

1. **四条曲线高度重叠**：在视觉上几乎无法区分，都明显位于对角线上方，说明四种方法都远优于随机猜测。
2. **AUC 数值差异极小**：0.8628–0.8644，差距仅 0.0016，统计上不显著。
3. **曲线形状相似**：在低 FPR 区域（左下）快速上升，说明模型在保持低误报率时仍能捕获大量正例。

> 📌 **结论**：ROC 曲线告诉我们，**在区分能力（AUC）上，四种插补方法几乎没有差别**。要看出差异，需要看后续的校准曲线和 PR 曲线。

---

## 图 6c：校准曲线（Calibration Curve）——重要概念

### 3.1 什么是校准？为什么重要？

**校准（Calibration）**回答一个问题：*"模型说'这个患者有 80% 存活概率'时，这类患者中真的有 80% 存活吗？"*

- **区分能力（AUC）**：模型能否把高风险和低风险患者**排序**正确？
- **校准能力（Brier）**：模型的预测概率**数值**是否准确反映真实频率？

> 💡 **临床场景**：医生根据模型预测的"90% 存活概率"决定治疗方案。如果模型实际只有 70% 存活率却预测 90%，会导致过度乐观的治疗决策。**校准在医学预测模型中比 AUC 更重要**——这是 WHO 和 TRIPOD 指南强调的原则。

### 3.2 校准曲线的绘制原理

```python
prob_true, prob_pred = calibration_curve(
    y_test_imp, y_pred_proba, n_bins=10, strategy='uniform'
)
```

`calibration_curve()` 的工作流程：
1. **分箱**：将预测概率 [0, 1] 等宽分成 10 个箱子（每箱宽 0.1）：[0-0.1], [0.1-0.2], ..., [0.9-1.0]。
2. **统计每箱**：
   - `prob_pred`（X 轴）：该箱内所有样本的**预测概率均值**。
   - `prob_true`（Y 轴）：该箱内所有样本的**实际正例比例**。
3. **连线**：将 10 个 (prob_pred, prob_true) 点连成曲线。

**解读规则**：
- 曲线在对角线上 → 完美校准
- 曲线在对角线**上方** → 模型**低估**了概率（实际正例比预测的多）
- 曲线在对角线**下方** → 模型**高估**了概率（实际正例比预测的少）

### 3.3 代码逐行解析

```python
fig, ax = plt.subplots(figsize=(9, 7))

for r in results:
    ax.plot(r['Calibration_Pred'], r['Calibration_True'],
            marker='o', color=r['Color'], linewidth=2, markersize=8,
            label=f"{r['Method']} (Brier={r['Brier_Score']:.4f})")
```

- `r['Calibration_Pred']` 和 `r['Calibration_True']`：在模块 3 中预先计算好的 10 个分箱点。
- `marker='o'`：每个分箱点用圆形标记，便于看出离散点而非连续曲线。
- `markersize=8`：标记较大，强调"这是分箱统计结果"。
- `label` 中包含 Brier Score，因为 Brier 是校准的数值指标（越低越好）。

```python
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect Calibration')
```

**对角线 = 完美校准参考线**：
- 如果模型预测概率完全等于实际频率，所有点都落在这条线上。
- 这条线是"理想状态"，现实模型通常会有偏离。

```python
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Observed Fraction of Positives', fontsize=12)
```

- X 轴：平均预测概率（模型输出的概率均值）
- Y 轴：观察到的正例比例（真实标签的均值）

### 3.4 图表解读

![校准曲线](../img/06c_calibration_curves.png)

**从图中能读出什么？**

1. **四条曲线都接近对角线**：说明所有方法的校准都不错，没有严重高估或低估。
2. **Complete Case（灰色）偏离最大**：其 Brier Score 最高（0.1493），曲线在某些区间偏离对角线更明显。
3. **MICE（紫色）最贴近对角线**：Brier 最低（0.1462），校准最佳。
4. **图例中的 Brier 数值**：直接显示在图例中，便于横向对比。

> 📌 **核心结论**：插补方法（Mean/KNN/MICE）的校准都优于 Complete Case。这是因为 Complete Case 删除了约 15% 的样本，导致训练分布与全量分布有偏，模型校准受影响。

> 💡 **小贴士**：校准曲线的"分箱数"会影响曲线平滑度。`n_bins=10` 是常用值，样本量少时可降到 5，样本量大时可增到 20。`strategy='uniform'` 是等宽分箱，`strategy='quantile'` 是等频分箱（每箱样本数相同，更适合不平衡数据）。

---

## 图 6d：训练集规模对比

### 4.1 为什么要看训练集规模？

训练样本数量直接影响：
- **统计效力**：样本越多，模型估计的方差越小，泛化越稳定。
- **稀有类别学习**：少数类样本越多，模型越能学到稀有模式。
- **计算成本**：样本越多，训练时间越长（尤其 KNN）。

### 4.2 代码逐行解析

```python
fig, ax = plt.subplots(figsize=(9, 6))
methods_names = [r['Method'] for r in results]
train_sizes = [r['Training_Size'] for r in results]
colors = [r['Color'] for r in results]

bars = ax.bar(methods_names, train_sizes, color=colors, edgecolor='white', width=0.6)
for bar, val in zip(bars, train_sizes):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
```

代码结构与图 6a 的单子图完全一致，区别在于：
- `train_sizes` 是整数（47270, 56000, 56000, 56000）。
- `f'{val:,}'`：使用千分位逗号格式化（如 `56,000`），便于阅读大数字。

```python
ax.set_title('Training Set Size After Imputation', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Training Samples')
```

标题和 Y 轴标签清晰说明"这是插补后的训练集规模"。

### 4.3 图表解读

![训练集规模](../img/06d_training_size.png)

**从图中能读出什么？**

1. **Complete Case 只有 47,270 样本**：明显矮于其他三个柱子。
2. **三种插补方法都是 56,000 样本**：因为插补保留了所有原始训练样本（80,000 × 0.7 = 56,000）。
3. **样本量差距**：56,000 - 47,270 = **8,730 行被删除**，约占训练集 15.6%。

> ⚠️ **为什么 Complete Case 损失 15.6%？** 本数据集中 `Raca.Color`（人种）字段缺失率 15.31%，是缺失主因。Complete Case 删除任何含缺失的行，只要某行在 Raca.Color 上缺失就被删除，即使其他字段完整。这就是 Complete Case 在高缺失率特征上的"信息损失放大"效应。

> 💡 **权衡思考**：Complete Case 虽然样本少，但保留了"真实观测"——没有人工填补的噪声。三种插补方法样本多，但引入了插补不确定性。这是"数据真实性"与"数据完整性"的权衡。

---

## 图 6e：插补前后分布对比（Age）——最重要的图

### 5.1 为什么这张图最重要？

前面的图 6a-6d 关注**模型层面**的差异（AUC、校准、样本量），但模型差异的**根本原因**在于插补如何改变了**特征分布**。图 6e 直接展示"插补前后的数据长什么样"，是理解所有性能差异的钥匙。

### 5.2 子图布局：2×2 网格

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
```

- `2, 2`：2 行 2 列，共 4 个子图。
- `figsize=(14, 10)`：画布 14×10 英寸，每个子图约 7×5 英寸。
- `axes` 是 2×2 的 NumPy 数组，通过 `axes[0, 0]`、`axes[0, 1]`、`axes[1, 0]`、`axes[1, 1]` 访问。

**布局规划**：
| 位置 | 内容 |
|------|------|
| `axes[0, 0]` | 原始 vs Mean Imputation 直方图 |
| `axes[0, 1]` | 原始 vs KNN Imputation 直方图 |
| `axes[1, 0]` | 原始 vs MICE Imputation 直方图 |
| `axes[1, 1]` | 四种方法的 Age 方差对比条形图 |

### 5.3 数据准备

```python
age_data = {}
for method_name in ['Mean Imputation', 'KNN Imputation', 'MICE Imputation']:
    X_train_imp, _, _, _ = imputed_datasets[method_name]
    age_data[method_name] = X_train_imp['Age']

age_original = X_train.dropna(subset=['Age'])['Age']
```

- `age_data`：字典，键是方法名，值是该方法插补后的 Age 列（含插补值）。
- `age_original`：原始训练集中 Age 非缺失的子集（作为对比基准）。
- 注意：Complete Case 不参与分布对比，因为它就是删除缺失行，分布与 `age_original` 相同。

### 5.4 直方图代码（以 Mean Imputation 为例）

```python
ax = axes[0, 0]
ax.hist(age_original, bins=60, alpha=0.5, density=True,
        color='#7f8c8d', label=f'Original (n={len(age_original):,})', edgecolor='white')
ax.hist(age_data['Mean Imputation'], bins=60, alpha=0.4, density=True,
        color='#3498db', label=f'Mean Imp. (n={len(age_data["Mean Imputation"]):,})', edgecolor='white')
```

`ax.hist()` 关键参数详解：

| 参数 | 值 | 作用 |
|------|-----|------|
| `bins=60` | 60 个箱子 | Age 范围约 0–100，每箱约 1.7 岁，分辨率足够 |
| `alpha=0.5` / `0.4` | 透明度 | 0.5=半透明，0.4=更透明，让两个直方图叠加时都能看见 |
| `density=True` | 归一化为概率密度 | **关键参数**！使直方图总面积=1，便于不同样本量的分布对比 |
| `color` | 灰色/蓝色 | 区分原始与插补 |
| `edgecolor='white'` | 白色边框 | 箱子间有视觉分隔 |
| `label` | 含样本量 | 图例显示 n=样本数，让读者知道对比的样本规模 |

> 💡 **为什么用 `density=True` 而不是 `density=False`？**
>
> - `density=False`（默认）：Y 轴是**计数**（每箱样本数）。原始 47,270 样本 vs 插补 56,000 样本，计数不可比——插补的柱子天然更高。
> - `density=True`：Y 轴是**概率密度**（每箱占比 / 箱宽）。无论样本量多少，总面积都归一化为 1，**可以直接对比分布形状**。
>
> 这是对比不同样本量分布时的**必备技巧**。

### 5.5 方差对比条形图（右下子图）

```python
ax = axes[1, 1]
variances = []
var_labels = []
for method_name in ['Original (non-missing)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']:
    if method_name == 'Original (non-missing)':
        variances.append(age_original.var())
    else:
        variances.append(age_data[method_name].var())
    var_labels.append(method_name)

colors_variance = ['#7f8c8d', '#3498db', '#e67e22', '#9b59b6']
bars = ax.barh(var_labels, variances, color=colors_variance, edgecolor='white')
for bar, val in zip(bars, variances):
    ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}', va='center', fontsize=10, fontweight='bold')
```

- `ax.barh()`：**水平**条形图（bar horizontal），方法名在 Y 轴，方差在 X 轴。适合标签较长的场景。
- `ax.text(val + 0.2, ...)`：数值标注在条形右侧（`val + 0.2` 偏移避免与条形重叠）。
- 方差数值：原始 274.41，Mean 273.99，KNN ≈ 274，MICE ≈ 274。

### 5.6 图表解读

![分布影响](../img/06e_distribution_impact.png)

**从图中能读出什么？——这是全模块最关键的解读**

#### 左上：Original vs Mean Imputation

> 🔴 **均值插补的"假峰现象"**：
>
> 在 Age ≈ 64 岁（均值）位置会出现一个**异常高的尖峰**。这是因为所有缺失的 Age 值都被替换为同一个常数（均值），导致该点的密度激增。这个尖峰是**人工制造的伪信号**，原始数据中并不存在。
>
> 同时，分布的两侧（年轻和年老端）相对压低，因为样本被"吸"到均值附近。这就是均值插补**人为压缩方差**的直观体现。

#### 右上：Original vs KNN Imputation

KNN 插补的分布与原始分布**高度重合**，没有异常尖峰。因为 KNN 根据相似患者的实际 Age 值插补，每个缺失值得到不同的填补值，保留了分布的多样性。

#### 左下：Original vs MICE Imputation

MICE 插补的分布同样与原始分布**高度重合**，甚至比 KNN 更平滑。因为 MICE 利用其他特征（如 year、Gender、Diagnostic.means）回归预测 Age，并引入随机性，插补值的分布接近真实分布。

#### 右下：方差对比

| 方法 | Age 方差 | 与原始差距 |
|------|---------|-----------|
| Original | 274.41 | 基准 |
| Mean | 273.99 | -0.42 |
| KNN | ≈274 | ≈0 |
| MICE | ≈274 | ≈0 |

> 📌 **方差对比的解读**：
>
> 表面上看 Mean 的方差（273.99）与原始（274.41）差距很小，似乎"方差压缩"不严重。但这是因为 Age 缺失率仅 0.15%——只有约 84 个样本被插补为均值，对整体方差影响微乎其微。
>
> **如果缺失率提高到 30%–50%**（如 Raca.Color 的 15.31%），均值插补对方差的压缩会显著放大。本案例 Age 缺失率低，掩盖了均值插补的缺陷——这正是为什么分布图（直方图）比单一方差数值更能揭示问题。

> 💡 **小贴士**：评估插补方法时，**永远要看分布图**，不能只看汇总统计量（均值、方差）。分布图能揭示均值插补的假峰、KNN 的局部聚集等汇总统计量无法发现的模式。

---

## 图 6f：PR 曲线（Precision-Recall Curve）

### 6.1 PR 曲线 vs ROC 曲线

| 特性 | ROC 曲线 | PR 曲线 |
|------|---------|---------|
| X 轴 | FPR = FP/(FP+TN) | Recall = TP/(TP+FN) |
| Y 轴 | TPR = Recall | Precision = TP/(TP+FP) |
| 基线 | 对角线（AUC=0.5） | 水平线（AP=正例比例） |
| 对不平衡数据敏感度 | 较低 | **较高** |
| 适用场景 | 类别平衡 | **类别不平衡** |

> 💡 **为什么 PR 曲线对不平衡数据更敏感？**
>
> ROC 的 FPR 分母是 TN（真负例），在不平衡数据中 TN 数量巨大，FPR 变化被稀释。PR 曲线的 Precision 分母是 FP（假正例），直接反映"预测为正的样本中有多少是真的正例"，对少数类预测质量更敏感。
>
> 本案例 VIVO（存活）约占 75%，MORTO（死亡）约占 25%，属于轻度不平衡，PR 曲线能提供 ROC 看不到的额外信息。

### 6.2 代码逐行解析

```python
fig, ax = plt.subplots(figsize=(9, 7))

for r in results:
    method = r['Method']
    color = r['Color']
    X_t, _, y_t, _ = imputed_datasets[method]
    lr = r['Model']
    y_prob_pr = lr.predict_proba(X_test_imp if method != 'Complete Case'
                                  else imputed_datasets['Complete Case'][1])[:, 1]
    y_true_pr = (y_test_imp if method != 'Complete Case'
                 else imputed_datasets['Complete Case'][3])
```

测试集选择逻辑与图 6b 完全相同——Complete Case 用删减后的测试集，其他方法用完整测试集。

```python
    precision, recall, _ = precision_recall_curve(y_true_pr, y_prob_pr)
    ap = r['Avg_Precision']
    ax.plot(recall, precision, color=color, linewidth=2,
            label=f'{method} (AP = {ap:.4f})')
```

`precision_recall_curve()` 返回三个值：
- `precision`：精确率数组
- `recall`：召回率数组
- `_`：阈值数组（忽略）

**注意与 ROC 的区别**：
- ROC 是 `ax.plot(fpr, tpr)`——X 是 FPR，Y 是 TPR。
- PR 是 `ax.plot(recall, precision)`——X 是 Recall，Y 是 Precision。
- 顺序反过来，因为 PR 曲线习惯上 X 轴是 Recall。

`ap = r['Avg_Precision']`：Average Precision，即 PR 曲线下面积，是 PR 曲线的" AUC"。`average_precision_score()` 在模块 3 已计算好。

```python
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves (Imbalanced Focus)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower left')
```

- 标题特别标注 `(Imbalanced Focus)`，提示这是为不平衡数据设计的评估。
- 图例放在左下角（`loc='lower left'`），因为 PR 曲线通常从右上向左下延伸，左下区域空间最多。

### 6.3 图表解读

![PR曲线](../img/06f_pr_curves.png)

**从图中能读出什么？**

1. **四条曲线有可见差异**：与 ROC 曲线（图 6b）的高度重叠不同，PR 曲线能区分出方法差异。
2. **AP 数值差异更明显**：
   - Complete Case: AP = 0.6879（最低）
   - Mean: AP = 0.7256
   - KNN: AP = 0.7247
   - MICE: AP = 0.7263（最高）
3. **Complete Case 明显低于其他三种**：差距约 0.04，比 AUC 的差距（0.0016）大 25 倍。
4. **三种插补方法接近**：MICE 略优，Mean 和 KNN 几乎相同。

> 📌 **关键洞察**：PR 曲线揭示了 ROC 曲线掩盖的差异——**Complete Case 在少数类（MORTO）预测上明显劣于插补方法**。这是因为删除 15% 样本主要损失了少数类信息，而插补保留了完整样本。

> 💡 **小贴士**：在医学预测模型中，死亡（少数类）的预测往往比存活（多数类）更重要——漏诊死亡的代价远高于误报。因此 PR 曲线的差异在临床上具有实际意义。

---

## 核心讨论：结果综合解读

### 9.1 四种方法性能差异极小

从图 6a 和图 6b 看，四种方法在 AUC 和 Recall 上的差距 < 0.002，**统计上不显著**。这说明在**区分能力**层面，插补方法的选择对本案例影响有限。

### 9.2 Complete Case 的优势与代价

**优势**（AUC/Recall 略优）：
- 0.8644 vs 0.8632（Mean）vs 0.8628（KNN）vs 0.8637（MICE）
- 原因：保留了"真实观测"，没有插补引入的噪声。
- 但优势微乎其微，可能源于随机性。

**代价**（Brier/PR-AUC 劣势）：
- Brier 0.1493（最差）vs 0.1462–0.1467
- PR-AUC 0.6879（最差）vs 0.7247–0.7263
- 原因：删除 15.6% 样本导致训练分布偏移，模型校准和少数类预测受影响。

### 9.3 插补方法的优势

**Brier/PR-AUC 略优**：
- 保留全部 56,000 样本，训练分布更接近真实分布。
- 校准更好（Brier 更低），少数类预测更准（PR-AUC 更高）。

### 9.4 KNN 的性价比问题

| 方法 | AUC | 耗时 |
|------|-----|------|
| Mean | 0.8632 | 0.0s |
| KNN | 0.8628 | **19.5s** |
| MICE | 0.8637 | 0.1s |

KNN 耗时 19.5 秒，是 MICE 的 195 倍，但性能**没有任何优势**（AUC 甚至略低）。原因：
- KNN 复杂度 O(n²)，在 56,000 样本上计算量大。
- KNN 插补的局部相似性优势，在低缺失率场景下无法体现。
- **结论**：本案例下 KNN 性价比最差，不推荐。

### 9.5 关键洞察：缺失率低时方法差异有限

> 🎯 **核心结论**：
>
> 本案例最高缺失率 15.31%（Raca.Color），Age 仅 0.15%。在**低缺失率**场景下：
>
> 1. **四种方法性能差异极小**（AUC 差距 < 0.002）——因为缺失值占比低，无论怎么处理，对整体数据分布的影响都有限。
> 2. **均值插补的方差压缩不明显**——Age 方差仅从 274.41 降到 273.99，因为只插补了 84 个样本。
> 3. **KNN 的计算成本无法被性能收益证明**——19.5 秒 vs 0.1 秒（MICE），性能却更差。
>
> **但当缺失率提高到 30%–50% 时**，差异会显著放大：
> - 均值插补会严重压缩方差，破坏分布。
> - Complete Case 会损失大量样本，统计效力下降。
> - KNN 和 MICE 的优势会显现。
>
> **实践建议**：缺失率 < 10% 时，简单方法（Mean/Complete Case）即可；缺失率 10%–30% 时，优先 MICE；缺失率 > 30% 时，谨慎评估是否该特征可用，或考虑专门的多重插补流程。

---

## 小贴士

### 📌 可视化设计原则

1. **颜色一致性**：同一方法在所有图中使用相同颜色，降低读者认知负担。
2. **多指标对比**：用 1×3 子图同时展示 AUC/Recall/Brier，避免单指标的盲区。
3. **数值标注**：柱状图顶部标注精确数值，兼顾"直观对比"和"精确读数"。
4. **参考线**：ROC 和校准曲线都画对角线，提供"随机/完美"的基准。
5. **隐藏多余边框**：去掉上、右边框，减少视觉噪音（`spines['top'].set_visible(False)`）。

### 📌 评估指标选择建议

| 场景 | 推荐指标 | 原因 |
|------|---------|------|
| 类别平衡 | AUC + ROC 曲线 | 整体区分能力 |
| 类别不平衡 | PR-AUC + PR 曲线 | 少数类敏感 |
| 医学/风险预测 | Brier + 校准曲线 | 概率准确性至关重要 |
| 综合评估 | 以上全部 | 单一指标会掩盖问题 |

---

## 常见问题

### Q1: 为什么 ROC 曲线四条线几乎重合，但 PR 曲线有差异？

**A**: ROC 的 FPR 分母是 TN，在不平衡数据中 TN 数量大，FPR 变化被稀释。PR 曲线的 Precision 直接反映少数类预测质量，对 Complete Case 的样本损失更敏感。**不平衡数据下，PR 曲线比 ROC 更有诊断价值**。

### Q2: 均值插补的"假峰"为什么在方差对比中看不出来？

**A**: 因为 Age 缺失率仅 0.15%，只有约 84 个样本被插补为均值。这 84 个点在均值处形成尖峰，但对整体方差的贡献很小（方差从 274.41 降到 273.99，仅降 0.15%）。**分布图（直方图）比汇总统计量（方差）更能揭示局部异常**。如果缺失率提高到 30%，方差压缩会显著放大。

### Q3: KNN 插补耗时 19.5 秒，为什么这么慢？

**A**: KNN 插补对每个缺失值需要计算与所有其他样本的距离，复杂度 O(n²)。在 56,000 样本 × 5 特征上，需要计算约 15 亿次距离。MICE 用迭代回归，复杂度 O(n × iter × features)，快得多。**大数据集上慎用 KNN 插补**，可考虑先降维或用近似最近邻算法（如 FAISS）。

### Q4: Complete Case 在 AUC 上略优，为什么还说插补方法更好？

**A**: AUC 优势仅 0.0016，统计上不显著（可能源于随机性）。但 Complete Case 在 Brier 和 PR-AUC 上劣势明显（差距 0.03–0.04），且损失 15.6% 样本。综合来看，插补方法在**校准和少数类预测**上更可靠，且保留了样本量。**多指标综合评估**比单一 AUC 更有说服力。

### Q5: 校准曲线的"对角线"和 ROC 的"对角线"含义相同吗？

**A**: **完全不同**！
- ROC 对角线：随机分类器（AUC=0.5），任何有效模型应在上方。
- 校准对角线：完美校准（预测概率=实际频率），模型曲线应贴近它。
两者方向、含义、解读都不同，初学者常混淆。

### Q6: 为什么 MICE 比 KNN 快这么多，但性能还略好？

**A**: MICE（IterativeImputer）用迭代回归，每次迭代是线性回归，复杂度低。KNN 需要计算样本间距离矩阵，复杂度高。性能上，MICE 利用特征间关系建模，比 KNN 的"局部相似"更能捕捉全局结构。**MICE 是医学研究的首选插补方法**——快、准、理论严谨。

---

## 本模块小结

本模块通过 6 张可视化图表，全面解读了四种插补方法对模型性能和特征分布的影响：

| 图表 | 核心发现 |
|------|---------|
| 图 6a 性能柱状图 | AUC/Recall 差异极小，Brier 能区分 Complete Case 较差 |
| 图 6b ROC 曲线 | 四条曲线高度重叠，AUC 差距 < 0.002 |
| 图 6c 校准曲线 | 插补方法校准优于 Complete Case，MICE 最佳 |
| 图 6d 训练集规模 | Complete Case 损失 15.6% 样本（47,270 vs 56,000） |
| 图 6e 分布影响 | **均值插补出现假峰**，KNN/MICE 保留原始分布 |
| 图 6f PR 曲线 | Complete Case 在少数类预测上明显劣于插补方法 |

**核心技能掌握**：
- ✅ matplotlib 多子图布局（1×3、2×2）
- ✅ `zip(axes, data)` 并行遍历技巧
- ✅ 柱状图数值标注（`ax.text()`）
- ✅ 直方图 `density=True` 归一化对比
- ✅ ROC/PR/校准曲线的绘制与解读
- ✅ 多指标综合评估的思维

**核心结论**：
1. 低缺失率下（< 15%），四种方法性能差异有限。
2. Complete Case 损失样本量，校准和少数类预测较差。
3. 均值插补破坏分布（假峰现象），但在低缺失率下影响可控。
4. KNN 性价比最差（耗时高、无性能优势）。
5. **MICE 是综合最优**——快、校准好、保留分布、理论严谨。

---

## 整个案例教程 3 的总结

### 案例回顾

本案例教程以**巴西癌症登记数据**为背景，系统比较了四种缺失值处理策略：

```
模块 1: 数据加载与缺失值分析
  → 识别缺失模式（MCAR/MAR/MNAR），计算缺失率
  → 发现 Raca.Color 缺失 15.31%，Age 缺失 0.15%

模块 2: 四种插补方法实现
  → Complete Case: 删除缺失行
  → Mean Imputation: 均值/众数填补
  → KNN Imputation: K近邻填补
  → MICE: 多重插补（IterativeImputer）

模块 3: 模型训练与评估
  → Logistic Regression + class_weight='balanced'
  → 评估指标: AUC, Recall, Brier, PR-AUC, 校准曲线

模块 4: 可视化与结果解读（本模块）
  → 6 张图表全面对比
  → 综合解读与最佳实践建议
```

### 核心知识点

1. **缺失值处理不是"数据清洗"的附属步骤，而是影响模型性能的关键决策**。
2. **不同插补方法有不同的适用场景**：
   - 低缺失率（< 10%）：Mean 或 Complete Case 即可
   - 中缺失率（10%–30%）：优先 MICE
   - 高缺失率（> 30%）：评估特征可用性，或用专业多重插补流程
3. **多指标评估优于单一指标**：AUC 看区分能力，Brier 看校准，PR-AUC 看少数类，分布图看数据保真度。
4. **可视化是数据科学沟通的核心**：表格数字密集，图表一目了然。

### 方法选择决策树

```
缺失率 < 5%?
  ├─ 是 → Complete Case 或 Mean Imputation（简单高效）
  └─ 否 → 缺失率 5%–30%?
            ├─ 是 → MICE（综合最优）
            └─ 否 → 缺失率 > 30%?
                      ├─ 关键特征 → MICE + 敏感性分析
                      └─ 非关键特征 → 考虑删除特征
```

### 进阶学习方向

- **多重插补的统计理论**：Rubin's rules, MI inference
- **深度学习插补**：DataWig, GAIN（Generative Adversarial Imputation Networks）
- **时间序列插补**：前向填充、后向填充、季节性插补
- **因果推断中的缺失值处理**：IPW（Inverse Probability Weighting）

---

> 📚 **下一步学习**：完成本案例教程后，建议继续学习案例教程 4（特征缩放与正则化），深入理解标准化、归一化、L1/L2 正则化对模型的影响。
>
> 本案例的完整代码位于 `src/03_preprocessing_imputation.py`，结果图片位于 `img/` 目录，结果数据位于 `results/` 目录。建议动手运行代码，修改参数（如缺失率、KNN 的 K 值、MICE 的迭代次数），观察图表变化，加深理解。
