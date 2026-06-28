# 模块 3：综合可视化与结果保存 — 6 张子图解读稳定性

> 本模块是案例教程 17「SHAP 稳定性 Bootstrap 分析」的**收尾模块**。 
>
> 本模块最核心的知识点有三个：**一是 6 张子图的"分工"**——每张子图回答一个不同的稳定性问题；**二是 matplotlib 的 2×3 子图布局**——`plt.subplots(2, 3)` 创建 6 个 axes 对象； 

***

## 学习目标

学完本模块后，你将能够：

1. **理解 6 张子图的"分工"**：知道每张子图回答什么稳定性问题，如何读取。
2. **掌握 matplotlib 的 2×3 子图布局**：明白 `plt.subplots(2, 3, figsize=(20, 12))` 如何创建 6 个 axes。
3. **理解条形图 + 误差棒的绘制**：知道 `ax.bar(..., yerr=...)` 如何同时显示均值和置信区间。
4. **掌握水平条形图的绘制**：知道 `ax.barh()` 与 `ax.bar()` 的区别，以及颜色映射 `plt.cm.RdYlGn_r` 的用法。
5. **理解箱线图 + 散点抖动图的叠加**：知道如何用 `ax.boxplot()` + `ax.scatter()` 同时显示分布和原始数据。
6. **掌握时间序列折线图的绘制**：知道如何用 `ax.plot()` 绘制 30 次迭代的重要性轨迹。
7. **理解直方图 + 垂直线的绘制**：知道如何用 `ax.hist()` + `ax.axvline()` 显示分布和均值/中位数。
 
***

## 一、可视化概述

```python
# ============================================================================
# 3. 综合可视化 (2×3)
# ============================================================================
print(f"\n[3] 生成稳定性分析可视化...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(f'SHAP Stability Bootstrap Analysis ({N_BOOTSTRAP} iterations)\n'
             f'Mean Rank Correlation = {np.mean(rank_correlations):.4f}',
             fontsize=14, fontweight='bold')
```

### 1.1 `fig, axes = plt.subplots(2, 3, figsize=(20, 12))`

创建一个 2 行 3 列的子图网格：

- **`2, 3`**：2 行 3 列，共 6 个子图。
- **`figsize=(20, 12)`**：图形大小，宽 20 英寸，高 12 英寸。这个尺寸适合 6 个子图，每个子图约 6.7×6 英寸。
- **返回值**：
  - `fig`：Figure 对象，整个图形。
  - `axes`：形状 `(2, 3)` 的 NumPy 数组，每个元素是一个 Axes 对象（子图）。

访问子图的方式：

- `axes[0, 0]`：左上角子图（子图 1）。
- `axes[0, 1]`：中上子图（子图 2）。
- `axes[0, 2]`：右上角子图（子图 3）。
- `axes[1, 0]`：左下角子图（子图 4）。
- `axes[1, 1]`：中下子图（子图 5）。
- `axes[1, 2]`：右下角子图（子图 6）。

### 1.2 `fig.suptitle(...)`

设置**总标题**（super title），位于整个图形的顶部。

```python
fig.suptitle(f'SHAP Stability Bootstrap Analysis ({N_BOOTSTRAP} iterations)\n'
             f'Mean Rank Correlation = {np.mean(rank_correlations):.4f}',
             fontsize=14, fontweight='bold')
```

- **`f'SHAP Stability Bootstrap Analysis ({N_BOOTSTRAP} iterations)\n'`**：第一行，显示"30 iterations"。
- **`f'Mean Rank Correlation = {np.mean(rank_correlations):.4f}'`**：第二行，显示平均 Spearman ρ = 0.9067。
- **`fontsize=14`**：字体大小 14。
- **`fontweight='bold'`**：粗体。
- **`\n`**：换行符，让标题分两行显示。

**实际显示**：

```
SHAP Stability Bootstrap Analysis (30 iterations)
Mean Rank Correlation = 0.9067
```

### 1.3 6 张子图的"分工"

> 💡 **重点概念：6 张子图的读取指南**
>
> ```
> ┌─────────────────────────┬─────────────────────────┬─────────────────────────┐
> │ 子图 1: 重要性+CI       │ 子图 2: CV 稳定性排名    │ 子图 3: Bootstrap 分布   │
> │                         │                         │                         │
> │  条形图 + 误差棒:       │  水平条形图:             │  箱线图 + 散点:          │
> │  越高=越重要            │  越靠左=越稳定           │  越窄=越稳定             │
> │  误差棒越短=越稳定      │  绿色=稳定 红色=不稳定   │  无离群点=好             │
> ├─────────────────────────┼─────────────────────────┼─────────────────────────┤
> │ 子图 4: 变化轨迹         │ 子图 5: 排名稳定性       │ 子图 6: CI 宽度排名       │
> │                         │                         │                         │
> │  折线图:                 │  直方图:                 │  条形图:                 │
> │  每条线=一个特征         │  越集中越靠右=            │  越高=该特征的不确定性大  │
> │  线波动越小=越稳定       │  排名越稳定              │  即"不接受的结论不可靠"   │
> └─────────────────────────┴─────────────────────────┴─────────────────────────┘
> ```

***

## 二、子图 1：特征重要性与 95% CI

```python
# === 子图 1: 特征重要性与 95% CI ===
ax1 = axes[0, 0]
top_n = len(feature_names)
x_pos = np.arange(top_n)
top_idx = sorted_idx
ax1.bar(x_pos, mean_imp[top_idx],
        yerr=[mean_imp[top_idx] - ci_lower[top_idx],
              ci_upper[top_idx] - mean_imp[top_idx]],
        capsize=5, color='steelblue', alpha=0.7,
        error_kw={'linewidth': 2, 'ecolor': 'darkred'})
ax1.set_xticks(x_pos)
ax1.set_xticklabels([feature_names[i][:15] for i in top_idx],
                     rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Mean |SHAP Value|', fontsize=11)
ax1.set_title('Feature Importance with 95% CI', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2, axis='y')
```

### 2.1 准备数据

```python
ax1 = axes[0, 0]
top_n = len(feature_names)  # 6
x_pos = np.arange(top_n)    # [0, 1, 2, 3, 4, 5]
top_idx = sorted_idx         # 按重要性降序的索引
```

- **`ax1 = axes[0, 0]`**：选择左上角子图。
- **`top_n = len(feature_names)`**：特征数 = 6。
- **`x_pos = np.arange(top_n)`**：x 轴位置 `[0, 1, 2, 3, 4, 5]`。
- **`top_idx = sorted_idx`**：按 Mean SHAP 降序排列的索引（最重要的在前）。

### 2.2 `ax1.bar(...)` 绘制条形图 + 误差棒

```python
ax1.bar(x_pos, mean_imp[top_idx],
        yerr=[mean_imp[top_idx] - ci_lower[top_idx],
              ci_upper[top_idx] - mean_imp[top_idx]],
        capsize=5, color='steelblue', alpha=0.7,
        error_kw={'linewidth': 2, 'ecolor': 'darkred'})
```

逐参数解释：

- **`x_pos`**：条形的 x 位置 `[0, 1, 2, 3, 4, 5]`。
- **`mean_imp[top_idx]`**：条形高度，按重要性降序排列的 Mean SHAP 值。
- **`yerr=[下误差, 上误差]`**：误差棒！这是一个 2 行 N 列的数组：
  - 第 1 行 `mean_imp[top_idx] - ci_lower[top_idx]`：**下误差** = 均值 - CI 下界。
  - 第 2 行 `ci_upper[top_idx] - mean_imp[top_idx]`：**上误差** = CI 上界 - 均值。
  - 这样误差棒的范围正好是 \[CI\_Lower, CI\_Upper]。
- **`capsize=5`**：误差棒顶端的"帽子"宽度（像素）。
- **`color='steelblue'`**：条形颜色，钢蓝色。
- **`alpha=0.7`**：透明度 0.7（70% 不透明），让条形不完全遮挡网格线。
- **`error_kw={'linewidth': 2, 'ecolor': 'darkred'}`**：误差棒样式：
  - `linewidth=2`：误差棒线宽 2。
  - `ecolor='darkred'`：误差棒颜色，暗红色（与蓝色条形形成对比）。

### 2.3 设置 x 轴标签

```python
ax1.set_xticks(x_pos)
ax1.set_xticklabels([feature_names[i][:15] for i in top_idx],
                     rotation=30, ha='right', fontsize=9)
```

- **`set_xticks(x_pos)`**：设置 x 轴刻度位置。
- **`set_xticklabels(...)`**：设置 x 轴刻度标签。
  - `[feature_names[i][:15] for i in top_idx]`：按重要性降序的特征名，每个截取前 15 个字符（防止过长）。
  - `rotation=30`：标签旋转 30 度（防止重叠）。
  - `ha='right'`：水平对齐方式为"右对齐"（旋转后标签的右端对齐刻度）。
  - `fontsize=9`：字体大小 9。

### 2.4 设置标签和标题

```python
ax1.set_ylabel('Mean |SHAP Value|', fontsize=11)
ax1.set_title('Feature Importance with 95% CI', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.2, axis='y')
```

- **`set_ylabel('Mean |SHAP Value|')`**：y 轴标签"Mean |SHAP Value|"。
- **`set_title('Feature Importance with 95% CI')`**：子图标题。
- **`grid(True, alpha=0.2, axis='y')`**：显示 y 轴网格线，透明度 0.2（淡淡的）。

### 2.5 子图 1 的解读

![子图1：特征重要性与95%CI](../img/21a_shap_stability_bootstrap.png)

**如何读取子图 1**：

- **条形高度** = Mean SHAP，越高越重要。
- **误差棒长度** = CI 宽度，越短越稳定。
- **观察**：
  - `year` 最高（0.225），误差棒较短 → 重要性高且稳定。
  - `Code.Profession` 最低（0.024），误差棒相对高度较大 → 重要性低且不稳定。

***

## 三、子图 2：CV 稳定性排名

```python
# === 子图 2: 变异系数 CV 排名 ===
ax2 = axes[0, 1]
cv_sorted = np.argsort(cv_imp)
colors = plt.cm.RdYlGn_r(1 - cv_imp[cv_sorted] / (cv_imp.max() + 1e-8))
bars = ax2.barh(range(top_n), cv_imp[cv_sorted], color=colors, alpha=0.8, edgecolor='white')
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([feature_names[i][:15] for i in cv_sorted], fontsize=9)
ax2.set_xlabel('Coefficient of Variation (CV)', fontsize=11)
ax2.set_title('Feature Stability Ranking (CV, lower=more stable)',
              fontsize=12, fontweight='bold')
for bar, val in zip(bars, cv_imp[cv_sorted]):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', ha='left', va='center', fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
```

### 3.1 准备数据

```python
ax2 = axes[0, 1]
cv_sorted = np.argsort(cv_imp)  # 按 CV 升序（最稳定的在前）
```

- **`ax2 = axes[0, 1]`**：选择中上子图。
- **`cv_sorted = np.argsort(cv_imp)`**：按 CV 升序排列的索引。注意这里**没有** `[::-1]`，所以是升序——CV 最小（最稳定）的特征在前。

### 3.2 颜色映射

```python
colors = plt.cm.RdYlGn_r(1 - cv_imp[cv_sorted] / (cv_imp.max() + 1e-8))
```

这行代码为每个条形生成颜色，**绿色表示稳定，红色表示不稳定**：

- **`plt.cm.RdYlGn_r`**：Red-Yellow-Green 反转色图。`_r` 表示反转，所以是"红-黄-绿"。
- **`cv_imp[cv_sorted] / (cv_imp.max() + 1e-8)`**：归一化的 CV 值，范围 \[0, 1]。CV 越大，值越接近 1。
- **`1 - ...`**：反转，使 CV 小的（稳定的）对应高值。
- **结果**：CV 小 → 绿色，CV 大 → 红色。

> 💡 **小贴士：matplotlib 色图的用法**
>
> `plt.cm.色图名(值)` 返回 RGBA 颜色，值范围 \[0, 1]。
>
> 常用色图：
>
> - `plt.cm.Reds`：白色到红色。
> - `plt.cm.Blues`：白色到蓝色。
> - `plt.cm.RdYlGn`：红-黄-绿（交通灯风格）。
> - `plt.cm.RdYlGn_r`：绿-黄-红（反转）。
> - `plt.cm.viridis`：紫-蓝-绿-黄（默认色图）。
>
> 加 `_r` 后缀表示反转色图。

### 3.3 `ax2.barh(...)` 绘制水平条形图

```python
bars = ax2.barh(range(top_n), cv_imp[cv_sorted], color=colors, alpha=0.8, edgecolor='white')
```

- **`barh`**：horizontal bar，水平条形图（条形横向延伸）。
- **`range(top_n)`**：y 轴位置 `[0, 1, 2, 3, 4, 5]`。
- **`cv_imp[cv_sorted]`**：条形长度，按 CV 升序排列。
- **`color=colors`**：每个条形的颜色（绿到红）。
- **`alpha=0.8`**：透明度。
- **`edgecolor='white'`**：条形边缘白色，让条形更清晰。

### 3.4 设置 y 轴标签

```python
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([feature_names[i][:15] for i in cv_sorted], fontsize=9)
```

- **`set_yticks(range(top_n))`**：y 轴刻度位置。
- **`set_yticklabels(...)`**：y 轴标签，按 CV 升序的特征名。

### 3.5 在条形上添加数值标签

```python
for bar, val in zip(bars, cv_imp[cv_sorted]):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', ha='left', va='center', fontsize=8)
```

遍历每个条形，在条形右端添加 CV 数值：

- **`zip(bars, cv_imp[cv_sorted])`**：把条形对象和 CV 值配对。
- **`ax2.text(x, y, text, ...)`**：在 (x, y) 位置添加文本。
  - `x = val + 0.005`：条形右端再往右 0.005（留一点空隙）。
  - `y = bar.get_y() + bar.get_height()/2`：条形垂直中心。
  - `text = f'{val:.3f}'`：CV 值，保留 3 位小数。
  - `ha='left'`：水平左对齐（文本左端对齐 x）。
  - `va='center'`：垂直居中。
  - `fontsize=8`：字体大小 8。

### 3.6 美化：移除上边和右边框

```python
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
```

- **`spines['top']`**：上边框。
- **`spines['right']`**：右边框。
- **`set_visible(False)`**：隐藏。

这是数据可视化的常见美化技巧——移除不必要的边框，让图表更简洁。

### 3.7 子图 2 的解读

**如何读取子图 2**：

- **条形长度** = CV，越短越稳定。
- **颜色**：绿色 = 稳定，红色 = 不稳定。
- **观察**：
  - `year`（CV=0.064）和 `Extension`（CV=0.083）是绿色短条 → 高度稳定。
  - `Code.Profession`（CV=0.395）是红色长条 → 不稳定。

***

## 四、子图 3：Bootstrap 分布箱线图

```python
# === 子图 3: Bootstrap 分布箱线图 (Top 3) ===
ax3 = axes[0, 2]
top_3_idx = sorted_idx[:3]
box_data = [bootstrap_shap_importance[:, idx] for idx in top_3_idx]
bp = ax3.boxplot(box_data, labels=[feature_names[idx][:12] for idx in top_3_idx],
                 patch_artist=True, widths=0.5)
colors_box = plt.cm.Set2(np.linspace(0, 0.8, 3))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)

# 叠加散点抖动图
for i, idx in enumerate(top_3_idx):
    y_jitter = bootstrap_shap_importance[:, idx]
    x_jitter = np.random.normal(i + 1, 0.05, len(y_jitter))
    ax3.scatter(x_jitter, y_jitter, alpha=0.3, s=15, color='gray')

ax3.set_ylabel('SHAP Importance Distribution', fontsize=11)
ax3.set_title('Top 3 Features: Bootstrap Distribution\n(scatter + boxplot)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.2, axis='y')
```

### 4.1 准备数据

```python
ax3 = axes[0, 2]
top_3_idx = sorted_idx[:3]  # 前 3 个最重要特征的索引
box_data = [bootstrap_shap_importance[:, idx] for idx in top_3_idx]
```

- **`ax3 = axes[0, 2]`**：选择右上角子图。
- **`top_3_idx = sorted_idx[:3]`**：最重要的 3 个特征的索引（year、Extension、Raca.Color）。
- **`box_data`**：一个列表，包含 3 个数组，每个数组是该特征在 30 次迭代中的重要性值。

### 4.2 `ax3.boxplot(...)` 绘制箱线图

```python
bp = ax3.boxplot(box_data, labels=[feature_names[idx][:12] for idx in top_3_idx],
                 patch_artist=True, widths=0.5)
```

- **`box_data`**：输入数据，列表的每个元素是一个箱体。
- **`labels`**：每个箱体的标签（特征名，截取前 12 字符）。
- **`patch_artist=True`**：允许填充箱体颜色（默认只有边框）。
- **`widths=0.5`**：箱体宽度 0.5。
- **返回值** **`bp`**：字典，包含箱体各部分的引用（`boxes`、`medians`、`whiskers`、`caps`、`fliers`）。

> 💡 **重点概念：箱线图的 5 个要素**
>
> 箱线图（Box Plot）显示数据的 5 个统计量：
>
> 1. **箱体下边** = Q1（第 25 百分位数）。
> 2. **箱体上边** = Q3（第 75 百分位数）。
> 3. **箱体高度** = IQR = Q3 - Q1（四分位距）。
> 4. **中间线** = 中位数（Q2，第 50 百分位数）。
> 5. **须（whiskers）**：从箱体延伸出去的线，通常到 Q1-1.5×IQR 和 Q3+1.5×IQR。
>
> **离群点（fliers）**：超出须的范围的点，用圆点显示。

### 4.3 设置箱体颜色

```python
colors_box = plt.cm.Set2(np.linspace(0, 0.8, 3))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
```

- **`plt.cm.Set2(np.linspace(0, 0.8, 3))`**：从 Set2 色图生成 3 个颜色。`np.linspace(0, 0.8, 3)` 生成 `[0, 0.4, 0.8]`，对应 3 种颜色。
- **`for patch, color in zip(bp['boxes'], colors_box)`**：遍历每个箱体和颜色。
- **`patch.set_facecolor(color)`**：设置箱体填充颜色。
- **`patch.set_alpha(0.7)`**：透明度 0.7。

### 4.4 设置中位数线样式

```python
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(2)
```

把中位数线设为黑色、线宽 2，让中位数更醒目。

### 4.5 叠加散点抖动图

```python
# 叠加散点抖动图
for i, idx in enumerate(top_3_idx):
    y_jitter = bootstrap_shap_importance[:, idx]
    x_jitter = np.random.normal(i + 1, 0.05, len(y_jitter))
    ax3.scatter(x_jitter, y_jitter, alpha=0.3, s=15, color='gray')
```

在箱线图上叠加**散点抖动图**（jitter plot），显示每个数据点的位置：

- **`enumerate(top_3_idx)`**：遍历 3 个特征，`i` 是索引（0, 1, 2），`idx` 是特征索引。
- **`y_jitter = bootstrap_shap_importance[:, idx]`**：该特征的 30 个重要性值（y 坐标）。
- **`x_jitter = np.random.normal(i + 1, 0.05, len(y_jitter))`**：x 坐标，以 `i+1` 为中心、标准差 0.05 的正态分布。这样散点在箱体中心附近"抖动"，避免重叠。
- **`ax3.scatter(...)`**：绘制散点。
  - `alpha=0.3`：透明度 0.3（淡淡的）。
  - `s=15`：点大小 15。
  - `color='gray'`：灰色。

> 💡 **小贴士：为什么用抖动图？**
>
> 箱线图虽然显示分布的统计量，但隐藏了原始数据。叠加抖动图可以：
>
> 1. **显示每个数据点**：看到 30 个点的实际分布。
> 2. **发现离群点**：箱线图标记的离群点在抖动图中也能看到。
> 3. **判断分布形状**：点的密度反映分布形状（均匀、聚集、双峰等）。
>
> "箱线图 + 抖动图"是数据可视化的最佳实践之一。

### 4.6 子图 3 的解读

**如何读取子图 3**：

- **箱体高度** = IQR，越窄越稳定。
- **中位数线** = 30 次迭代的中位数。
- **散点** = 30 个原始数据点。
- **观察**：
  - `year` 箱体窄、中位数高 → 重要性高且稳定。
  - `Extension` 箱体稍宽，但无离群点 → 稳定。
  - `Raca.Color` 箱体较宽 → 有一定波动。

***

## 五、子图 4：Bootstrap 变化轨迹

```python
# === 子图 4: Bootstrap 过程时间序列 ===
ax4 = axes[1, 0]
top_5_idx = sorted_idx[:5]
colors_ts = plt.cm.tab10(np.linspace(0, 0.8, 5))
for k, idx in enumerate(top_5_idx):
    ax4.plot(bootstrap_shap_importance[:, idx],
             label=feature_names[idx][:15],
             color=colors_ts[k], alpha=0.8, linewidth=1.8)
ax4.set_xlabel('Bootstrap Iteration', fontsize=11)
ax4.set_ylabel('SHAP Importance', fontsize=11)
ax4.set_title('Importance Trajectory Across Bootstraps\n(should be stable, not erratic)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.2)
```

### 5.1 准备数据

```python
ax4 = axes[1, 0]
top_5_idx = sorted_idx[:5]  # 前 5 个最重要特征的索引
colors_ts = plt.cm.tab10(np.linspace(0, 0.8, 5))
```

- **`ax4 = axes[1, 0]`**：选择左下角子图。
- **`top_5_idx = sorted_idx[:5]`**：最重要的 5 个特征的索引。
- **`colors_ts = plt.cm.tab10(np.linspace(0, 0.8, 5))`**：从 tab10 色图生成 5 个颜色。

### 5.2 绘制折线

```python
for k, idx in enumerate(top_5_idx):
    ax4.plot(bootstrap_shap_importance[:, idx],
             label=feature_names[idx][:15],
             color=colors_ts[k], alpha=0.8, linewidth=1.8)
```

- **`enumerate(top_5_idx)`**：遍历 5 个特征。
- **`bootstrap_shap_importance[:, idx]`**：该特征在 30 次迭代中的重要性序列，形状 `(30,)`。
- **`ax4.plot(...)`**：绘制折线图。
  - `label=feature_names[idx][:15]`：图例标签（特征名，截取前 15 字符）。
  - `color=colors_ts[k]`：线条颜色。
  - `alpha=0.8`：透明度。
  - `linewidth=1.8`：线宽 1.8。

### 5.3 设置标签和图例

```python
ax4.set_xlabel('Bootstrap Iteration', fontsize=11)
ax4.set_ylabel('SHAP Importance', fontsize=11)
ax4.set_title('Importance Trajectory Across Bootstraps\n(should be stable, not erratic)',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='upper left')
ax4.grid(True, alpha=0.2)
```

- **x 轴**：Bootstrap 迭代次数（0–29）。
- **y 轴**：SHAP 重要性。
- **标题**："Importance Trajectory Across Bootstraps (should be stable, not erratic)"——"应该稳定，不应 erratic（ erratic = 不规则、跳跃）"。
- **`legend(loc='upper left')`**：图例放在左上角。

### 5.4 子图 4 的解读

**如何读取子图 4**：

- **每条线** = 一个特征在 30 次迭代中的重要性变化。
- **线波动越小** = 越稳定。
- **线波动越大** = 越不稳定。
- **观察**：
  - `year` 的线在 0.20–0.25 之间波动，幅度小 → 稳定。
  - `Extension` 的线在 0.11–0.15 之间波动 → 稳定。
  - `Code.Profession`（如果在前 5）的线波动剧烈 → 不稳定。

> 💡 **小贴士：为什么用折线图而不是散点图？**
>
> 折线图把同一特征的 30 个点连起来，能看出"趋势"和"波动模式"：
>
> - 如果线条平稳（近似水平），说明特征重要性稳定。
> - 如果线条跳跃剧烈，说明特征重要性不稳定。
> - 如果线条有上升/下降趋势，可能说明 Bootstrap 序列有偏差（但通常不应有趋势）。
>
> 散点图只能看到点的分布，看不到"顺序信息"。

***

## 六、子图 5：排名稳定性分布

```python
# === 子图 5: 排名稳定性分布 ===
ax5 = axes[1, 1]
ax5.hist(rank_correlations, bins=15, color='skyblue',
         edgecolor='black', alpha=0.7, density=True)
ax5.axvline(np.mean(rank_correlations), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(rank_correlations):.4f}')
ax5.axvline(np.median(rank_correlations), color='darkred', linestyle=':',
            linewidth=1.5, label=f'Median = {np.median(rank_correlations):.4f}')
ax5.set_xlabel("Spearman's Rank Correlation", fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title('Feature Ranking Stability\n(higher = more stable ranking)',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.2, axis='y')
```

### 6.1 绘制直方图

```python
ax5 = axes[1, 1]
ax5.hist(rank_correlations, bins=15, color='skyblue',
         edgecolor='black', alpha=0.7, density=True)
```

- **`ax5 = axes[1, 1]`**：选择中下子图。
- **`rank_correlations`**：30 个 Spearman ρ 值。
- **`bins=15`**：分 15 个箱子。
- **`color='skyblue'`**：天蓝色。
- **`edgecolor='black'`**：箱子边缘黑色。
- **`alpha=0.7`**：透明度。
- **`density=True`**：归一化为密度（面积=1），而不是计数。这样直方图可以和概率密度曲线比较。

### 6.2 添加均值和中位数垂直线

```python
ax5.axvline(np.mean(rank_correlations), color='red', linestyle='--',
            linewidth=2, label=f'Mean = {np.mean(rank_correlations):.4f}')
ax5.axvline(np.median(rank_correlations), color='darkred', linestyle=':',
            linewidth=1.5, label=f'Median = {np.median(rank_correlations):.4f}')
```

- **`axvline(x, ...)`**：在 x 位置画垂直线。
- **均值线**：红色虚线（`linestyle='--'`），线宽 2。
- **中位数线**：暗红色点线（`linestyle=':'`），线宽 1.5。
- **`label=...`**：图例标签，显示具体数值。

### 6.3 设置标签

```python
ax5.set_xlabel("Spearman's Rank Correlation", fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title('Feature Ranking Stability\n(higher = more stable ranking)',
              fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.2, axis='y')
```

- **x 轴**：Spearman 排名相关系数。
- **y 轴**：密度。
- **标题**："Feature Ranking Stability (higher = more stable ranking)"。

### 6.4 子图 5 的解读

**如何读取子图 5**：

- **直方图** = 30 个 ρ 值的分布。
- **红色虚线** = 平均 ρ（0.9067）。
- **暗红点线** = 中位数 ρ（1.0000）。
- **观察**：
  - 大部分 ρ 值集中在 1.0 附近（排名完全一致）。
  - 有少数 ρ 值较低（如 0.4857），对应排名大幅偏离的迭代。
  - 中位数 = 1.0 说明超过一半的迭代排名完全一致。

***

## 七、子图 6：CI 宽度排名

```python
# === 子图 6: 置信区间宽度排名 ===
ax6 = axes[1, 2]
ci_sorted = np.argsort(ci_width)[::-1]  # 按 CI 宽度降序（最宽的在前）
colors_ci = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
ax6.bar(range(top_n), ci_width[ci_sorted], color=colors_ci, alpha=0.75, edgecolor='white')
ax6.set_xticks(range(top_n))
ax6.set_xticklabels([feature_names[i][:15] for i in ci_sorted],
                     rotation=30, ha='right', fontsize=9)
ax6.set_ylabel('CI Width (97.5% - 2.5%)', fontsize=11)
ax6.set_title('Feature Uncertainty Ranking (wider = less certain)',
              fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.2, axis='y')
```

### 7.1 准备数据

```python
ax6 = axes[1, 2]
ci_sorted = np.argsort(ci_width)[::-1]  # 按 CI 宽度降序
colors_ci = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))
```

- **`ax6 = axes[1, 2]`**：选择右下角子图。
- **`ci_sorted = np.argsort(ci_width)[::-1]`**：按 CI 宽度**降序**排列的索引（最宽的在前）。`[::-1]` 反转升序为降序。
- **`colors_ci = plt.cm.Reds(np.linspace(0.3, 0.9, top_n))`**：从 Reds 色图生成 6 个颜色，从浅红（0.3）到深红（0.9）。CI 越宽，颜色越深。

### 7.2 绘制条形图

```python
ax6.bar(range(top_n), ci_width[ci_sorted], color=colors_ci, alpha=0.75, edgecolor='white')
```

- **`range(top_n)`**：x 轴位置 `[0, 1, 2, 3, 4, 5]`。
- **`ci_width[ci_sorted]`**：条形高度，按 CI 宽度降序排列。
- **`color=colors_ci`**：颜色，从浅红到深红。
- **`alpha=0.75`**：透明度。
- **`edgecolor='white'`**：边缘白色。

### 7.3 设置 x 轴标签

```python
ax6.set_xticks(range(top_n))
ax6.set_xticklabels([feature_names[i][:15] for i in ci_sorted],
                     rotation=30, ha='right', fontsize=9)
```

按 CI 宽度降序的特征名，旋转 30 度。

### 7.4 设置标签

```python
ax6.set_ylabel('CI Width (97.5% - 2.5%)', fontsize=11)
ax6.set_title('Feature Uncertainty Ranking (wider = less certain)',
              fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.2, axis='y')
```

- **y 轴**：CI 宽度（P97.5 - P2.5）。
- **标题**："Feature Uncertainty Ranking (wider = less certain)"——"越宽 = 越不确定"。

### 7.5 子图 6 的解读

**如何读取子图 6**：

- **条形高度** = CI 宽度，越高越不确定。
- **颜色**：深红 = 高不确定性。
- **观察**：
  - `Raca.Color` 的 CI 宽度最大（0.052）→ 不确定性最高。
  - `Age` 和 `Diagnostic.means` 的 CI 宽度最小（≈0.02）→ 不确定性最低。
  - 注意：CI 宽度小不等于重要性高，只是"估计精确"。

***

## 八、保存图片

```python
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "21a_shap_stability_bootstrap.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [图] 21a_shap_stability_bootstrap.png 已保存")
```

### 8.1 `plt.tight_layout()`

自动调整子图间距，防止标题、标签重叠。这是 matplotlib 多子图绘制的标准做法。

### 8.2 `plt.savefig(...)`

保存图片到文件：

- **`os.path.join(IMG_DIR, "21a_shap_stability_bootstrap.png")`**：文件路径 `img/21a_shap_stability_bootstrap.png`。
- **`dpi=150`**：分辨率 150（每英寸 150 像素）。对于 20×12 英寸的图形，总像素 3000×1800，足够清晰。
- **`bbox_inches='tight'`**：自动裁剪空白边缘，让图片更紧凑。

### 8.3 `plt.close()`

关闭图形，释放内存。在循环或批量绘图时很重要，防止内存泄漏。

### 8.4 实际输出

图片保存在 `img/21a_shap_stability_bootstrap.png`，包含 6 张子图。

![SHAP稳定性Bootstrap分析综合图](../img/21a_shap_stability_bootstrap.png)

***

## 九、保存结果到 CSV

```python
# ============================================================================
# 4. 保存详细结果
# ============================================================================
stability_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': mean_imp,
    'Std_SHAP': std_imp,
    'CV': cv_imp,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper,
    'CI_Width': ci_width
}).sort_values('Mean_SHAP', ascending=False)

stability_df.to_csv(os.path.join(RESULTS_DIR, "24_shap_stability_bootstrap.csv"),
                    index=False, float_format='%.6f')
```

### 9.1 构造 DataFrame

```python
stability_df = pd.DataFrame({
    'Feature': feature_names,
    'Mean_SHAP': mean_imp,
    'Std_SHAP': std_imp,
    'CV': cv_imp,
    'CI_Lower': ci_lower,
    'CI_Upper': ci_upper,
    'CI_Width': ci_width
}).sort_values('Mean_SHAP', ascending=False)
```

把 6 个统计量组装成 DataFrame：

- **`'Feature'`**：特征名（NumPy 数组）。
- **`'Mean_SHAP'`**：平均重要性。
- **`'Std_SHAP'`**：标准差。
- **`'CV'`**：变异系数。
- **`'CI_Lower'`**：CI 下界。
- **`'CI_Upper'`**：CI 上界。
- **`'CI_Width'`**：CI 宽度。
- **`.sort_values('Mean_SHAP', ascending=False)`**：按 Mean\_SHAP 降序排列。

### 9.2 保存到 CSV

```python
stability_df.to_csv(os.path.join(RESULTS_DIR, "24_shap_stability_bootstrap.csv"),
                    index=False, float_format='%.6f')
```

- **`to_csv(...)`**：保存到 CSV 文件。
- **`index=False`**：不保存行索引（避免多一列序号）。
- **`float_format='%.6f'`**：浮点数保留 6 位小数。

### 9.3 实际 CSV 内容

根据 `results/24_shap_stability_bootstrap.csv`：

```csv
Feature,Mean_SHAP,Std_SHAP,CV,CI_Lower,CI_Upper,CI_Width
year,0.224881,0.014320,0.063678,0.199852,0.246152,0.046300
Extension,0.132820,0.010982,0.082685,0.113037,0.153675,0.040638
Raca.Color,0.072443,0.013179,0.181927,0.046228,0.098210,0.051983
Diagnostic.means,0.050581,0.006417,0.126870,0.039798,0.060872,0.021074
Age,0.035635,0.005813,0.163131,0.027106,0.047151,0.020045
Code.Profession,0.024023,0.009485,0.394822,0.011533,0.044311,0.032779
```

***
 

## 十、6 张子图的完整解读

![SHAP稳定性Bootstrap分析综合图](../img/21a_shap_stability_bootstrap.png)

### 10.1 子图 1：特征重要性与 95% CI

- **条形高度**：Mean SHAP，从高到低排列。
- **误差棒**：95% CI 范围。
- **解读**：`year` 最高（0.225），误差棒短 → 重要性高且估计精确。`Code.Profession` 最低（0.024），误差棒相对高度大 → 重要性低且不确定。

### 10.2 子图 2：CV 稳定性排名

- **条形长度**：CV，从短到长排列（最稳定的在顶部）。
- **颜色**：绿色 = 稳定，红色 = 不稳定。
- **解读**：`year`（CV=0.064）和 `Extension`（CV=0.083）是绿色短条 → 高度稳定。`Code.Profession`（CV=0.395）是红色长条 → 不稳定。

### 10.3 子图 3：Top 3 特征的 Bootstrap 分布

- **箱体**：30 次 Bootstrap 的分布（Q1, 中位数, Q3）。
- **散点**：30 个原始数据点。
- **解读**：`year` 箱体窄、位置高 → 重要性高且集中。`Raca.Color` 箱体较宽 → 有一定波动。

### 10.4 子图 4：重要性变化轨迹

- **每条线**：一个特征在 30 次迭代中的重要性变化。
- **解读**：`year` 的线平稳在 0.20–0.25 → 稳定。其他特征的线波动较小，整体平稳。

### 10.5 子图 5：排名稳定性分布

- **直方图**：30 个 Spearman ρ 的分布。
- **红色虚线**：平均 ρ（0.9067）。
- **暗红点线**：中位数 ρ（1.0000）。
- **解读**：大部分 ρ 集中在 1.0 附近（排名完全一致），有少数低值（如 0.4857）。

### 10.6 子图 6：CI 宽度排名

- **条形高度**：CI 宽度，从宽到窄排列。
- **颜色**：深红 = 高不确定性。
- **解读**：`Raca.Color` 的 CI 宽度最大（0.052）→ 不确定性最高。`Age` 和 `Diagnostic.means` 的 CI 宽度最小 → 估计最精确。

***
 

 

## 整体总结：4 个模块的核心收获

| 模块   | 主题            | 核心收获                                       |
| ---- | ------------- | ------------------------------------------ |
| 模块 0 | 数据加载与预处理      | 设置实验参数（5000 样本、30 次 Bootstrap），完成数据准备      |
| 模块 1 | Bootstrap 主循环 | 30 次有放回抽样、训练 RF、计算 SHAP，得到 `(30, 6)` 重要性数组 |
| 模块 2 | 稳定性统计量        | 计算 5 个指标（Mean、Std、CV、CI、Rank ρ），量化稳定性      |
| 模块 3 | 可视化与结果保存      | 6 张子图综合展示，保存 CSV 和 TXT 结果文件                |

### 核心结论

1. **`year`** **是最稳定且最重要的特征**（Mean=0.225, CV=0.064）——结论高度可信。
2. **`Code.Profession`** **是最不稳定的特征**（Mean=0.024, CV=0.395）——结论需谨慎。
3. **整体排名稳定**（平均 ρ=0.9067），但有少数迭代排名大幅偏离（最小 ρ=0.486）。
4. **Bootstrap 将点估计变为区间估计**——"year 重要性 0.22 \[95% CI 0.20–0.25]"比"0.27"更科学。
5. **交叉验证和 Bootstrap 稳定性互补**——一个评估性能稳定性，一个评估解释稳定性。

***
 

