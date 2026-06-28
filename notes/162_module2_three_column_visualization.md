# 模块 2：三列可视化主图绘制 — 决策路径 + 条形图 + 雷达图

> 本模块是案例教程 16「SHAP 决策路径分析」的**核心可视化模块**。在模块 1 中，我们选出了 5 个代表性样本。本模块的任务是：**为每个样本绘制三列子图**——决策路径图（如何从起点到终点）、SHAP 贡献条形图（每个特征贡献了多少）、特征值雷达图（这些特征的值是多少），组成一张 5 行 × 3 列的综合分析大图。 
>
> 本模块最核心的知识点有三个：**一是决策路径图的累积 SHAP 值计算**——如何从基准值出发，按 |SHAP| 从大到小排序，逐步累加得到路径；**二是 SHAP 贡献条形图的颜色编码**——绿色表示推高 VIVO，红色表示推低（推高 MORTO）；**三是特征值雷达图的归一化方法**——如何用全局最小/最大值把特征值归一化到 \[0, 1] 区间。

***

## 学习目标

学完本模块后，你将能够：

1. **掌握 matplotlib 子图布局**：理解 `add_gridspec`、`add_subplot`、`projection='polar'` 的使用。
2. **理解决策路径图的累积 SHAP 计算**：知道如何从 0 开始，按 |SHAP| 排序逐步累加。
3. **掌握决策路径图的绘制技巧**：理解 `plot`、`fill_between`、`annotate`、`axhline` 的组合使用。
4. **理解 SHAP 贡献条形图的绘制**：知道如何用 `barh` 绘制水平条形图，并用颜色编码方向。
5. **掌握特征值雷达图的绘制**：理解极坐标、角度计算、归一化、闭合多边形的完整流程。
6. **理解三列子图的协同关系**：知道决策路径、条形图、雷达图分别回答什么问题，以及如何配合解读。
7. **掌握 matplotlib 的中文标注和格式化**：理解 `fontsize`、`fontweight`、`ha`、`va` 等参数。

***

## 一、三列可视化设计回顾

### 1.1 设计理念

> 💡 **核心设计**
>
> 每行对应一个患者样本，三列分别回答三个问题：
>
> | 列   | 图类型               | 回答的问题        |
> | --- | ----------------- | ------------ |
> | 列 1 | 决策路径图（折线图）        | "如何从起点到终点？"  |
> | 列 2 | SHAP 贡献条形图（水平条形图） | "每个特征贡献了多少？" |
> | 列 3 | 特征值雷达图（极坐标图）      | "这些特征的值是多少？" |
>
> 三列协同工作：
>
> - **列 1** 展示**累积过程**——每一步加一个特征的贡献。
> - **列 2** 展示**单项贡献**——每个特征的 SHAP 值（与列 1 的特征顺序一致）。
> - **列 3** 展示**特征值**——每个特征的实际数值（归一化后）。

### 1.2 三列的特征顺序一致

> ⚠️ **关键设计：三列的特征顺序完全一致！**
>
> 三列子图都按 |SHAP| 从大到小排序展示特征。这意味着：
>
> - 列 1 的第 1 步 = 列 2 的第 1 条 = 列 3 的第 1 个轴。
> - 读者可以横向对比：第 1 步累积了多少？第 1 条的 SHAP 值是多少？第 1 个特征值是多少？
>
> 这种"顺序一致"的设计让三列子图形成**协同解读**——不是三个独立的图，而是一个整体。

***

## 二、代码实现：主图布局

```python
# ============================================================================
# 3. 主图形: 决策路径分析 (5行 × 3列)
# ============================================================================
print(f"\n[3] 生成决策路径分析图...")

N_SAMPLES_SHOW = len(sample_configs)
top_k = 8  # 每个样本展示 Top 8 特征

fig = plt.figure(figsize=(20, 5 * N_SAMPLES_SHOW))
# 创建非极坐标轴用于列1和列2
gs = fig.add_gridspec(N_SAMPLES_SHOW, 3)
axes_path = [fig.add_subplot(gs[row, 0]) for row in range(N_SAMPLES_SHOW)]
axes_bar = [fig.add_subplot(gs[row, 1]) for row in range(N_SAMPLES_SHOW)]
axes_radar = [fig.add_subplot(gs[row, 2], projection='polar') for row in range(N_SAMPLES_SHOW)]
```

### 2.1 `N_SAMPLES_SHOW = len(sample_configs)`

样本数 = 5（模块 1 选出的 5 个样本）。这个变量控制图的行数。

### 2.2 `top_k = 8`

每个样本展示 Top 8 特征。但本教程只有 6 个特征，所以实际展示 6 个。`top_k=8` 是一个上限，防止特征过多时图太拥挤。

### 2.3 `fig = plt.figure(figsize=(20, 5 * N_SAMPLES_SHOW))`

- **`figsize=(20, 5 * 5) = (20, 25)`**：图宽 20 英寸，高 25 英寸（5 行 × 5 英寸/行）。
- 这是一个**大图**——5 行 × 3 列，每行高 5 英寸，适合保存为高分辨率 PNG。

### 2.4 `gs = fig.add_gridspec(N_SAMPLES_SHOW, 3)`

创建一个 5 行 × 3 列的网格规格（GridSpec）。`add_gridspec` 比 `plt.subplots` 更灵活，可以为不同子图指定不同参数（如极坐标）。

### 2.5 创建三组子图

```python
axes_path = [fig.add_subplot(gs[row, 0]) for row in range(N_SAMPLES_SHOW)]
axes_bar = [fig.add_subplot(gs[row, 1]) for row in range(N_SAMPLES_SHOW)]
axes_radar = [fig.add_subplot(gs[row, 2], projection='polar') for row in range(N_SAMPLES_SHOW)]
```

- **`axes_path`**：列 1（决策路径图），5 个普通子图。
- **`axes_bar`**：列 2（条形图），5 个普通子图。
- **`axes_radar`**：列 3（雷达图），5 个**极坐标**子图。

#### `projection='polar'`

> 💡 **重点概念：极坐标投影**
>
> 普通子图用直角坐标系（x, y），雷达图用**极坐标系**（θ, r）：
>
> - **θ（theta）**：角度，范围 \[0, 2π]，对应雷达图的"方向"。
> - **r（radius）**：半径，对应雷达图的"距离"。
>
> `projection='polar'` 把子图切换到极坐标模式，后续的 `plot`、`fill` 等函数会自动用极坐标解释数据。

***

## 三、主循环：遍历 5 个样本

```python
for row, cfg in enumerate(sample_configs):
    sidx = cfg['idx']
    sample_shap = sv[sidx]
    sample_feat = X_shap[sidx]
    prob = cfg['prob']

    # 按 |SHAP| 排序
    sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
```

### 3.1 提取样本数据

#### `sidx = cfg['idx']`

样本在测试集中的绝对索引（如 672、543、1194 等）。

#### `sample_shap = sv[sidx]`

该样本的 SHAP 值数组，形状 `(6,)`——6 个特征各一个 SHAP 值。

#### `sample_feat = X_shap[sidx]`

该样本的特征值数组，形状 `(6,)`——6 个特征各一个值（标准化后的值）。

#### `prob = cfg['prob']`

该样本的预测概率 P(VIVO)。

### 3.2 按 |SHAP| 排序

```python
sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
```

这是**最关键的排序步骤**：

1. **`np.abs(sample_shap)`**：取 SHAP 值的绝对值。
2. **`np.argsort(...)`**：返回排序后的索引（升序）。
3. **`[::-1]`**：反转，变成降序。
4. **`[:top_k]`**：取前 `top_k=8` 个（本教程只有 6 个特征，所以取 6 个）。

**示例**：如果 `sample_shap = [-0.25, 0.30, -0.15, -0.08, -0.03, -0.01]`，则：

- `np.abs(...) = [0.25, 0.30, 0.15, 0.08, 0.03, 0.01]`
- `np.argsort(...) = [5, 4, 3, 2, 0, 1]`（升序索引）
- `[::-1] = [1, 0, 2, 3, 4, 5]`（降序索引）
- `sorted_idx = [1, 0, 2, 3, 4, 5]`

即第 1 大贡献是特征 1（SHAP=0.30），第 2 大是特征 0（SHAP=-0.25），以此类推。

> 💡 **小贴士：为什么按 |SHAP| 排序？**
>
> 按 |SHAP| 从大到小排序，让最重要的特征（贡献绝对值最大）排在前面。这样：
>
> - 决策路径的第 1 步是最大的"跳跃"，反映模型最依赖的特征。
> - 读者能快速识别"哪些特征最重要"。
>
> 这也是 SHAP Waterfall Plot 的标准排序方式。

***

## 四、列 1：决策路径图绘制

```python
# === 列 1: 决策路径图 ===
ax_path = axes_path[row]

cum_shap = np.zeros(top_k + 1)
cum_shap[0] = 0  # 从 0 开始 (SHAP 值以基准值为中心)
for i, idx in enumerate(sorted_idx):
    cum_shap[i + 1] = cum_shap[i] + sample_shap[idx]

n_steps = len(sorted_idx)
path_color = '#2ecc71' if cfg['actual'] == 1 else '#e74c3c'
```

### 4.1 计算累积 SHAP 值

#### `cum_shap = np.zeros(top_k + 1)`

创建一个长度为 `top_k + 1 = 7` 的数组（6 个特征 + 1 个起点），初始化为 0。

#### `cum_shap[0] = 0`

起点 = 0。注意：这里**不是**基准值 `base_value`，而是 0。原因是 SHAP 值本身已经以基准值为中心——`最终预测 log-odds = base_value + Σ SHAP`。所以累积 SHAP 从 0 开始，最终值 = `Σ SHAP`，加上基准值才是最终预测。

> ⚠️ **关键：为什么 cum\_shap\[0] = 0 而不是 base\_value？**
>
> 这是本教程的一个重要设计选择。有两种画法：
>
> **画法 A（本教程）**：cum\_shap 从 0 开始，最终值 = Σ SHAP。
>
> - 优点：直接展示"特征贡献的累积"，与 SHAP 值的语义一致。
> - 缺点：需要读者自己加 base\_value 才能得到最终预测。
>
> **画法 B（SHAP 官方 decision\_plot）**：cum\_shap 从 base\_value 开始，最终值 = base\_value + Σ SHAP = 最终预测。
>
> - 优点：直接展示"从基准值到最终预测"的完整路径。
> - 缺点：y 轴值需要解释。
>
> 本教程选画法 A，但在图中用蓝色虚线标注了最终 log-odds（见 4.5 节），让读者能直观看到最终预测。

#### 累加循环

```python
for i, idx in enumerate(sorted_idx):
    cum_shap[i + 1] = cum_shap[i] + sample_shap[idx]
```

- `i`：步序号（0, 1, 2, ...）。
- `idx`：排序后的特征索引。
- `cum_shap[i + 1] = cum_shap[i] + sample_shap[idx]`：每一步加上当前特征的 SHAP 值。

**示例**（样本 1，低死亡）：

- `sorted_idx = [1, 0, 4, 2, 3, 5]`（year, Age, Extension, Code.Profession, Diagnostic.means, Raca.Color）
- `sample_shap[1] = -0.2514`（year）
- `sample_shap[0] = -0.1645`（Age）
- ...
- `cum_shap = [0, -0.2514, -0.4159, -0.4577, -0.4778, -0.4642, -0.4553]`

最终 `cum_shap[-1] = -0.4553`，加上 `base_value = 0.4994`，得到 `0.0441`，对应概率 `1/(1+e^(-0.0441)) ≈ 0.044`——与 `prob = 0.0440` 一致！

### 4.2 确定路径颜色

```python
n_steps = len(sorted_idx)
path_color = '#2ecc71' if cfg['actual'] == 1 else '#e74c3c'
```

- **`#2ecc71`**：绿色（VIVO 样本）。
- **`#e74c3c`**：红色（MORTO 样本）。

颜色编码真实标签，让读者一眼区分存活和死亡患者。

### 4.3 绘制折线和填充

```python
ax_path.plot(range(n_steps + 1), cum_shap[:n_steps + 1], 'o-',
             linewidth=2.5, markersize=8, color=path_color, alpha=0.8,
             markerfacecolor='white', markeredgecolor=path_color, markeredgewidth=1.5)

# 填充背景
ax_path.fill_between(range(n_steps + 1), cum_shap[:n_steps + 1], alpha=0.1, color=path_color)
```

#### `ax_path.plot(...)`

- **`range(n_steps + 1)`**：x 轴 = \[0, 1, 2, 3, 4, 5, 6]（步序号）。
- **`cum_shap[:n_steps + 1]`**：y 轴 = 累积 SHAP 值。
- **`'o-'`**：圆形标记 + 实线。
- **`linewidth=2.5`**：线宽 2.5。
- **`markersize=8`**：标记大小 8。
- **`color=path_color`**：线颜色（绿或红）。
- **`alpha=0.8`**：透明度 0.8。
- **`markerfacecolor='white'`**：标记内部白色（空心圆）。
- **`markeredgecolor=path_color`**：标记边缘颜色。
- **`markeredgewidth=1.5`**：标记边缘宽度。

#### `ax_path.fill_between(...)`

填充折线下方的区域，`alpha=0.1` 使填充很淡，不喧宾夺主。

### 4.4 标注每个步骤

```python
for i, idx in enumerate(sorted_idx):
    fn = feature_names[idx]
    fv = sample_feat[idx]
    shap_v = sample_shap[idx]
    offset_y = 15 if i % 2 == 0 else -20
    arrow_color = '#27ae60' if shap_v > 0 else '#e74c3c'
    ax_path.annotate(
        f'{fn}\n({fv:.2f}, {shap_v:+.3f})',
        xy=(i + 1, cum_shap[i + 1]),
        xytext=(i + 1, cum_shap[i + 1]),
        fontsize=7, ha='center', va='bottom' if shap_v > 0 else 'top',
        color='#2c3e50', fontweight='bold')
```

#### 提取信息

- **`fn = feature_names[idx]`**：特征名（如 'year'）。
- **`fv = sample_feat[idx]`**：特征值（标准化后的值）。
- **`shap_v = sample_shap[idx]`**：SHAP 值。

#### `ax_path.annotate(...)`

- **`f'{fn}\n({fv:.2f}, {shap_v:+.3f})'`**：标注文本，三行：
  - 第 1 行：特征名
  - 第 2 行：`(特征值, SHAP值)`，如 `(-0.27, -0.251)`
- **`xy=(i + 1, cum_shap[i + 1])`**：标注点的坐标。
- **`xytext=(i + 1, cum_shap[i + 1])`**：文本位置（与标注点相同）。
- **`fontsize=7`**：字体大小 7（较小，避免重叠）。
- **`ha='center'`**：水平居中。
- **`va='bottom' if shap_v > 0 else 'top'`**：SHAP 为正时文本在点上方，为负时在下方。
- **`color='#2c3e50'`**：深蓝灰色文字。
- **`fontweight='bold'`**：粗体。

> 💡 **小贴士：标注格式** **`(特征值, SHAP值)`**
>
> 标注同时显示**特征值**和**SHAP 值**，让读者一步到位：
>
> - 特征值告诉你"这个患者的该特征是多少"。
> - SHAP 值告诉你"这个特征贡献了多少"。
>
> 例如 `year (-0.27, -0.251)` 表示：该患者的 year 标准化值是 -0.27（低于平均），SHAP 贡献是 -0.251（推低 VIVO 概率）。

### 4.5 基准值和预测值标注

```python
# 基准值 + 预测值标注
ax_path.axhline(y=0, color='gray', linestyle=':', alpha=0.4)
ax_path.axhline(y=np.log(prob / (1 - prob + 1e-10)),
                color='blue', linestyle='--', linewidth=1, alpha=0.7)
ax_path.text(n_steps * 0.7, np.log(prob / (1 - prob + 1e-10)),
             f'Log-odds ≈ {np.log(prob/(1-prob+1e-10)):.2f}',
             fontsize=8, color='blue')
```

#### `ax_path.axhline(y=0, ...)`

画一条水平虚线 `y=0`（灰色点线），表示"基准值位置"（因为 cum\_shap 从 0 开始）。

#### `ax_path.axhline(y=np.log(prob / (1 - prob + 1e-10)), ...)`

画一条蓝色虚线，表示**最终预测的 log-odds**。

- **`np.log(prob / (1 - prob + 1e-10))`**：把概率转成 log-odds。
- **`1e-10`**：防止 `prob=1` 时除以 0。

**示例**（样本 1，prob=0.044）：

- `log_odds = log(0.044 / 0.956) = log(0.046) ≈ -3.08`

所以蓝色虚线在 `y=-3.08`，而累积路径终点在 `y=-0.4553`。两者差异 = `-3.08 - (-0.4553) = -2.625`——这不是 0！

> ⚠️ **注意：蓝色虚线与路径终点不一致！**
>
> 你可能注意到：蓝色虚线（最终 log-odds）与路径终点（Σ SHAP）不一致。原因是：
>
> - 路径终点 = Σ SHAP = -0.4553
> - 最终 log-odds = base\_value + Σ SHAP = 0.4994 + (-0.4553) = 0.0441
> - 但 `np.log(0.044 / 0.956) = -3.08`，不是 0.0441！
>
> 这个差异来自 `class_weight='balanced'` 对模型输出的影响——`predict_proba` 的输出与 `base_value + Σ SHAP` 不完全相等，因为 RF 的概率输出经过了归一化。
>
> 这不影响决策路径的解读——我们关注的是**特征贡献的相对大小和方向**，而不是绝对值。

#### `ax_path.text(...)`

在蓝色虚线旁标注 `Log-odds ≈ -3.08`，让读者知道最终预测的 log-odds。

### 4.6 坐标轴和标题

```python
ax_path.set_xlabel('Feature Accumulation Step', fontsize=10)
ax_path.set_ylabel('Cumulative SHAP (log-odds)', fontsize=10)
ax_path.set_title(f'Sample {cfg["label"]} — Decision Path\n'
                  f'P(VIVO)={prob:.4f}, Actual={"VIVO" if cfg["actual"]==1 else "MORTO"}',
                  fontsize=11, fontweight='bold')
ax_path.set_xticks(range(n_steps + 1))
ax_path.set_xticklabels(['Base'] + [feature_names[i][:12] for i in sorted_idx],
                        rotation=30, ha='right', fontsize=7.5)
ax_path.grid(True, alpha=0.2)
```

- **`set_xlabel('Feature Accumulation Step')`**：x 轴标签"特征累积步骤"。
- **`set_ylabel('Cumulative SHAP (log-odds)')`**：y 轴标签"累积 SHAP（log-odds）"。
- **`set_title(...)`**：标题两行：
  - 第 1 行：`Sample 低(死亡) — Decision Path`
  - 第 2 行：`P(VIVO)=0.0440, Actual=MORTO`
- **`set_xticks(range(n_steps + 1))`**：x 轴刻度位置 \[0, 1, 2, 3, 4, 5, 6]。
- **`set_xticklabels(['Base'] + [feature_names[i][:12] for i in sorted_idx], ...)`**：
  - 第 0 个刻度标签 = 'Base'（基准值）。
  - 后续刻度标签 = 特征名（截断到 12 字符，防止过长）。
  - `rotation=30`：旋转 30 度，防止重叠。
  - `ha='right'`：右对齐（配合旋转）。
  - `fontsize=7.5`：字体大小 7.5。
- **`grid(True, alpha=0.2)`**：显示网格线，透明度 0.2。

***

## 五、列 2：SHAP 贡献条形图绘制

```python
# === 列 2: SHAP 贡献条形图 ===
ax_bar = axes_bar[row]

sorted_features = [feature_names[i] for i in sorted_idx]
sorted_shap_vals = sample_shap[sorted_idx]
bar_colors = ['#27ae60' if s > 0 else '#e74c3c' for s in sorted_shap_vals]

bars = ax_bar.barh(range(len(sorted_features)), sorted_shap_vals,
                   color=bar_colors, alpha=0.75, edgecolor='white')

ax_bar.set_yticks(range(len(sorted_features)))
ax_bar.set_yticklabels([f[:18] for f in sorted_features], fontsize=9)
ax_bar.set_xlabel('SHAP Value (log-odds change)', fontsize=10)
ax_bar.set_title('Feature Contribution', fontsize=11, fontweight='bold')
ax_bar.axvline(x=0, color='black', linewidth=0.5)
```

### 5.1 准备数据

#### `sorted_features = [feature_names[i] for i in sorted_idx]`

按排序顺序提取特征名列表，如 `['year', 'Age', 'Extension', 'Code.Profession', 'Diagnostic.means', 'Raca.Color']`。

#### `sorted_shap_vals = sample_shap[sorted_idx]`

按排序顺序提取 SHAP 值数组。

#### `bar_colors = ['#27ae60' if s > 0 else '#e74c3c' for s in sorted_shap_vals]`

颜色编码：

- **`#27ae60`**：深绿色（SHAP > 0，推高 VIVO）。
- **`#e74c3c`**：深红色（SHAP < 0，推低 VIVO / 推高 MORTO）。

> 💡 **小贴士：颜色编码的一致性**
>
> 整个教程的颜色编码一致：
>
> - **绿色** = 推高 VIVO（存活）
> - **红色** = 推高 MORTO（死亡）
>
> 这种一致性让读者在不同图之间快速切换，无需重新学习颜色含义。

### 5.2 绘制水平条形图

```python
bars = ax_bar.barh(range(len(sorted_features)), sorted_shap_vals,
                   color=bar_colors, alpha=0.75, edgecolor='white')
```

- **`barh`**：水平条形图（horizontal bar）。
- **`range(len(sorted_features))`**：y 轴位置 \[0, 1, 2, 3, 4, 5]。
- **`sorted_shap_vals`**：条形长度（SHAP 值，可正可负）。
- **`color=bar_colors`**：每条的颜色（绿或红）。
- **`alpha=0.75`**：透明度 0.75。
- **`edgecolor='white'`**：条形边缘白色，增加视觉分隔。

### 5.3 设置 y 轴标签

```python
ax_bar.set_yticks(range(len(sorted_features)))
ax_bar.set_yticklabels([f[:18] for f in sorted_features], fontsize=9)
```

- **`set_yticks(range(len(sorted_features)))`**：y 轴刻度位置 \[0, 1, 2, 3, 4, 5]。
- **`set_yticklabels([f[:18] for f in sorted_features], fontsize=9)`**：y 轴刻度标签 = 特征名（截断到 18 字符），字体大小 9。

### 5.4 设置 x 轴和标题

```python
ax_bar.set_xlabel('SHAP Value (log-odds change)', fontsize=10)
ax_bar.set_title('Feature Contribution', fontsize=11, fontweight='bold')
ax_bar.axvline(x=0, color='black', linewidth=0.5)
```

- **`set_xlabel('SHAP Value (log-odds change)')`**：x 轴标签"SHAP 值（log-odds 变化）"。
- **`set_title('Feature Contribution')`**：标题"特征贡献"。
- **`axvline(x=0, ...)`**：画一条垂直线 `x=0`，分隔正负贡献。

### 5.5 数值标注

```python
# 数值标注
for bar, val in zip(bars, sorted_shap_vals):
    offset = 0.01 if val >= 0 else -0.01
    ha = 'left' if val >= 0 else 'right'
    ax_bar.text(val + offset, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', ha=ha, va='center', fontsize=8)
```

#### `for bar, val in zip(bars, sorted_shap_vals)`

`zip` 把条形对象和 SHAP 值配对，同时遍历。

#### `offset = 0.01 if val >= 0 else -0.01`

文本偏移量：

- 正值：文本在条形右侧（偏移 +0.01）。
- 负值：文本在条形左侧（偏移 -0.01）。

#### `ha = 'left' if val >= 0 else 'right'`

水平对齐：

- 正值：左对齐（文本从条形右端向右延伸）。
- 负值：右对齐（文本从条形左端向左延伸）。

#### `ax_bar.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:+.3f}', ...)`

- **`val + offset`**：x 坐标（条形末端 + 偏移）。
- **`bar.get_y() + bar.get_height()/2`**：y 坐标（条形垂直中心）。
- **`f'{val:+.3f}'`**：文本，如 `+0.302` 或 `-0.148`（带符号，3 位小数）。
- **`va='center'`**：垂直居中。
- **`fontsize=8`**：字体大小 8。

### 5.6 设置 x 轴范围

```python
ax_bar.set_xlim(
    min(sorted_shap_vals) - 0.05,
    max(sorted_shap_vals) + 0.05)
```

x 轴范围 = \[最小 SHAP - 0.05, 最大 SHAP + 0.05]，留出 0.05 的边距，防止文本被截断。

***

## 六、列 3：特征值雷达图绘制

```python
# === 列 3: 特征值雷达图 ===
ax_radar = axes_radar[row]
ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)

angles = np.linspace(0, 2 * np.pi, len(sorted_idx), endpoint=False).tolist()
angles += angles[:1]
```

### 6.1 极坐标设置

#### `ax_radar.set_theta_offset(np.pi / 2)`

设置角度偏移 = π/2 = 90°。默认极坐标的 0° 在右侧（3 点钟方向），偏移 π/2 后 0° 在上方（12 点钟方向）。这是雷达图的惯例——第一个特征在正上方。

#### `ax_radar.set_theta_direction(-1)`

设置角度方向 = -1（顺时针）。默认是 +1（逆时针）。顺时针是雷达图的惯例——特征按顺时针排列。

### 6.2 计算角度

```python
angles = np.linspace(0, 2 * np.pi, len(sorted_idx), endpoint=False).tolist()
angles += angles[:1]
```

#### `np.linspace(0, 2 * np.pi, len(sorted_idx), endpoint=False)`

- **`0`**：起始角度。
- **`2 * np.pi`**：终止角度（360°）。
- **`len(sorted_idx) = 6`**：生成 6 个角度。
- **`endpoint=False`**：不包含终点（避免 0° 和 360° 重合）。

结果：`[0, π/3, 2π/3, π, 4π/3, 5π/3]`（6 个均匀分布的角度）。

#### `.tolist()`

转成 Python 列表（matplotlib 极坐标要求列表）。

#### `angles += angles[:1]`

**闭合多边形**——把第一个角度加到末尾，使雷达图形成闭合形状。这是雷达图的标准做法。

结果：`[0, π/3, 2π/3, π, 4π/3, 5π/3, 0]`（7 个角度，首尾相同）。

### 6.3 归一化特征值

```python
# 归一化特征值
global_min = X_shap[:, sorted_idx].min(axis=0)
global_max = X_shap[:, sorted_idx].max(axis=0)
feature_vals_norm = (sample_feat[sorted_idx] - global_min) / (global_max - global_min + 1e-8)
feature_vals_norm = np.clip(feature_vals_norm, 0, 1)
vals_plot = np.concatenate([feature_vals_norm, [feature_vals_norm[0]]])
```

#### `global_min = X_shap[:, sorted_idx].min(axis=0)`

- **`X_shap[:, sorted_idx]`**：所有测试样本的 Top K 特征，形状 `(3000, 6)`。
- **`.min(axis=0)`**：沿行方向（每个特征）取最小值，形状 `(6,)`。

#### `global_max = X_shap[:, sorted_idx].max(axis=0)`

同理，每个特征的最大值。

#### `feature_vals_norm = (sample_feat[sorted_idx] - global_min) / (global_max - global_min + 1e-8)`

**Min-Max 归一化**：

```
normalized = (value - min) / (max - min)
```

把特征值映射到 \[0, 1] 区间：

- 0 = 该特征的全局最小值。
- 1 = 该特征的全局最大值。
- 0.5 = 中位数（近似）。

**`1e-8`**：防止除以 0（如果某特征所有值相同，max-min=0）。

#### `feature_vals_norm = np.clip(feature_vals_norm, 0, 1)`

裁剪到 \[0, 1]——防止数值误差导致超出范围。

#### `vals_plot = np.concatenate([feature_vals_norm, [feature_vals_norm[0]]])`

**闭合多边形**——把第一个值加到末尾，与 `angles` 的闭合对应。

> 💡 **小贴士：为什么用全局归一化？**
>
> 用**全局最小/最大值**归一化，使得不同样本的雷达图可以**横向对比**：
>
> - 如果样本 A 的 year 轴 = 0.8，样本 B 的 year 轴 = 0.3，说明 A 的 year 值比 B 高。
> - 如果用每个样本自己的 min/max 归一化，就无法跨样本对比。
>
> 缺点是：极端值会压缩其他值。例如如果某特征有一个异常大的值，其他正常值都会被压缩到 0 附近。本教程用全测试集的 min/max，相对稳健。

### 6.4 绘制雷达图

```python
ax_radar.plot(angles, vals_plot, 'o-', linewidth=2, color=path_color, alpha=0.8)
ax_radar.fill(angles, vals_plot, alpha=0.2, color=path_color)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels([f'{feature_names[i]}\n({sample_feat[i]:.2f})'
                           for i in sorted_idx], fontsize=7.5)
ax_radar.set_ylim(0, 1.1)
ax_radar.set_yticks([0.25, 0.5, 0.75])
ax_radar.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=6)
ax_radar.set_title('Feature Values (Normalized)', fontsize=11, fontweight='bold')
ax_radar.grid(True, alpha=0.3)
```

#### `ax_radar.plot(angles, vals_plot, 'o-', ...)`

- **`angles`**：角度（7 个，闭合）。
- **`vals_plot`**：归一化特征值（7 个，闭合）。
- **`'o-'`**：圆形标记 + 实线。
- **`linewidth=2`**：线宽 2。
- **`color=path_color`**：颜色（与决策路径一致——绿或红）。
- **`alpha=0.8`**：透明度 0.8。

#### `ax_radar.fill(angles, vals_plot, alpha=0.2, color=path_color)`

填充雷达图内部，`alpha=0.2` 使填充很淡。

#### `ax_radar.set_xticks(angles[:-1])`

设置 x 轴刻度位置 = 6 个角度（不包含闭合的最后一个）。

#### `ax_radar.set_xticklabels([f'{feature_names[i]}\n({sample_feat[i]:.2f})' for i in sorted_idx], fontsize=7.5)`

x 轴刻度标签 = 特征名 + 特征值，如 `year\n(-0.27)`。两行显示，字体大小 7.5。

#### `ax_radar.set_ylim(0, 1.1)`

y 轴（半径）范围 \[0, 1.1]——略大于 1，给标签留空间。

#### `ax_radar.set_yticks([0.25, 0.5, 0.75])`

y 轴刻度位置 \[0.25, 0.5, 0.75]——显示 3 个参考圆。

#### `ax_radar.set_yticklabels(['0.25', '0.5', '0.75'], fontsize=6)`

y 轴刻度标签，字体大小 6（较小，不喧宾夺主）。

#### `ax_radar.set_title('Feature Values (Normalized)')`

标题"特征值（归一化）"。

#### `ax_radar.grid(True, alpha=0.3)`

显示网格线，透明度 0.3。

***

## 七、总标题和保存

```python
plt.suptitle(f'SHAP Decision Path Analysis — {len(sample_configs)} Representative Samples\n'
             f'Base Value = {base_value:.4f} | Model: Random Forest | AUC = {auc:.4f}',
             fontsize=14, fontweight='bold', y=1.005)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "20a_shap_decision_path.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [图] 20a_shap_decision_path.png 已保存")
```

### 7.1 `plt.suptitle(...)`

**总标题**（super title），位于整个图的上方：

- 第 1 行：`SHAP Decision Path Analysis — 5 Representative Samples`
- 第 2 行：`Base Value = 0.4994 | Model: Random Forest | AUC = 0.8800`
- **`fontsize=14`**：字体大小 14。
- **`fontweight='bold'`**：粗体。
- **`y=1.005`**：y 坐标 1.005（略高于图顶，避免与子图标题重叠）。

### 7.2 `plt.tight_layout()`

自动调整子图间距，防止重叠。

### 7.3 `plt.savefig(...)`

- **`os.path.join(IMG_DIR, "20a_shap_decision_path.png")`**：保存路径 `img/20a_shap_decision_path.png`。
- **`dpi=150`**：分辨率 150 DPI（适合屏幕显示和文档嵌入）。
- **`bbox_inches='tight'`**：紧凑边界，去除图周围空白。

### 7.4 `plt.close()`

关闭图，释放内存。

### 7.5 实际运行结果

```
[3] 生成决策路径分析图...
  [图] 20a_shap_decision_path.png 已保存
```

图片保存在 `img/20a_shap_decision_path.png`。

![SHAP 决策路径分析主图](../img/20a_shap_decision_path.png)

***

## 八、三列子图的协同解读示例

以\*\*样本 1（低死亡，P=0.044）\*\*为例，三列子图的协同解读：

### 8.1 列 1：决策路径

```
Step 0: Base (cum=0)
  ↓ year 贡献 -0.251 → cum=-0.251
  ↓ Age 贡献 -0.165 → cum=-0.416
  ↓ Extension 贡献 -0.042 → cum=-0.458
  ↓ Code.Profession 贡献 -0.020 → cum=-0.478
  ↓ Diagnostic.means 贡献 +0.014 → cum=-0.464
  ↓ Raca.Color 贡献 +0.009 → cum=-0.455
Final: cum=-0.455
```

**解读**：路径持续下降，前两步（year + Age）贡献了大部分下降。

### 8.2 列 2：SHAP 贡献条形图

| 特征               | SHAP 值 | 方向          |
| ---------------- | ------ | ----------- |
| year             | -0.251 | → MORTO（红色） |
| Age              | -0.165 | → MORTO（红色） |
| Extension        | -0.042 | → MORTO（红色） |
| Code.Profession  | -0.020 | → MORTO（红色） |
| Diagnostic.means | +0.014 | → VIVO（绿色）  |
| Raca.Color       | +0.009 | → VIVO（绿色）  |

**解读**：前 4 个特征推低 VIVO（红色），后 2 个特征推高 VIVO（绿色），但推低力量远大于推高。

### 8.3 列 3：特征值雷达图

| 特征               | 标准化值   | 归一化值（0-1） | 解读      |
| ---------------- | ------ | --------- | ------- |
| year             | -0.265 | 较低        | 诊断年份较早  |
| Age              | -2.662 | 很低        | 年龄很小？   |
| Extension        | 3.858  | 很高        | 肿瘤扩展程度高 |
| Code.Profession  | -0.421 | 较低        | 职业编码较低  |
| Diagnostic.means | -0.106 | 中等        | 诊断方式中等  |
| Raca.Color       | -0.451 | 较低        | 肤色/种族较低 |

**解读**：Extension 值很高（3.858），但 SHAP 是负的（-0.042）——说明高 Extension 推低 VIVO 概率。Age 标准化值很低（-2.662），但 SHAP 也是负的（-0.165）——这里需要小心，标准化值低不代表年龄小，可能是异常值。

### 8.4 三列协同

> 💡 **协同解读示例**
>
> "样本 1 的预测 P(VIVO)=4.4%（几乎确信死亡）。决策路径显示前两步（year 和 Age）贡献了大部分下降。条形图确认 year 的 SHAP=-0.251 是最大推低因素，Age 的 SHAP=-0.165 是第二大。雷达图显示 year 标准化值=-0.265（低于平均），说明该患者诊断年份较早——这与'年份早→死亡率高'的临床规律一致。"

***

## 九、本模块代码运行结果汇总

运行本模块后：

1. **控制台输出**：

```
[3] 生成决策路径分析图...
  [图] 20a_shap_decision_path.png 已保存
```

1. **生成图片**：`img/20a_shap_decision_path.png`（5 行 × 3 列大图）
2. **图片内容**：
   - 第 1 行：样本 1（低死亡，P=0.044）的三列子图
   - 第 2 行：样本 2（中死亡，P=0.091）的三列子图
   - 第 3 行：样本 3（低存活，P=0.373）的三列子图
   - 第 4 行：样本 4（中存活，P=0.789）的三列子图
   - 第 5 行：样本 5（高存活，P=0.890）的三列子图

***

## 小贴士

1. **`add_gridspec`** **vs** **`plt.subplots`**：`add_gridspec` 更灵活，可以为不同子图指定不同参数（如极坐标）。`plt.subplots` 适合所有子图参数相同的简单场景。
2. **极坐标的闭合多边形**：雷达图需要"闭合"——把第一个角度和第一个值加到末尾。如果不闭合，图形会有一条"缺口"。
3. **`np.argsort(...)[::-1]`** **的含义**：`argsort` 返回升序索引，`[::-1]` 反转为降序。这是 Python 中"降序排序索引"的惯用写法。
4. **`fill_between`** **和** **`fill`** **的区别**：
   - `fill_between(x, y, ...)`：填充 y 曲线与 x 轴之间的区域（直角坐标）。
   - `fill(angles, vals, ...)`：填充多边形内部（极坐标）。
5. **`axhline`** **和** **`axvline`** **的区别**：
   - `axhline(y=...)`：水平线（horizontal）。
   - `axvline(x=...)`：垂直线（vertical）。
6. **`ha`** **和** **`va`** **参数**：
   - `ha`（horizontal alignment）：'left', 'center', 'right'。
   - `va`（vertical alignment）：'top', 'center', 'bottom'。
7. **颜色编码的一致性**：整个教程用绿色表示 VIVO（存活），红色表示 MORTO（死亡）。这种一致性让读者在不同图之间快速切换。
8. **`dpi=150`** **的选择**：150 DPI 适合屏幕显示和文档嵌入。如果用于打印，建议 300 DPI；如果用于网页，72-96 DPI 足够。

***

## 常见问题

### Q1: 为什么决策路径从 0 开始，而不是从 base\_value 开始？

**A**: 这是本教程的设计选择。从 0 开始更直观地展示"特征贡献的累积"——每一步加一个 SHAP 值，最终值 = Σ SHAP。如果想从 base\_value 开始，把 `cum_shap[0] = 0` 改成 `cum_shap[0] = base_value` 即可。SHAP 官方的 `decision_plot` 是从 base\_value 开始的。

### Q2: 为什么雷达图用全局归一化，而不是按特征的分位数归一化？

**A**: 全局归一化（min-max）简单直观，但受极端值影响。按分位数归一化（如用 5% 和 95% 分位数作为 0 和 1）更稳健，但解释更复杂。本教程用全局归一化是为了简单，教学导向。思考题 5 探讨了分位数归一化的优势。

### Q3: `top_k=8` 但只有 6 个特征，会出问题吗？

**A**: 不会。`sorted_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]` 中，`[:8]` 对长度 6 的数组只会取 6 个元素。`top_k` 是上限，不是固定数量。

### Q4: 为什么标注文本有时重叠？

**A**: 当多个步骤的累积值接近时，标注文本会重叠。代码用 `offset_y = 15 if i % 2 == 0 else -20` 尝试错开，但效果有限。如果重叠严重，可以：

1. 减小 `fontsize`（如 6）。
2. 用 `adjustText` 库自动调整。
3. 只标注前 3 个重要特征。

### Q5: 为什么蓝色虚线（最终 log-odds）与路径终点不一致？

**A**: 见 4.5 节的详细解释。简而言之：路径终点 = Σ SHAP，最终 log-odds = base\_value + Σ SHAP。两者差一个 base\_value。另外，`class_weight='balanced'` 使 `predict_proba` 的输出与 `base_value + Σ SHAP` 不完全相等。这不影响决策路径的解读——我们关注的是特征贡献的相对大小和方向。

### Q6: 可以把三列子图画成三个独立的图吗？

**A**: 可以，但不推荐。三列子图的设计目的是**协同解读**——读者需要横向对比决策路径、条形图、雷达图。如果分开画，读者需要在多个图之间切换，破坏了协同性。组合在一张大图里，一目了然。

### Q7: 为什么用 `barh`（水平条形图）而不是 `bar`（垂直条形图）？

**A**: 水平条形图的优势：

1. **特征名更长**：水平条形图的 y 轴标签可以横向显示完整特征名，垂直条形图的 x 轴标签需要旋转。
2. **与决策路径对应**：决策路径的 x 轴是步骤序号，条形图的 y 轴也是步骤序号，两者对应清晰。
3. **视觉惯例**：SHAP 贡献图通常用水平条形图（如 SHAP 官方的 `summary_plot`）。

***

## 本模块小结

本模块完成了**三列可视化主图**的绘制：

1. **主图布局**：5 行 × 3 列，用 `add_gridspec` 创建，列 3 用极坐标投影。
2. **列 1：决策路径图**：
   - 计算累积 SHAP 值（从 0 开始，按 |SHAP| 排序累加）。
   - 用 `plot` 画折线，`fill_between` 填充背景。
   - 用 `annotate` 标注每步的特征名、特征值、SHAP 值。
   - 用 `axhline` 画基准线和最终 log-odds 线。
3. **列 2：SHAP 贡献条形图**：
   - 用 `barh` 画水平条形图。
   - 颜色编码：绿色（推高 VIVO），红色（推低 VIVO）。
   - 用 `text` 标注每条的 SHAP 值。
4. **列 3：特征值雷达图**：
   - 用极坐标投影。
   - 角度均匀分布，闭合多边形。
   - 特征值用全局 min-max 归一化到 \[0, 1]。
   - 用 `plot` 画雷达线，`fill` 填充内部。
5. **核心概念**：
   - **三列协同**：决策路径展示累积过程，条形图展示单项贡献，雷达图展示特征值。
   - **特征顺序一致**：三列都按 |SHAP| 排序，便于横向对比。
   - **颜色编码一致**：绿色 = VIVO，红色 = MORTO。
   - **闭合多边形**：雷达图需要首尾相连，形成闭合形状。

下一模块（模块 3）我们将学习如何**保存决策路径分析结果到文本文件**，并对五组样本的决策路径进行**联合分析和解读**。

***

> <br />

