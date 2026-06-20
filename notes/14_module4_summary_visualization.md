# 模块 4：汇总与可视化 — 统计显著 ≠ 预测力强

> **核心思想**：在模块 2 和模块 3 中，我们分别对数值特征和分类特征做了统计检验，得到了 p 值和效应量。但一堆孤立的数字很难让人一眼看出规律。本模块的核心任务是**把所有结果汇总到一张表里，并用 5 张图把它们讲清楚**。更重要的是，我们要回答一个初学者最容易忽略的问题：**p 值显著，就代表这个特征对预测有用吗？** 答案是——**不一定**。在大样本下，p 值会"夸大"微小差异的显著性，而效应量才是衡量"实际预测力"的更可靠锚点。

本模块对应代码文件 `src/02_statistical_analysis.py` 的第 307–603 行，涵盖汇总逻辑和 5 张可视化图（图 5a–5e）。

---

## 学习目标

完成本模块后，你将能够：

1. **理解汇总逻辑**：学会用列表 + 字典的方式把数值型结果（`num_df`）和分类型结果（`cat_df`）合并到统一的 `summary_df`，并按 p 值排序。
2. **掌握 `-log10(p)` 转换**：理解为什么要把 p 值取负对数再画图，并能口算 p=0.05→1.3、p=0.001→3、p=1e-10→10 的对应关系。
3. **读懂效应量分级配色**：理解 ">0.3 红、>0.1 橙、≤0.1 蓝" 的三色逻辑，以及它对应 Cohen/Cramér 的"中/小/弱"效应分级。
4. **解读 p 值 vs 效应量散点图**：这是本模块最重要的一张图，学会从"左上角"和"右下角"两个区域读出"统计显著但实际无用"和"实际有用且统计显著"两种特征。
5. **掌握 matplotlib 自定义图例**：理解 `from matplotlib.lines import Line2D` 如何手动构造图例元素，解决散点图多编码（颜色+大小+形状）下默认图例不够用的问题。
6. **掌握 `nsmallest` 取 Top 特征**：理解 `num_df.nsmallest(min(4, len(num_df)), 'P_Value')` 中 `min` 的保护作用。
7. **理解 `pd.crosstab(..., normalize='index')`**：明白按行归一化让每个类别内 VIVO/MORTO 占比之和=1，从而直接读"存活率"。
8. **批判性理解"大样本陷阱"**：明白 21 万样本下即使 V=0.017 也会 p<0.05，统计检验关心"有无差异"，机器学习关心"能否泛化"。

---

## 1. 背景回顾

在前面的模块中，我们已经完成了两件事：

- **模块 2**：对 4 个数值特征逐一做检验（Shapiro-Wilk 正态性 → t 检验或 Mann-Whitney U 检验），输出 `num_df`，包含 `Feature`、`Test`、`P_Value`、`Effect_Size`（Cohen's d）、`Effect_Type`、`Significant_0.05`、`Significant_Bonf` 等列。
- **模块 3**：对 18 个分类特征做卡方检验，输出 `cat_df`，包含 `Feature`、`P_Value`、`Cramér_V`、`Significant_0.05`、`Significant_Bonf` 等列。

现在的问题是：**这两张表的列名不一样**（数值表叫 `Effect_Size`，分类表叫 `Cramér_V`），**检验方法也不一样**（t 检验 / Mann-Whitney U vs 卡方检验）。要把它们放在一起比较，必须先"统一格式"。这就是本模块第一步要做的事。

---

## 2. 汇总：合并数值与分类结果

### 2.1 设计思路

我们希望建立一张"统一格式"的汇总表 `summary_df`，每一行是一个特征，列包括：

| 列名 | 含义 |
|------|------|
| `Feature` | 特征名 |
| `Type` | 类型（"数值型" / "分类型"） |
| `Test` | 使用的检验方法 |
| `P_Value` | p 值 |
| `Effect_Size` | 效应量（数值型是 Cohen's d，分类型是 Cramér's V） |
| `Effect_Type` | 效应量类型（"Cohen's d" 或 "Cramér's V"） |
| `Significant_0.05` | 在 α=0.05 水平是否显著（"Yes"/"No"） |
| `Significant_Bonf` | Bonferroni 校正后是否显著 |

> **重要概念**：Cohen's d 和 Cramér's V 虽然公式完全不同（一个是均值差除以标准差，一个是基于卡方统计量的修正），但它们的取值范围和"小/中/大"分级阈值非常接近（0.2/0.5/0.8 vs 0.1/0.3/0.5）。所以在汇总表里把它们都放进 `Effect_Size` 列做横向比较是合理的——这也是为什么后面的图 5b 能把它们画在同一根 x 轴上。

### 2.2 代码实现

```python
all_results = []

for _, row in num_df.iterrows():
    all_results.append({
        'Feature': row['Feature'], 'Type': '数值型', 'Test': row['Test'],
        'P_Value': row['P_Value'], 'Effect_Size': row['Effect_Size'],
        'Effect_Type': row['Effect_Type'],
        'Significant_0.05': row['Significant_0.05'],
        'Significant_Bonf': row['Significant_Bonf']
    })

for _, row in cat_df.iterrows():
    all_results.append({
        'Feature': row['Feature'], 'Type': '分类型', 'Test': '卡方检验',
        'P_Value': row['P_Value'], 'Effect_Size': row['Cramér_V'],
        'Effect_Type': "Cramér's V",
        'Significant_0.05': row['Significant_0.05'],
        'Significant_Bonf': row['Significant_Bonf']
    })

summary_df = pd.DataFrame(all_results).sort_values('P_Value')
```

### 2.3 逐行解释

#### `all_results = []`

先建一个空列表，用来"收集"每一行结果。这种"先收集再转 DataFrame"的写法在 pandas 中比"逐行 append 到 DataFrame"要快得多，因为后者每次都会创建新对象。

#### `for _, row in num_df.iterrows():`

- `iterrows()` 是 pandas 提供的按行迭代器，每次返回 `(index, row)` 二元组。
- `_` 是 Python 约定俗成的"我不关心这个变量"的占位符，这里我们不需要原始索引。
- `row` 是一个 Series，可以用 `row['Feature']`、`row['P_Value']` 等方式访问列值。

#### `all_results.append({...})`

把每一行结果**重新打包成一个统一格式的字典**，再追加到列表。注意几个关键映射：

- 数值型的 `Effect_Size` 直接来自 `row['Effect_Size']`（Cohen's d）。
- 分类型的效应量在原表里叫 `Cramér_V`，但我们在字典里把它**重命名**为 `Effect_Size`，这样两张表的效应量就统一到同一列名下。
- `Test` 列：数值型沿用原表的检验方法（可能是 t 检验或 Mann-Whitney U），分类型统一写 `'卡方检验'`。

#### `summary_df = pd.DataFrame(all_results).sort_values('P_Value')`

- `pd.DataFrame(all_results)`：把列表 of 字典转成 DataFrame，字典的 key 自动成为列名。
- `.sort_values('P_Value')`：按 p 值**升序**排序（默认 `ascending=True`），p 值最小（最显著）的特征排在最前面。

> **小贴士**：为什么按 p 值排序而不是按效应量排序？因为汇总表的第一用途是"看哪些特征显著"，p 值是最直接的显著性指标。后面画图 5b 时会再按效应量排序，画图 5c 时则保持原顺序——同一张表，不同视角。

### 2.4 实际运行结果

代码运行后，控制台会输出：

```
  ▶ 共分析 22 个特征:
     数值型: 4  |  分类型: 18

  ▶ 在 α=0.05 水平显著: 22 个
  ▶ Bonferroni 校正后显著: 22 个
```

> **关键发现**：22 个特征**全部显著**！这听起来很"漂亮"，但其实是**大样本陷阱**的典型表现——21 万条数据下，几乎任何微小的差异都会被检验出统计显著性。这正是本模块标题"统计显著 ≠ 预测力强"要警告你的事。下一节的 5 张图会把这个"陷阱"可视化出来。

---

## 3. 图 5a：p 值对比柱状图

### 3.1 设计思路

p 值本身跨度极大：从 0.05 一直到 1e-300 都有可能。如果直接画柱状图，小 p 值的柱子会"矮到看不见"。解决办法是**对 p 值取负对数**：`-log10(p)`。

### 3.2 完整代码

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 数值特征 p-value
if len(num_df) > 0:
    num_plot = num_df.sort_values('P_Value')
    colors_num = ['#e74c3c' if p < 0.05 else '#3498db' for p in num_plot['P_Value']]
    ax = axes[0]
    bars = ax.bar(range(len(num_plot)), -np.log10(num_plot['P_Value'].values),
                  color=colors_num, edgecolor='white')
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5,
               label=f'α=0.05 (-log10={-np.log10(0.05):.2f})')
    ax.set_xticks(range(len(num_plot)))
    ax.set_xticklabels(num_plot['Feature'].values, rotation=30, ha='right')
    ax.set_ylabel('-log10(p-value)', fontsize=11)
    ax.set_title('Numerical Features: Statistical Significance', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# 分类特征 p-value（逻辑完全相同，省略）
```

### 3.3 关键点详解

#### 3.3.1 `-np.log10(p_value)` 转换——为什么用负对数？

这是本图最核心的设计。我们逐个看几个典型 p 值的转换结果：

| p 值 | -log10(p) | 含义 |
|------|-----------|------|
| 0.05 | 1.30 | 显著性阈值 |
| 0.01 | 2.00 | 较强显著 |
| 0.001 | 3.00 | 强显著 |
| 1e-10 | 10.00 | 极强显著 |
| 1e-50 | 50.00 | 几乎不可能由随机产生 |
| 1e-300 | 300.00 | 极端显著 |

**为什么用负对数？**

1. **把极小的 p 值"放大"**：p=1e-50 在线性尺度下几乎看不见，但取对数后变成 50，柱子立刻清晰可见。
2. **单调变换**：p 值越小 → -log10(p) 越大，"柱子越高 = 越显著"的直觉一致。
3. **跨越多个数量级**：对数尺度天然适合处理"几个数量级跨度"的数据，类似地震震级、pH 值的设计。
4. **阈值可视化**：α=0.05 对应 -log10=1.30，画一条水平线就能直观看出哪些柱子"过了线"。

> **小贴士**：在基因组学（GWAS）中，-log10(p) 是标准做法，常用阈值是 5（即 p < 1e-5）或 7（p < 1e-7）。本教程用的是 1.3（p < 0.05）。

#### 3.3.2 列表推导式生成颜色

```python
colors_num = ['#e74c3c' if p < 0.05 else '#3498db' for p in num_plot['P_Value']]
```

这是一个**列表推导式（list comprehension）**，等价于：

```python
colors_num = []
for p in num_plot['P_Value']:
    if p < 0.05:
        colors_num.append('#e74c3c')   # 红色 = 显著
    else:
        colors_num.append('#3498db')   # 蓝色 = 不显著
```

- `#e74c3c`：十六进制红色（flat-red），表示"显著"。
- `#3498db`：十六进制蓝色（flat-blue），表示"不显著"。
- 列表推导式更简洁，且运行速度比 for 循环 + append 略快。

> **小贴士**：matplotlib 的 `color` 参数接受多种格式：颜色名（`'red'`）、十六进制（`'#e74c3c'`）、RGB 元组（`(0.9, 0.3, 0.2)`）、灰度字符串（`'0.5'`）。十六进制最精确，是数据可视化的主流选择。

#### 3.3.3 `ax.axhline()` 参考线

```python
ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1.5,
           label=f'α=0.05 (-log10={-np.log10(0.05):.2f})')
```

- `axhline`（axis horizontal line）：画一条**水平参考线**，横跨整个 x 轴范围。
- `y=-np.log10(0.05)`：y 坐标 = 1.30，即 α=0.05 在 -log10 尺度下的位置。
- `color='red'`：红色，醒目。
- `linestyle='--'`：虚线，区别于数据线。
- `linewidth=1.5`：线宽 1.5 磅。
- `label=f'α=0.05 (-log10={-np.log10(0.05):.2f})'`：图例文本，用 f-string 动态计算并格式化（保留 2 位小数 → "1.30"）。

> **重要概念**：参考线是数据可视化的"标尺"。没有这条红线，读者只能看到柱子高低，但不知道"多高才算显著"。加上参考线后，**柱子超过红线 = p < 0.05 = 统计显著**，一目了然。

#### 3.3.4 其他细节

- `plt.subplots(1, 2, figsize=(14, 6))`：1 行 2 列子图，左图是数值特征，右图是分类特征。
- `ax.bar(range(len(num_plot)), ...)`：x 轴用 0,1,2,... 的整数位置，再用 `set_xticklabels` 把特征名贴上去。
- `rotation=30, ha='right'`：x 轴标签旋转 30 度，右对齐，避免长特征名重叠。
- `ax.spines['top'].set_visible(False)` / `ax.spines['right'].set_visible(False)`：隐藏顶部和右侧的边框，让图更"干净"（这是数据可视化的常见美学处理）。

### 3.4 结果图

![p值对比图](../img/05_pvalue_comparison.png)

### 3.5 如何解读这张图

- **左图（数值特征）**：4 个数值特征的柱子都远远超过红线，最高的柱子（`Code.of.Morphology`）甚至冲到几十的高度——这意味着它的 p 值小到几乎为零。
- **右图（分类特征）**：18 个分类特征的柱子也都超过红线，但高度差异巨大。最高的几个（如 `Morphology.Description`）柱子极高，而像 `Type.of.Death` 这样的特征柱子虽然过了红线，但高度很矮——**这就是"统计显著但效应量小"的视觉信号**。

> **关键观察**：所有柱子都过了红线（22/22 显著），但柱子高度差异巨大。这暗示我们：**单看 p 值会丢失重要信息**，必须结合效应量。这正是图 5b 要展示的。

---

## 4. 图 5b：效应量对比图

### 4.1 设计思路

图 5a 告诉我们"哪些显著"，但没告诉我们"显著到什么程度"。图 5b 直接画效应量，并按"小/中/大"三级配色，让我们一眼看出哪些特征"显著但没用"（小效应量），哪些"显著且有用"（大效应量）。

### 4.2 完整代码

```python
if len(all_results) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = summary_df.sort_values('Effect_Size', ascending=True)

    colors_effect = ['#e74c3c' if row['Effect_Size'] > 0.3
                     else ('#f39c12' if row['Effect_Size'] > 0.1 else '#3498db')
                     for _, row in plot_df.iterrows()]

    bars = ax.barh(range(len(plot_df)), plot_df['Effect_Size'].values,
                   color=colors_effect, edgecolor='white')
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df['Feature'].values, fontsize=9)
    ax.set_xlabel('Effect Size', fontsize=11)
    ax.set_title('Feature Effect Sizes (Numerical & Categorical)', fontsize=13, fontweight='bold')

    # 效应量参考线
    ax.axvline(x=0.1, color='gray', linestyle=':', alpha=0.7, label='Small (0.1)')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium (0.3)')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Large (0.5)')
    ax.legend(fontsize=9)

    for bar, val in zip(bars, plot_df['Effect_Size'].values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "05b_effect_size_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
```

### 4.3 关键点详解

#### 4.3.1 三色逻辑

```python
colors_effect = ['#e74c3c' if row['Effect_Size'] > 0.3
                 else ('#f39c12' if row['Effect_Size'] > 0.1 else '#3498db')
                 for _, row in plot_df.iterrows()]
```

这是一个**嵌套的三元表达式 + 列表推导式**，等价于：

```python
colors_effect = []
for _, row in plot_df.iterrows():
    if row['Effect_Size'] > 0.3:
        colors_effect.append('#e74c3c')   # 红色 = 中等及以上效应
    elif row['Effect_Size'] > 0.1:
        colors_effect.append('#f39c12')   # 橙色 = 小效应
    else:
        colors_effect.append('#3498db')   # 蓝色 = 极弱效应
```

颜色与效应量分级的对应关系：

| 效应量范围 | 颜色 | 含义 | 对应 Cramér's V 分级 | 对应 Cohen's d 分级 |
|------------|------|------|----------------------|---------------------|
| > 0.3 | `#e74c3c` 红 | 中等及以上效应 | Medium / Large | 接近 Medium |
| 0.1 ~ 0.3 | `#f39c12` 橙 | 小效应 | Small | Small |
| ≤ 0.1 | `#3498db` 蓝 | 极弱效应 | Negligible | Negligible |

> **重要概念**：这里用 0.1 和 0.3 作为分界，是 Cramér's V 的标准分级（Cohen, 1988）。注意 Cohen's d 的标准分级是 0.2/0.5/0.8，但因为本图把两类效应量混在一起画，统一用 0.1/0.3/0.5 的"宽松"分级更直观。

#### 4.3.2 `ax.axvline()` 垂直参考线

```python
ax.axvline(x=0.1, color='gray', linestyle=':', alpha=0.7, label='Small (0.1)')
ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium (0.3)')
ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Large (0.5)')
```

- `axvline`（axis vertical line）：画**垂直参考线**（与 `axhline` 对称）。
- 三条线分别对应小（0.1）、中（0.3）、大（0.5）效应量阈值。
- `alpha=0.7`：透明度 0.7（0=全透明，1=不透明），让参考线不抢戏。
- `linestyle=':'`：点线（更细的虚线），区别于 `--`（虚线）。

> **小贴士**：`axhline` 和 `axvline` 是 matplotlib 中**唯一可以画"无限长"参考线**的函数。普通的 `ax.plot([x1,x2],[y1,y2])` 需要指定端点，而 `axhline`/`axvline` 会自动横跨整个坐标轴。

#### 4.3.3 在柱子末端标注数值

```python
for bar, val in zip(bars, plot_df['Effect_Size'].values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', fontsize=8)
```

- `zip(bars, plot_df['Effect_Size'].values)`：把柱子对象和效应量值一一配对。
- `ax.text(x, y, s, ...)`：在坐标 `(x, y)` 处写文本 `s`。
- `val + 0.005`：x 坐标 = 柱子末端再往右偏 0.005，让文字不挡住柱子。
- `bar.get_y() + bar.get_height() / 2`：y 坐标 = 柱子的垂直中心。
- `f'{val:.4f}'`：保留 4 位小数（如 `0.6959`）。
- `va='center'`：垂直居中对齐。

#### 4.3.4 `barh` 水平柱状图

```python
bars = ax.barh(range(len(plot_df)), plot_df['Effect_Size'].values, ...)
```

- `barh`（bar horizontal）：水平柱状图，柱子从左向右生长。
- 适合**特征名较长**的场景，因为特征名放在 y 轴，水平排列更易读。
- `plot_df.sort_values('Effect_Size', ascending=True)`：升序排序，最小的在底部，最大的在顶部——这样最大的柱子在最上面，视觉焦点集中。

### 4.4 结果图

![效应量对比图](../img/05b_effect_size_comparison.png)

### 4.5 如何解读这张图

- **顶部红色长柱**：`Morphology.Description`（V=0.6959），效应量最大，远超 0.5 的"大效应"红线——这是真正的"强预测特征"。
- **底部蓝色短柱**：`Type.of.Death`（V=0.0171），效应量极小，连 0.1 的"小效应"线都没碰到——尽管它在图 5a 里也"显著"，但实际预测力几乎可以忽略。
- **中间橙色柱**：效应量在 0.1~0.3 之间，有一定预测价值但不强。
- **关键对比**：图 5a 里 22 个特征"全部显著"，但图 5b 里它们的效应量从 0.017 到 0.70 跨越了 40 倍——**这就是"统计显著 ≠ 预测力强"的最直观证据**。

---

## 5. 图 5c：p 值 vs 效应量散点图（最重要的一张图）

### 5.1 为什么这是最重要的一张图？

图 5a 只看 p 值，图 5b 只看效应量，都是"单维度"视角。图 5c 把两个维度**同时画在一张图上**：

- **x 轴 = 效应量**（实际预测力）
- **y 轴 = -log10(p 值)**（统计显著性）

这样每个特征就是一个点，点的位置同时编码了"统计显著"和"实际有用"两个信息。这是数据可视化中经典的"双维度联合分析"思路。

### 5.2 完整代码

```python
if len(all_results) >= 3:
    fig, ax = plt.subplots(figsize=(10, 7))

    for _, row in summary_df.iterrows():
        color = '#e74c3c' if row['Type'] == '数值型' else '#2ecc71'
        size = 80 if row['Significant_0.05'] == 'Yes' else 40
        marker = 'o' if row['Type'] == '数值型' else 's'
        ax.scatter(row['Effect_Size'], -np.log10(row['P_Value']),
                   c=color, s=size, marker=marker, alpha=0.7, edgecolors='gray',
                   linewidths=0.5)

        # 标注特征名
        ax.annotate(row['Feature'],
                    (row['Effect_Size'], -np.log10(row['P_Value'])),
                    fontsize=8, ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points')

    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=1,
               label='α = 0.05')
    ax.axvline(x=0.1, color='gray', linestyle=':', alpha=0.7, label='Small effect (0.1)')
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Medium effect (0.3)')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=8, label='Numerical'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
               markersize=8, label='Categorical'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, alpha=0.5, label='Not significant')
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-3:],
              fontsize=8, loc='lower right')

    ax.set_xlabel('Effect Size', fontsize=11)
    ax.set_ylabel('-log10(p-value)', fontsize=11)
    ax.set_title('P-value vs Effect Size: "Statistical vs Practical" Significance',
                 fontsize=13, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "05c_pvalue_vs_effectsize.png"), dpi=150, bbox_inches='tight')
    plt.close()
```

### 5.3 关键点详解

#### 5.3.1 三重编码：颜色 + 大小 + 形状

每个散点同时被三个视觉通道编码：

| 视觉通道 | 编码变量 | 取值 | 含义 |
|----------|----------|------|------|
| 颜色 (`c`) | 特征类型 | `#e74c3c` 红 / `#2ecc71` 绿 | 数值型 / 分类型 |
| 大小 (`s`) | 是否显著 | 80 / 40 | 显著 / 不显著 |
| 形状 (`marker`) | 特征类型 | `'o'` 圆 / `'s'` 方 | 数值型 / 分类型 |

```python
color = '#e74c3c' if row['Type'] == '数值型' else '#2ecc71'
size = 80 if row['Significant_0.05'] == 'Yes' else 40
marker = 'o' if row['Type'] == '数值型' else 's'
```

> **重要概念**：颜色和形状都编码了"类型"，看起来冗余。但这是**故意的双重编码**——对色盲用户友好（即使看不出红绿，也能看出圆方），也增强了视觉区分度。这是数据可视化的最佳实践之一。

> **小贴士**：`s=80` 的单位是"点的平方"（points²），不是像素。80 对应直径约 9 磅，40 对应约 6.3 磅。如果想精确控制大小，可以算 `s = π * (radius_pt)²`。

#### 5.3.2 `ax.annotate()` 标注特征名

```python
ax.annotate(row['Feature'],
            (row['Effect_Size'], -np.log10(row['P_Value'])),
            fontsize=8, ha='center', va='bottom',
            xytext=(0, 5), textcoords='offset points')
```

- `row['Feature']`：要显示的文本（特征名）。
- `(row['Effect_Size'], -np.log10(row['P_Value']))`：被标注的点的坐标（**数据坐标**）。
- `fontsize=8`：字号 8 磅。
- `ha='center'`（horizontal alignment）：水平居中对齐。
- `va='bottom'`（vertical alignment）：垂直底部对齐——文字在点的上方。
- `xytext=(0, 5)`：文本相对于点的偏移量，`(0, 5)` 表示 x 方向不偏移，y 方向往上偏 5 个单位。
- `textcoords='offset points'`：**关键参数**！指定 `xytext` 的坐标系是"点"（points），而不是"数据坐标"。

> **重要概念**：`textcoords='offset points'` 的含义是"偏移量以点为单位，而不是以数据坐标为单位"。为什么这么设计？因为如果用数据坐标偏移，不同坐标轴尺度下偏移效果会完全不同——在 y 轴跨度 0~300 的图里，`xytext=(0, 5)` 几乎看不见；在 y 轴跨度 0~10 的图里，`xytext=(0, 5)` 又会偏到天上去。用"点"作单位，无论坐标轴尺度如何，文字始终偏移固定的视觉距离（5 磅 ≈ 1.8 毫米），效果稳定可控。

#### 5.3.3 自定义图例：`from matplotlib.lines import Line2D`

```python
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
           markersize=8, label='Numerical'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ecc71',
           markersize=8, label='Categorical'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=8, alpha=0.5, label='Not significant')
]
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-3:],
          fontsize=8, loc='lower right')
```

**为什么需要自定义图例？**

因为我们在 `scatter` 里用了三重编码（颜色+大小+形状），matplotlib 默认生成的图例会很混乱——它不知道你想用哪个通道做图例。所以我们需要**手动构造图例元素**。

**`Line2D` 是什么？**

`Line2D` 是 matplotlib 中表示"线对象"的类。虽然名字叫"Line"，但它可以只画一个 marker（标记），不画线——这正是构造图例图标的标准技巧。

**参数详解**：

- `[0], [0]`：线的 x 和 y 数据，这里只给一个点 `[0]`，因为图例图标只需要一个标记。
- `marker='o'`：标记形状，圆点。
- `color='w'`：线的颜色为白色（`'w'` = white），实际上就是把线"藏起来"，只显示 marker。
- `markerfacecolor='#e74c3c'`：marker 的填充色。
- `markersize=8`：marker 大小。
- `label='Numerical'`：图例文本。

**合并图例**：

```python
ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][-3:],
          fontsize=8, loc='lower right')
```

- `ax.get_legend_handles_labels()`：返回当前 Axes 已有的图例句柄和标签，是一个二元组 `(handles, labels)`。
- `[0]`：取句柄列表。
- `[-3:]`：取最后 3 个（对应 `axhline` 和两个 `axvline` 的参考线图例）。
- `legend_elements + ...`：把自定义的散点图例和参考线图例合并。
- `handles=...`：`legend` 的 `handles` 参数指定图例条目。
- `loc='lower right'`：图例放在右下角（因为左下角通常有点，右下角比较空）。

> **小贴士**：`Line2D` 自定义图例是 matplotlib 进阶技巧。当你遇到"默认图例不好看"或"想合并多个图的图例"时，记住这个套路：构造 `Line2D` 列表 → 传给 `ax.legend(handles=...)`。

#### 5.3.4 参考线划分四个象限

图中有 4 条参考线（1 条水平 + 3 条垂直），把图划分成多个区域。最关键的两个区域是：

| 区域 | 位置 | 含义 | 典型特征 |
|------|------|------|----------|
| **左上角** | 高 -log10(p) + 低效应量 | 统计显著但实际无用 | `Type.of.Death`（p=2.9e-9, V=0.017） |
| **右下角** | 低 -log10(p) + 高效应量 | 实际有用但统计边缘 | （本数据集中没有，因为大样本下大效应量必然 p 极小） |
| **右上角** | 高 -log10(p) + 高效应量 | 既显著又有用（理想特征） | `Morphology.Description`（p=0, V=0.70） |
| **左下角** | 低 -log10(p) + 低效应量 | 既不显著也无用 | （本数据集中没有，因为 22 个全显著） |

### 5.4 结果图

![p值vs效应量散点图](../img/05c_pvalue_vs_effectsize.png)

### 5.5 重点解读：左上角 vs 右下角

这是本模块最关键的解读环节。

#### 左上角：统计显著但实际无用

**典型代表**：`Type.of.Death`（p=2.9e-9, V=0.017）

- **y 轴很高**：-log10(2.9e-9) ≈ 8.5，柱子远超 α=0.05 红线（1.3）。
- **x 轴很低**：V=0.017，远低于"小效应"灰线（0.1）。
- **含义**：这个特征与目标变量"确实有关联"（p 值很小，几乎不可能是随机产生），但关联**强度极弱**（V=0.017 几乎可以忽略）。
- **为什么会出现这种情况？** 因为样本量太大（21 万），即使 0.017 的弱关联也能被检测出"显著"。这就是**大样本陷阱**。
- **机器学习启示**：把 `Type.of.Death` 加入模型，预测力提升几乎可以忽略。它"显著"但不"有用"。

#### 右下角：实际有用但统计边缘

**典型代表**：（本数据集中没有，但理论上存在）

- **y 轴很低**：-log10(p) 接近 1.3 甚至更低。
- **x 轴很高**：效应量 > 0.3 甚至 > 0.5。
- **含义**：这个特征与目标变量的关联**强度很大**，但因为样本量小或方差大，p 值没到显著阈值。
- **机器学习启示**：这种特征往往**值得纳入模型**——效应量大意味着预测力强，p 值不显著可能只是样本量不够。在交叉验证中，如果它稳定提升模型表现，就应该保留。

#### 右上角：理想特征

**典型代表**：`Morphology.Description`（p=0, V=0.70）

- **y 轴极高**：p 值小到机器精度下变成 0，-log10(0) 理论上是无穷大（实际代码会做保护处理）。
- **x 轴极高**：V=0.70，远超"大效应"红线（0.5）。
- **含义**：既统计显著，又实际有用——这是特征选择中的"金矿"。
- **机器学习启示**：这种特征应该**优先纳入模型**。

> **核心论点**：
>
> **统计显著 ≠ 预测力强**。p 值回答的是"有无差异"，效应量回答的是"差异有多大"。在机器学习场景下，后者比前者重要得多。
>
> 读这张图时，**先看 x 轴（效应量），再看 y 轴（p 值）**。x 轴大的特征才是真正有预测力的特征；y 轴大但 x 轴小的特征，只是被大样本"撑起来"的虚假显著。

---

## 6. 图 5d：Top 数值特征箱线图

### 6.1 设计思路

前面三张图都是"宏观"视角——所有特征一起看。图 5d 和图 5e 转向"微观"视角——挑出最有区分力的几个特征，**直接看它们在 VIVO（存活）和 MORTO（死亡）两组间的分布差异**。

### 6.2 完整代码

```python
if len(num_df) > 0:
    top_num = num_df.nsmallest(min(4, len(num_df)), 'P_Value')
    fig, axes = plt.subplots(1, len(top_num), figsize=(6 * len(top_num), 5))
    if len(top_num) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top_num.iterrows()):
        col = row['Feature']
        plot_data = df_model[[col, 'target']].dropna()
        if len(plot_data) > 50000:
            plot_data = plot_data.sample(50000, random_state=42)

        vivo_data = plot_data.loc[plot_data['target'] == 1, col]
        morto_data = plot_data.loc[plot_data['target'] == 0, col]

        bp = ax.boxplot([vivo_data.values, morto_data.values],
                        labels=['VIVO', 'MORTO'],
                        patch_artist=True, widths=0.4)
        bp['boxes'][0].set_facecolor('#2ecc71')
        bp['boxes'][1].set_facecolor('#e74c3c')
        for w in bp['whiskers']:
            w.set_color('gray')
        for c in bp['caps']:
            c.set_color('gray')
        for m in bp['medians']:
            m.set_color('black')
            m.set_linewidth(2)

        ax.set_title(f'{col}\n(p={row["P_Value"]:.2e}, d={row["Effect_Size"]:.3f})',
                     fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Top Discriminative Numerical Features by Target Group',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "05d_top_numerical_boxplots.png"), dpi=150, bbox_inches='tight')
    plt.close()
```

### 6.3 关键点详解

#### 6.3.1 `nsmallest` 取 Top 特征

```python
top_num = num_df.nsmallest(min(4, len(num_df)), 'P_Value')
```

- `nsmallest(n, columns)`：pandas 方法，返回 `columns` 列最小的 `n` 行。等价于 `df.sort_values(columns).head(n)`，但效率更高（用堆算法，不需要全排序）。
- `min(4, len(num_df))`：**保护性写法**。如果数值特征少于 4 个（比如只有 3 个），就取 `len(num_df)`，避免 `nsmallest(4, ...)` 在只有 3 行时出错。
- `'P_Value'`：按 p 值取最小的，即"最显著"的 4 个。

> **小贴士**：`min(4, len(num_df))` 这种写法在数据量不确定时非常重要。如果直接写 `nsmallest(4, ...)`，当 `num_df` 只有 3 行时虽然不会报错（pandas 会返回 3 行），但写 `min` 更明确地表达了"最多 4 个"的意图，代码可读性更好。

#### 6.3.2 大数据集采样

```python
plot_data = df_model[[col, 'target']].dropna()
if len(plot_data) > 50000:
    plot_data = plot_data.sample(50000, random_state=42)
```

- 21 万样本全画箱线图会非常慢（matplotlib 要画 21 万个 flier 点）。
- 采样到 5 万：箱线图的分位数估计几乎不变（5 万样本已经足够精确），但绘图速度提升数倍。
- `random_state=42`：固定随机种子，保证每次运行结果一致（可复现）。

#### 6.3.3 按目标分组

```python
vivo_data = plot_data.loc[plot_data['target'] == 1, col]
morto_data = plot_data.loc[plot_data['target'] == 0, col]
```

- `plot_data['target'] == 1`：布尔 Series，标记哪些行是 VIVO（存活）。
- `.loc[..., col]`：用布尔索引选行，再选 `col` 列。
- 注意：这里 `target=1` 是 VIVO，`target=0` 是 MORTO（这是模块 0 中定义的）。

#### 6.3.4 箱线图绘制与美化

```python
bp = ax.boxplot([vivo_data.values, morto_data.values],
                labels=['VIVO', 'MORTO'],
                patch_artist=True, widths=0.4)
bp['boxes'][0].set_facecolor('#2ecc71')
bp['boxes'][1].set_facecolor('#e74c3c')
```

- `ax.boxplot(data, labels, patch_artist, widths)`：
  - `data`：列表 of 数组，每个数组画一个箱体。
  - `labels`：每个箱体的 x 轴标签。
  - `patch_artist=True`：**关键参数**！开启后箱体变成 `Patch` 对象，可以用 `set_facecolor` 填充颜色。默认 `False` 时箱体只有边框，无法填色。
  - `widths=0.4`：箱体宽度（在 0~1 的 x 轴范围内）。
- `bp`：返回值是一个字典，包含 `boxes`（箱体）、`whiskers`（须线）、`caps`（端点）、`medians`（中位线）、`fliers`（离群点）。
- `bp['boxes'][0].set_facecolor('#2ecc71')`：第一个箱体（VIVO）填绿色。
- `bp['boxes'][1].set_facecolor('#e74c3c')`：第二个箱体（MORTO）填红色。

> **重要概念**：matplotlib 的 `boxplot` 返回值是一个**字典**，每个 key 对应一类图形元素。这种设计让我们可以**精细控制每个部分的颜色和样式**——比如把所有须线设为灰色、中位线设为黑色加粗。这是 matplotlib 比 seaborn 更底层、更灵活的体现。

#### 6.3.5 标题中嵌入统计量

```python
ax.set_title(f'{col}\n(p={row["P_Value"]:.2e}, d={row["Effect_Size"]:.3f})',
             fontsize=11, fontweight='bold')
```

- f-string 嵌入特征名、p 值、效应量。
- `:.2e`：科学计数法保留 2 位小数（如 `2.91e-09`）。
- `:.3f`：浮点数保留 3 位小数（如 `0.123`）。
- `\n`：换行，让标题分两行——第一行是特征名，第二行是统计量。

### 6.4 结果图

![Top数值特征箱线图](../img/05d_top_numerical_boxplots.png)

### 6.5 如何解读这张图

- 每个子图是一个数值特征，按 VIVO/MORTO 两组画箱线图。
- **如果两组箱体高度重叠** → 这个特征区分力弱（效应量小）。
- **如果两组箱体明显分离** → 这个特征区分力强（效应量大）。
- 标题里的 `d=` 是 Cohen's d，可以直接和箱体分离程度对应——d 越大，分离越明显。
- 例如 `Age` 特征：VIVO 组的中位数通常比 MORTO 组低（年轻患者存活率更高），箱体会有明显错位。

> **小贴士**：箱线图是"分布对比"最经典的工具。但它的局限是只展示分位数，看不到分布形状。如果想看更详细的分布，可以用小提琴图（violin plot）或叠加 KDE（核密度估计）。

---

## 7. 图 5e：Top 分类特征堆叠柱状图

### 7.1 设计思路

分类特征不能用箱线图（没有连续数值）。最直观的做法是：**对每个类别，看 VIVO/MORTO 的占比**。如果某个类别下 VIVO 占比明显高于其他类别，说明这个分类特征有区分力。

### 7.2 完整代码

```python
if len(cat_df) > 0:
    top_cat = cat_df.nsmallest(min(4, len(cat_df)), 'P_Value')
    fig, axes = plt.subplots(1, len(top_cat), figsize=(5 * len(top_cat), 5))
    if len(top_cat) == 1:
        axes = [axes]

    for ax, (_, row) in zip(axes, top_cat.iterrows()):
        col = row['Feature']
        data = df_model[[col, 'target']].dropna()
        # 取 Top 8 类别
        top_categories = data[col].value_counts().head(8).index
        data_subset = data[data[col].isin(top_categories)]

        ct = pd.crosstab(data_subset[col], data_subset['target'], normalize='index')
        ct.columns = ['MORTO (Dead)', 'VIVO (Alive)']
        ct.sort_values('VIVO (Alive)', ascending=True, inplace=True)

        ct.plot(kind='barh', stacked=True, ax=ax,
                color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.7)
        ax.set_title(f'{col}\n(p={row["P_Value"]:.2e}, V={row["Cramér_V"]:.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Proportion')
        ax.set_ylabel('')
        ax.legend(fontsize=7, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 在柱上标注存活率
        for i, (idx, r) in enumerate(ct.iterrows()):
            vivo_pct = r['VIVO (Alive)'] * 100
            if vivo_pct > 3:
                ax.text(r['MORTO (Dead)'] + r['VIVO (Alive)'] / 2, i,
                        f'{vivo_pct:.1f}%', ha='center', va='center',
                        fontsize=7, fontweight='bold', color='white')

    plt.suptitle('Top Discriminative Categorical Features: Survival Rate by Category',
                 fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "05e_top_categorical_stackedbar.png"), dpi=150, bbox_inches='tight')
    plt.close()
```

### 7.3 关键点详解

#### 7.3.1 `pd.crosstab(..., normalize='index')` 按行归一化

```python
ct = pd.crosstab(data_subset[col], data_subset['target'], normalize='index')
```

**`pd.crosstab` 是什么？**

`pd.crosstab(index, columns)` 用来计算两个变量的**交叉频数表**（contingency table）。例如：

| | target=0 (MORTO) | target=1 (VIVO) |
|---|---|---|
| Category A | 100 | 900 |
| Category B | 500 | 500 |

**`normalize='index'` 的含义**：

`normalize` 参数控制归一化方式：

| 取值 | 含义 | 结果 |
|------|------|------|
| `False`（默认） | 不归一化 | 原始频数 |
| `'index'` 或 `0` | **按行归一化** | 每行之和 = 1 |
| `'columns'` 或 `1` | 按列归一化 | 每列之和 = 1 |
| `True` 或 `'all'` | 全局归一化 | 所有格子之和 = 1 |

按行归一化后：

| | target=0 (MORTO) | target=1 (VIVO) | 行和 |
|---|---|---|---|
| Category A | 0.10 | 0.90 | 1.0 |
| Category B | 0.50 | 0.50 | 1.0 |

> **重要概念**：按行归一化的语义是"**在每个类别内部**，VIVO/MORTO 的占比"。这正是我们想要的——回答"Category A 的患者存活率是多少？"答案是 90%。如果不归一化，原始频数受类别样本量影响，无法直接比较。

#### 7.3.2 取 Top 8 类别

```python
top_categories = data[col].value_counts().head(8).index
data_subset = data[data[col].isin(top_categories)]
```

- `value_counts()`：统计每个类别的频数，按频数降序排列。
- `.head(8)`：取前 8 个最常见的类别。
- `.index`：取类别名（index 是类别名，values 是频数）。
- `data[col].isin(top_categories)`：布尔索引，标记哪些行的类别在 Top 8 内。

> **小贴士**：为什么要取 Top 8？因为某些分类特征可能有几十甚至上百个类别（如 `Morphology.Description`），全画在一张图上会拥挤不堪。取 Top 8 是可视化性能和代表性的平衡。

#### 7.3.3 `ct.plot(kind='barh', stacked=True)` 水平堆叠柱状图

```python
ct.plot(kind='barh', stacked=True, ax=ax,
        color=['#e74c3c', '#2ecc71'], edgecolor='white', width=0.7)
```

- `ct.plot(...)`：pandas DataFrame 自带的绘图方法，底层调用 matplotlib。
- `kind='barh'`：水平柱状图（bar horizontal）。
- `stacked=True`：**堆叠**——每行的两列（MORTO + VIVO）堆在一根柱子上，柱子总长度 = 1（因为按行归一化了）。
- `color=['#e74c3c', '#2ecc71']`：第一列（MORTO）红色，第二列（VIVO）绿色。
- `edgecolor='white'`：柱子边缘白色，让相邻柱子有视觉分隔。
- `width=0.7`：柱子宽度（0~1 之间）。

> **重要概念**：堆叠柱状图 + 按行归一化 = "占比图"。每根柱子总长 1，红色段长度 = 死亡占比，绿色段长度 = 存活占比。一眼就能看出"哪个类别存活率高"。

#### 7.3.4 排序让图更易读

```python
ct.sort_values('VIVO (Alive)', ascending=True, inplace=True)
```

- 按 VIVO 占比升序排序，存活率低的类别在底部，存活率高的在顶部。
- 这样图从下到上就是"存活率从低到高"，视觉上有"递增"的节奏感，更容易读。

#### 7.3.5 在柱上标注存活率

```python
for i, (idx, r) in enumerate(ct.iterrows()):
    vivo_pct = r['VIVO (Alive)'] * 100
    if vivo_pct > 3:
        ax.text(r['MORTO (Dead)'] + r['VIVO (Alive)'] / 2, i,
                f'{vivo_pct:.1f}%', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')
```

- `enumerate(ct.iterrows())`：同时获取行号 `i` 和行内容 `(idx, r)`。
- `vivo_pct = r['VIVO (Alive)'] * 100`：把占比转成百分比（0~100）。
- `if vivo_pct > 3`：**保护性判断**——如果 VIVO 占比小于 3%，文字会挤在柱子边缘看不清，干脆不标。
- `ax.text(x, y, s, ...)`：
  - `x = r['MORTO (Dead)'] + r['VIVO (Alive)'] / 2`：x 坐标 = MORTO 段长度 + VIVO 段一半，即 VIVO 段的中心。
  - `y = i`：y 坐标 = 第 i 根柱子。
  - `s = f'{vivo_pct:.1f}%'`：文本，如 `"85.3%"`。
  - `ha='center', va='center'`：水平+垂直居中对齐。
  - `color='white'`：白色文字（写在绿色 VIVO 段上，对比度高）。

### 7.4 结果图

![Top分类特征堆叠图](../img/05e_top_categorical_stackedbar.png)

### 7.5 如何解读这张图

- 每根柱子是一个类别，总长 1（100%）。
- **绿色段越长 = 存活率越高**，红色段越长 = 死亡率越高。
- 如果所有柱子的绿色段长度差不多 → 这个分类特征区分力弱（V 值小）。
- 如果绿色段长度差异巨大（有的接近 0，有的接近 1）→ 这个分类特征区分力强（V 值大）。
- 例如 `Morphology.Description`（V=0.70）：不同形态学描述的存活率差异极大，有些类型几乎全部死亡，有些类型几乎全部存活——这就是 V=0.70 的视觉表现。

---

## 8. 核心讨论：统计显著 ≠ 预测力强

> **统计显著 ≠ 预测力强**。
>
> 这是本模块最核心的论点，也是从"统计学思维"转向"机器学习思维"的关键认知。p 值显著只说明"差异存在"，不说明"差异大到可用于预测"。

### 8.1 大样本陷阱

本数据集有 **21 万条有标签数据**。在这么大的样本下，统计检验会变得"过于敏感"——即使极小的差异也会被检测为显著。

**数学直觉**：

以 t 检验为例，检验统计量 t = (均值差) / (标准误)，而标准误 = 标准差 / √n。当 n=210,000 时，√n ≈ 458，标准误被缩小到原来的 1/458。这意味着**即使均值差只有 0.01，t 值也可能很大，p 值可能极小**。

具体到本案例：

- `Type.of.Death` 的 Cramér's V = 0.0171，效应量极小。
- 但卡方统计量 = n × V² × (列数-1) ≈ 210000 × 0.0171² × 1 ≈ 61.5。
- 卡方分布下，统计量 61.5 对应的 p 值 ≈ 2.9e-9——极其显著！

**结论**：p=2.9e-9 的"极其显著"完全来自大样本，而不是强关联。这就是"大样本陷阱"。

### 8.2 效应量才是预测力的锚

| 指标 | 回答的问题 | 受样本量影响 | 对 ML 的指导意义 |
|------|------------|--------------|-------------------|
| p 值 | "有无差异？" | **极大**（n 越大越显著） | 弱 |
| 效应量 | "差异有多大？" | **很小**（与样本量基本无关） | 强 |

> **重要概念**：效应量是"标准化"的差异大小，设计上就**尽量不受样本量影响**。Cohen's d 把均值差除以标准差，Cramér's V 把卡方统计量除以 n 再开方——都是为了消除样本量的影响，让"差异大小"在不同研究间可比较。

### 8.3 统计检验 vs 机器学习：关心的问题不同

| 维度 | 统计检验 | 机器学习 |
|------|----------|----------|
| 核心问题 | "有无差异？"（存在性） | "能否泛化？"（预测性） |
| 主要指标 | p 值 | AUC、F1、准确率 |
| 样本量影响 | 极大（n 越大越显著） | 适中（n 越大估计越稳，但不会"虚假显著"） |
| 多重共线性 | 不关心 | 关心（相关特征会互相削弱） |
| 交互作用 | 通常不考虑 | 模型自动学习 |
| 结论形式 | "显著/不显著"（二值） | "提升多少 AUC"（连续） |

### 8.4 案例对比：Type.of.Death vs Morphology.Description

这是本模块最经典的对比，用表格一目了然：

| 特征 | p 值 | -log10(p) | Cramér's V | 效应量分级 | ML 价值 |
|------|------|-----------|------------|------------|---------|
| `Type.of.Death` | 2.9e-9 | 8.5 | 0.0171 | 极弱（<0.1） | 几乎无 |
| `Morphology.Description` | 0.0（机器精度下） | ∞ | 0.6959 | 大（>0.5） | 极高 |

**解读**：

- 两者都"统计显著"（p < 0.05），甚至 `Type.of.Death` 的 p 值小到 2.9e-9，看起来"更显著"。
- 但 `Type.of.Death` 的效应量只有 0.017，意味着它和目标变量的关联强度极弱——加入模型几乎不会提升预测力。
- `Morphology.Description` 的效应量 0.70，意味着它和目标变量强关联——加入模型大概率显著提升预测力。

> **核心论点**：
>
> 如果只看 p 值，你会把 `Type.of.Death` 和 `Morphology.Description` 都选进模型。但前者是"噪声"，后者是"信号"。**效应量是区分信号和噪声的锚**。

### 8.5 实践建议

1. **优先看效应量，再看 p 值**：特征选择时，先按效应量排序，再在效应量足够大的特征中筛 p 值显著的。
2. **用交叉验证做最终裁决**：统计检验只是"初筛"，真正的特征选择应该用交叉验证评估"加入这个特征后模型表现提升多少"。
3. **警惕"全部显著"**：如果所有特征都显著，先检查样本量——大样本下"全部显著"是常态，不代表所有特征都有用。
4. **报告效应量**：在论文或报告中，**必须同时报告 p 值和效应量**，只报 p 值是不专业的。

---

## 9. 实际运行结果汇总

### 9.1 总体结果

| 指标 | 数值 |
|------|------|
| 共分析特征数 | 22 |
| 数值型特征 | 4 |
| 分类型特征 | 18 |
| α=0.05 水平显著 | 22（全部） |
| Bonferroni 校正后显著 | 22（全部） |
| 效应量最大 | `Morphology.Description`（V=0.6959） |
| 效应量最小 | `Type.of.Death`（V=0.0171） |
| 效应量跨度 | 0.0171 ~ 0.6959（约 40 倍） |

### 9.2 关键观察

1. **22 个特征全部显著**——这是大样本（21 万）的典型表现，**不能**作为"所有特征都有用"的证据。
2. **效应量差异巨大**——从 0.017 到 0.70 跨越 40 倍，说明"显著"和"有用"是两回事。
3. **Bonferroni 校正也没筛掉任何特征**——因为 22 个特征时 Bonferroni 阈值是 0.05/22 ≈ 0.0023，而所有 p 值都远小于这个阈值。这说明**单纯靠多重检验校正无法解决大样本陷阱**，必须结合效应量。

### 9.3 效应量分级分布

按 Cramér's V / Cohen's d 的标准分级：

| 分级 | 阈值 | 特征数（约） | 典型代表 |
|------|------|--------------|----------|
| 大效应 | > 0.5 | 少数 | `Morphology.Description` (0.70) |
| 中效应 | 0.3 ~ 0.5 | 部分 | 部分形态学/分期相关特征 |
| 小效应 | 0.1 ~ 0.3 | 部分 | `Age`、`year` 等 |
| 极弱效应 | < 0.1 | 少数 | `Type.of.Death` (0.017) |

> **关键结论**：虽然 22 个特征"全部显著"，但真正有"大效应"的特征是少数。机器学习建模时，应该**优先选大效应特征，谨慎选小效应特征，果断剔除极弱效应特征**——即使它们 p 值显著。

---

## 10. 小贴士

> **小贴士 1**：`-log10(p)` 是统计可视化的"标准操作"。任何涉及 p 值跨数量级比较的场景（GWAS、差异表达分析、A/B 测试）都应该用这个变换。

> **小贴士 2**：列表推导式 `['#e74c3c' if p < 0.05 else '#3498db' for p in ...]` 是 matplotlib 配色的常用套路。它比 for 循环 + append 更简洁，且速度略快。但注意**不要在推导式里写太复杂的逻辑**，否则可读性会变差。

> **小贴士 3**：`axhline` 和 `axvline` 是画参考线的最佳选择，因为它们自动横跨整个坐标轴。用 `ax.plot([x1,x2],[y1,y2])` 画参考线需要手动算端点，容易出错。

> **小贴士 4**：`textcoords='offset points'` 是标注文字位置的"黄金参数"。它让文字偏移以"点"为单位，不受坐标轴尺度影响，效果稳定可控。任何需要"在数据点旁边写字"的场景都应该用它。

> **小贴士 5**：`Line2D` 自定义图例是 matplotlib 进阶技巧。当你遇到"默认图例不好看"或"想合并多个图的图例"时，记住这个套路：构造 `Line2D` 列表 → 传给 `ax.legend(handles=...)`。

> **小贴士 6**：`pd.crosstab(..., normalize='index')` 是分类变量占比可视化的"前置步骤"。按行归一化后，每行之和=1，可以直接读"每个类别内的存活率"。

> **小贴士 7**：大样本下"全部显著"是常态，不是异常。遇到这种情况不要慌，**转向效应量分析**，用 0.1/0.3/0.5 的分级把"真信号"和"虚假显著"分开。

---

## 11. 常见问题（FAQ）

### Q1：为什么要把 p 值取 -log10？直接画 p 值不行吗？

**A**：直接画 p 值会遇到两个问题：
1. **小 p 值看不见**：p=1e-50 在线性尺度下几乎为 0，柱子矮到看不见。
2. **跨度太大**：p 值从 0.05 到 1e-300 跨越 300 个数量级，线性尺度下无法同时展示。

`-log10(p)` 把"几个数量级"压缩成"几十的数值"，柱子高度差异可见，且"柱子越高 = 越显著"的直觉一致。

### Q2：为什么 `Type.of.Death` 的 p 值那么小（2.9e-9）但效应量那么小（0.017）？

**A**：这是**大样本陷阱**的典型表现。卡方统计量 = n × V² × (列数-1)，当 n=21 万时，即使 V=0.017，卡方统计量也能达到 60+，对应 p 值极小。**p 值反映的是"样本量 × 效应量²"，不只是效应量本身**。所以大样本下，p 值会被"放大"，而效应量保持稳定。

### Q3：`normalize='index'` 和 `normalize='columns'` 有什么区别？

**A**：
- `normalize='index'`（或 `0`）：**按行归一化**，每行之和=1。语义是"在每个行类别内，列类别的占比"。本案例中行是特征类别，列是 VIVO/MORTO，按行归一化就是"每个特征类别内的存活率"。
- `normalize='columns'`（或 `1`）：**按列归一化**，每列之和=1。语义是"在每个列类别内，行类别的占比"。本案例中就是"在 VIVO 组内，各特征类别的占比"。

选哪个取决于你想回答什么问题。本案例想回答"每个特征类别下存活率是多少"，所以用 `normalize='index'`。

### Q4：为什么要在散点图里同时用颜色和形状编码"类型"？不是冗余吗？

**A**：是**故意的双重编码**，原因有二：
1. **色盲友好**：约 8% 的男性有某种色盲，红绿色盲最常见。即使看不出红绿，也能通过圆/方区分类型。
2. **增强视觉区分度**：单一通道（只颜色或只形状）在小尺寸打印或投影时容易混淆，双重编码更稳健。

这是数据可视化的最佳实践（参考 Wilkinson《The Grammar of Graphics》）。

### Q5：`xytext=(0, 5)` 里的 5 是什么单位？为什么不用数据坐标？

**A**：5 是"点"（points，1 点 = 1/72 英寸 ≈ 0.35 毫米）。用"点"而不是数据坐标的原因是：**数据坐标的偏移量会随坐标轴尺度变化**。在 y 轴跨度 0~300 的图里，偏移 5 个数据单位几乎看不见；在 y 轴跨度 0~10 的图里，偏移 5 个数据单位又偏到天上去。用"点"作单位，无论坐标轴尺度如何，文字始终偏移固定的视觉距离（5 磅 ≈ 1.8 毫米），效果稳定可控。这就是 `textcoords='offset points'` 的作用。

### Q6：为什么 `nsmallest(min(4, len(num_df)), 'P_Value')` 要写 `min`？直接 `nsmallest(4, ...)` 不行吗？

**A**：直接 `nsmallest(4, ...)` 在 `num_df` 只有 3 行时不会报错（pandas 会返回 3 行），但写 `min` 有两个好处：
1. **意图明确**：`min(4, len(num_df))` 明确表达"最多 4 个"，代码可读性更好。
2. **防御性编程**：如果未来 pandas 行为变化，或者代码迁移到其他库（如 polars），`min` 写法更安全。

这是"防御性编程"的体现——用一点点冗余换取代码的健壮性。

### Q7：Bonferroni 校正后还是 22 个全显著，是不是校正没用？

**A**：不是"校正没用"，而是"样本量太大，校正也救不了"。Bonferroni 校正把阈值从 0.05 降到 0.05/22 ≈ 0.0023，但所有 p 值都远小于这个阈值（最小的也有 2.9e-9）。这说明**单纯靠多重检验校正无法解决大样本陷阱**——校正只是把"显著"的门槛提高了几倍，但大样本下 p 值可以小到 1e-300，几倍的门槛毫无意义。**必须结合效应量**。

### Q8：箱线图里 VIVO 和 MORTO 的箱体高度重叠，但 p 值还是很小，为什么？

**A**：还是大样本陷阱。箱体重叠说明"分布相似"（效应量小），但 t 检验/Mann-Whitney 检验关心的是"均值/中位数有无差异"，即使差异极小，大样本下也能检测出"显著"。**箱体重叠程度 ≈ 效应量大小**，比 p 值更直观。

---

## 12. 本模块小结

### 本模块学到的知识点

#### 理论知识

1. **汇总逻辑**：把不同来源（数值表 `num_df` + 分类表 `cat_df`）、不同列名（`Effect_Size` vs `Cramér_V`）的结果合并到统一格式 `summary_df`，需要用列表 of 字典作为中间结构。
2. **`-log10(p)` 转换**：把极小的 p 值"放大"到可视觉比较的尺度，是统计可视化的标准操作。p=0.05→1.3, p=0.001→3, p=1e-10→10。
3. **效应量分级**：Cramér's V 用 0.1/0.3/0.5 分小/中/大；Cohen's d 用 0.2/0.5/0.8。本模块统一用 0.1/0.3/0.5 的"宽松"分级。
4. **大样本陷阱**：21 万样本下，即使 V=0.017 的弱关联也会 p<0.05。p 值反映"样本量 × 效应量²"，不只是效应量。
5. **统计显著 ≠ 预测力强**：p 值回答"有无差异"，效应量回答"差异有多大"。机器学习关心后者。

#### 代码技能

6. **pandas 汇总**：
   - 列表 of 字典 → `pd.DataFrame(all_results)` 转表
   - `.sort_values('P_Value')` 按 p 值排序
   - `nsmallest(n, col)` 取最小的 n 行（比 sort_values + head 更高效）
   - `pd.crosstab(..., normalize='index')` 按行归一化交叉表

7. **matplotlib 可视化**：
   - `plt.subplots(1, 2, figsize=...)` 多子图
   - `ax.bar()` / `ax.barh()` 竖直/水平柱状图
   - `ax.scatter(c=, s=, marker=)` 三重编码散点图
   - `ax.axhline()` / `ax.axvline()` 水平/垂直参考线
   - `ax.annotate(text, xy, xytext, textcoords='offset points')` 标注文字
   - `ax.boxplot(data, patch_artist=True)` 填色箱线图
   - `ct.plot(kind='barh', stacked=True)` 堆叠柱状图

8. **进阶技巧**：
   - 列表推导式生成颜色列表 `['#e74c3c' if ... else '#3498db' for ...]`
   - `from matplotlib.lines import Line2D` 自定义图例
   - `bp['boxes'][0].set_facecolor(...)` 精细控制箱线图样式
   - `ax.text()` 在柱子上标注数值

#### 思维方法

9. **双维度联合分析**：图 5c 把 p 值和效应量画在同一张图上，划分"左上/右下/右上/左下"四个象限，分别对应"显著但无用/有用但边缘/既显著又有用/既不显著也无用"四种特征。
10. **效应量优先**：特征选择时先看效应量，再看 p 值。效应量是预测力的锚，p 值只是显著性的标签。
11. **批判性看待"全部显著"**：大样本下"全部显著"是常态，不代表所有特征都有用。必须结合效应量分级。

---

## 13. 整个案例教程 2 的总结

### 13.1 案例教程 2 的整体脉络

案例教程 2「统计学分析」共 4 个模块，构成了一条完整的"统计检验 → 效应量 → 汇总可视化"的分析链：

| 模块 | 主题 | 核心问题 |
|------|------|----------|
| 模块 1 | 数据准备与目标变量 | 数据长什么样？目标变量如何编码？ |
| 模块 2 | 数值特征统计检验 | 数值特征与目标变量有关联吗？用 t 检验还是 Mann-Whitney U？ |
| 模块 3 | 分类特征统计检验 | 分类特征与目标变量有关联吗？卡方检验 + Cramér's V |
| 模块 4 | 汇总与可视化 | 22 个特征全显著，但哪些真正有用？统计显著 ≠ 预测力强 |

### 13.2 核心方法论回顾

#### 第一步：选择正确的检验方法

| 数据类型 | 分布特征 | 推荐检验 | 效应量 |
|----------|----------|----------|--------|
| 数值 vs 分类 | 正态分布 | 独立样本 t 检验 | Cohen's d |
| 数值 vs 分类 | 非正态 | Mann-Whitney U 检验 | rank-biserial correlation |
| 分类 vs 分类 | 任意 | 卡方检验 | Cramér's V |

#### 第二步：同时报告 p 值和效应量

- **p 值**：回答"有无差异"（存在性）
- **效应量**：回答"差异有多大"（实用性）
- **两者缺一不可**：只报 p 值不专业，只报效应量无法判断统计显著性。

#### 第三步：多重检验校正

- 22 个特征同时检验 → 多重比较问题 → 假阳性率膨胀
- Bonferroni 校正：阈值 = 0.05 / 特征数
- 但校正不能解决"大样本陷阱"——必须结合效应量

#### 第四步：可视化汇总

- 图 5a：p 值对比（-log10 尺度）
- 图 5b：效应量对比（三色分级）
- 图 5c：p 值 vs 效应量散点图（双维度联合分析，最重要）
- 图 5d：Top 数值特征箱线图（分布对比）
- 图 5e：Top 分类特征堆叠柱状图（占比对比）

### 13.3 最核心的认知转变

> **统计显著 ≠ 预测力强**。

这是整个案例教程 2 想传递的最核心认知。具体表现为三点：

1. **大样本下"全部显著"是常态**：21 万样本下，22 个特征全部 p<0.05，甚至 Bonferroni 校正后也全部显著。但这不代表所有特征都有用。
2. **效应量才是预测力的锚**：效应量从 0.017（`Type.of.Death`）到 0.70（`Morphology.Description`）跨越 40 倍，这才是特征"有用程度"的真实排序。
3. **统计检验是"筛选器"，不是"判决书"**：统计检验帮我们快速排除"明显无关"的特征，但最终的特征选择应该用交叉验证评估"加入这个特征后模型表现提升多少"。

### 13.4 从统计学到机器学习的桥梁

案例教程 2 是整个 ML 教程体系中"从统计到机器学习"的过渡：

- **上游**（案例教程 1：EDA）：理解数据本身——缺失值、分布、离群值。
- **中游**（案例教程 2：统计分析）：量化特征与目标的关联——p 值、效应量。
- **下游**（后续案例教程：机器学习建模）：用模型预测——特征选择、模型训练、评估。

案例教程 2 的核心价值在于：**它告诉你统计检验能做什么、不能做什么**。统计检验能告诉你"哪些特征有关联"，但不能告诉你"哪些特征能提升模型预测力"。后者需要机器学习方法（交叉验证、特征重要性、SHAP 值等）来回答。

### 13.5 给学习者的建议

1. **永远同时报告 p 值和效应量**：这是统计规范，也是科学严谨性的体现。
2. **遇到"全部显著"先检查样本量**：大样本下"全部显著"是常态，不要被 p 值迷惑。
3. **用图 5c 的"双维度散点图"做特征初筛**：右上角的特征优先选，左上角的特征谨慎选，左下角的特征果断剔除。
4. **统计检验只是起点，不是终点**：真正的特征选择要用交叉验证 + 模型评估做最终裁决。
5. **养成"看效应量"的习惯**：读论文时，如果作者只报 p 值不报效应量，要警惕——可能效应量很小，作者在"用 p 值掩盖"。

---

> **结语**：完成案例教程 2 后，你应该能够独立完成一个完整的"统计检验 + 效应量 + 可视化汇总"分析流程，并理解为什么"统计显著 ≠ 预测力强"。这是从"统计学思维"转向"机器学习思维"的关键一步。在后续的机器学习教程中，我们将看到这些统计显著的变量在模型中的实际表现，并学习更高级的特征选择方法（Lasso、Boruta、SHAP 等）。
