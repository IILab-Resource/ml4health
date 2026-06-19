# 模块 1：数据概况

> 本模块是 EDA（探索性数据分析）的第一步实质性分析。在模块 0 中我们已经加载并清洗了数据，现在要回答几个最基本的问题：数据集有多大？有多少特征？标签分布如何？数据是否平衡？这些问题的答案将决定后续所有建模决策。 

***

## 学习目标

学完本模块后，你将能够：

1. **理解数据框形状属性**：掌握 `df.shape` 的返回值，区分 `df.shape[0]` 和 `df.shape[1]`。
2. **熟练使用** **`value_counts()`**：理解 `normalize=True` 参数，能够同时获取频数和比例。
3. **掌握字典取值技巧**：理解 `.get(key, default)` 的作用，避免 KeyError。
4. **精通 f-string 格式化**：掌握 `{:,}`（千分位）、`{:>10}`（右对齐）、`{:.2f}`（小数位）等格式化语法。
5. **判断数据平衡性**：理解 0.8–1.2、0.5–0.8/1.2–2.0、其他 这三档分类的逻辑。
6. **用 matplotlib 绘制条形图和饼图**：理解 `subplots`、`bar`、`pie`、`text` 等函数的每个参数。
7. **美化图表**：学会隐藏多余边框（spines）、调整布局（tight\_layout）、设置 DPI。
8. **正确保存和关闭图形**：理解 `bbox_inches='tight'`、`dpi=150`、`plt.close()` 的作用。

***

## 一、数据集基本形状

```python
n_samples = len(df)
n_features = df.shape[1]
print(f"\n  ▶ 样本数量 (Number of samples): {n_samples:,}")
print(f"  ▶ 特征数量 (Number of features): {n_features}")
print(f"  ▶ 其中包含 {n_features - 1} 个原始特征 + 1 个目标变量 (target)")
```

### 1.1 `df.shape` 返回什么

`df.shape` 是 pandas DataFrame 的一个属性，返回一个**元组 (tuple)**，格式是 `(行数, 列数)`。

```python
df.shape
# 返回：(209758, 39)
```

- `df.shape[0]` → 行数（样本数），即 209758
- `df.shape[1]` → 列数（特征数），即 39

### 1.2 `len(df)` vs `df.shape[0]`

获取行数有两种常见写法：

- `len(df)` → 返回行数，更简洁，Python 风格。
- `df.shape[0]` → 返回行数，更明确表示"形状的第 0 维"。

两者结果完全相同，本教程混用是为了展示不同写法。一般推荐 `len(df)` 用于行数，`df.shape[1]` 用于列数。

### 1.3 为什么特征数是 39

在模块 0 中，原始数据集有 38 个字段，我们新建了 `target` 列，所以处理后是 39 列。代码中特意强调：

```
其中包含 38 个原始特征 + 1 个目标变量 (target)
```

这是为了提醒读者：`target` 不是原始特征，而是我们从 `Status.Vital` 派生出来的标签列，**不能当作特征用于建模**（否则会造成数据泄露）。

> ⚠️ **重要概念**：在机器学习中，\*\*特征（feature）**和**目标变量（target/label）\*\*必须严格区分：
>
> - 特征：用于预测的输入变量（X）
> - 目标：要预测的输出变量（y）
>
> 如果不小心把 target 当作特征放进模型，模型会"作弊"地学到完美答案，但在实际预测时完全失效。这就是所谓的**数据泄露（data leakage）**。

### 1.4 f-string 格式化：`{n_samples:,}`

`f"..."` 是 Python 3.6+ 的**格式化字符串**（f-string），可以在字符串中直接嵌入变量和表达式。

`{n_samples:,}` 中的 `:,` 是**千分位分隔符**格式化：

```python
n_samples = 209758
print(f"{n_samples:,}")  # 输出：209,758
print(f"{n_samples}")    # 输出：209758
```

加了 `:,` 后，数字会从右往左每三位加一个逗号，提升可读性。对于大数字（如 1,778,176）特别有用。

> 💡 **小贴士**：f-string 是 Python 3.6+ 引入的，比旧的 `%` 格式化和 `str.format()` 都更简洁高效。建议优先使用 f-string。

***

## 二、标签分布统计

```python
# 标签分布
target_counts = df['target'].value_counts()
target_props = df['target'].value_counts(normalize=True) * 100

print(f"\n  ▶ 标签分布 (Target Distribution):")
print(f"      存活 (VIVO = 1): {target_counts.get(1, 0):>10,}  ({target_props.get(1, 0):.2f}%)")
print(f"      死亡 (MORTO = 0): {target_counts.get(0, 0):>10,}  ({target_props.get(0, 0):.2f}%)")
```

### 2.1 `value_counts()` 方法详解

`value_counts()` 是 pandas Series 的方法，用于**统计每个唯一值的出现次数**。它返回一个新的 Series，索引是唯一值，值是频数。

```python
df['target'].value_counts()
# 输出：
# 0    123862
# 1     85896
# Name: target, dtype: int64
```

默认行为：

- 按**频数降序**排列。
- 只统计非缺失值（`NaN` 会被忽略）。
- 返回的 Series 索引是原 Series 的唯一值。

### 2.2 `normalize=True` 参数

`normalize=True` 让 `value_counts()` 返回**比例**而不是频数：

```python
df['target'].value_counts(normalize=True)
# 输出：
# 0    0.590479
# 1    0.409521
# Name: target, dtype: float64
```

- `normalize=False`（默认）：返回频数（整数）。
- `normalize=True`：返回比例（小数，总和为 1）。

本教程中 `target_props = df['target'].value_counts(normalize=True) * 100`，乘以 100 是把比例转换成百分比（如 40.95% 而不是 0.4095）。

> 💡 **小贴士**：`value_counts()` 还有几个常用参数：
>
> - `ascending=True`：按频数升序排列。
> - `dropna=False`：把缺失值也统计进去（默认会忽略 NaN）。
> - `bins=10`：把数值型数据分成 10 个区间统计（适用于连续变量）。

### 2.3 `.get(1, 0)` 字典取值技巧

`target_counts` 是一个 Series，可以像字典一样用 `.get(key, default)` 取值：

```python
target_counts.get(1, 0)  # 取 key=1 的值，如果不存在返回 0
```

- `target_counts.get(1, 0)` → 取 `target=1`（存活）的频数，如果数据中没有 `target=1` 的样本，返回 0。
- `target_counts.get(0, 0)` → 取 `target=0`（死亡）的频数，同理。

**为什么用** **`.get()`** **而不是** **`[]`**：

- `target_counts[1]`：如果 key 不存在，会抛出 `KeyError`，程序崩溃。
- `target_counts.get(1, 0)`：如果 key 不存在，返回默认值 0，程序继续运行。

这是一种**防御性编程**的写法，让代码更健壮。虽然在本数据集中 `target=1` 和 `target=0` 都存在，但养成用 `.get()` 的习惯是好做法。

> ⚠️ **常见问题**：为什么本教程要这么谨慎？因为：
>
> 1. 在某些极端情况下（比如数据加载出错），可能某一类样本不存在。
> 2. 这段代码可能被复用到其他数据集，其他数据集可能只有一类样本。
> 3. 防御性编程让代码更通用、更不容易出错。

### 2.4 f-string 格式化详解

本段代码用了三种 f-string 格式化语法，我们逐一拆解：

#### `{target_counts.get(1, 0):>10,}`

这是一个复合格式化，由两部分组成：`>10` 和 `,`。

- `>`：**右对齐**（right-align）。
- `10`：字段宽度为 10 个字符。
- `,`：千分位分隔符。

```python
val = 85896
print(f"{val:>10,}")  # 输出：    85,896（前面有 4 个空格，总共 10 字符）
print(f"{val:>,}")    # 输出：85,896（没有对齐）
print(f"{val:>10}")   # 输出：     85896（没有千分位）
```

**为什么右对齐**：在打印表格时，数字右对齐能让小数点（或千分位）对齐，方便比较大小。比如：

```
存活 (VIVO = 1):     85,896  (40.95%)
死亡 (MORTO = 0):    123,862  (59.05%)
```

数字右对齐，看起来更整齐。

#### `{target_props.get(1, 0):.2f}`

- `.2f`：**浮点数保留 2 位小数**（f 表示 float）。

```python
val = 40.9521
print(f"{val:.2f}")  # 输出：40.95
print(f"{val:.4f}")  # 输出：40.9521
print(f"{val:.0f}")  # 输出：41（四舍五入到整数）
```

#### `{n_samples:,}`（前面已讲过）

千分位分隔符，把 `209758` 显示成 `209,758`。

> 💡 **小贴士**：f-string 格式化语法是 `{变量:格式}`，格式部分可以组合。常用格式：
>
> - `:,` 千分位
> - `:.2f` 两位小数
> - `:.2%` 百分比（自动乘 100 并加 %）
> - `:>10` 右对齐宽 10
> - `:<10` 左对齐宽 10
> - `:^10` 居中宽 10
> - `:e` 科学计数法

***

## 三、平衡性判断

```python
# 平衡性判断
ratio = target_props.get(1, 0) / target_props.get(0, 0)
if 0.8 <= ratio <= 1.2:
    balance_verdict = "数据基本平衡 (Balanced)"
elif 0.5 <= ratio < 0.8 or 1.2 < ratio <= 2.0:
    balance_verdict = "数据轻度不平衡 (Mildly Imbalanced)"
else:
    balance_verdict = "数据严重不平衡 (Severely Imbalanced)"

print(f"\n  ▶ 平衡性判断 (Balance Assessment): {balance_verdict}")
print(f"     存活/死亡 比例 = {ratio:.3f}")
```

### 3.1 比例的计算

```python
ratio = target_props.get(1, 0) / target_props.get(0, 0)
```

`ratio` 是**存活比例 / 死亡比例**，即正类比例除以负类比例。

本数据集中：

- 存活比例 = 40.95%
- 死亡比例 = 59.05%
- ratio = 40.95 / 59.05 ≈ 0.693

### 3.2 三档分类的逻辑

代码用 `if-elif-else` 把数据平衡性分成三档：

| 条件                                          | 判定    | 含义               |
| ------------------------------------------- | ----- | ---------------- |
| `0.8 <= ratio <= 1.2`                       | 基本平衡  | 正负类比例接近 1:1      |
| `0.5 <= ratio < 0.8` 或 `1.2 < ratio <= 2.0` | 轻度不平衡 | 正负类比例偏离 1:1，但不极端 |
| 其他（ratio < 0.5 或 ratio > 2.0）               | 严重不平衡 | 正负类比例严重失衡        |

### 3.3 为什么这样划分阈值

这三个阈值（0.8、1.2、0.5、2.0）是经验性的，背后逻辑是：

- **0.8–1.2（基本平衡）**：正负类比例在 4:5 到 5:4 之间，差异不大，大多数机器学习算法能直接处理，不需要特殊处理。
- **0.5–0.8 或 1.2–2.0（轻度不平衡）**：正负类比例在 1:2 到 4:5 之间，差异明显但不极端。可以尝试用 `class_weight='balanced'` 或轻微的过采样/欠采样。
- **<0.5 或 >2.0（严重不平衡）**：正负类比例超过 1:2，差异很大。必须用 SMOTE、ADASYN 等重采样方法，或者用适合不平衡数据的评估指标（如 F1、AUC，而不是 Accuracy）。

> ⚠️ **重要概念**：为什么不用 Accuracy 评估不平衡数据？
>
> 假设数据集 99% 是负类，1% 是正类。一个"全预测为负类"的傻瓜模型能达到 99% 的 Accuracy，但完全没用——它一个正类都没找出来。所以不平衡数据要用 Precision、Recall、F1、AUC 等指标。

### 3.4 Python 的链式比较

注意 `0.8 <= ratio <= 1.2` 这种写法，这是 Python 特有的**链式比较**，等价于：

```python
0.8 <= ratio and ratio <= 1.2
```

但在 Python 中可以直接写成 `0.8 <= ratio <= 1.2`，更简洁、更接近数学表达式。这是 Python 相比其他语言的一个语法糖。

### 3.5 本数据集的判定结果

```
存活/死亡 比例 = 0.693
```

- ratio = 0.693
- 落在 `0.5 <= ratio < 0.8` 区间
- 判定为：**数据轻度不平衡 (Mildly Imbalanced)**

> 💡 **小贴士**：本数据集的轻度不平衡在医学数据中很常见。癌症患者中死亡比例通常高于存活比例（因为癌症是严重疾病），所以 40.95% vs 59.05% 的分布在预期之内。这种程度的不平衡可以通过 `class_weight='balanced'` 处理，不需要复杂的重采样。

***

## 四、绘制标签分布图

接下来我们用 matplotlib 绘制一张包含两个子图的标签分布图：左边是条形图（计数），右边是饼图（比例）。

### 4.1 创建画布和子图

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
```

#### `plt.subplots()` 参数详解

`plt.subplots()` 一次性创建画布（Figure）和子图（Axes），返回一个元组 `(fig, axes)`。

- `1`：子图的**行数**（1 行）。
- `2`：子图的**列数**（2 列）。
- 所以 `1, 2` 表示 1 行 2 列，共 2 个子图，水平排列。
- `figsize=(12, 5)`：画布尺寸，**单位是英寸**，格式是 `(宽, 高)`。这里宽 12 英寸、高 5 英寸。

#### `fig` 和 `axes` 是什么

- `fig`：Figure 对象，代表整个画布。可以理解为"一张大图"。
- `axes`：Axes 对象的数组，代表每个子图。注意"Axes"是 matplotlib 的术语，指的是"一个坐标系区域"，不是"坐标轴"。
  - 当 `nrows=1` 时，`axes` 是一维数组，用 `axes[0]`、`axes[1]` 访问。
  - 当 `nrows>1` 且 `ncols>1` 时，`axes` 是二维数组，用 `axes[0][1]` 访问。

```python
axes[0]  # 左边的子图（条形图）
axes[1]  # 右边的子图（饼图）
```

> 💡 **小贴士**：`figsize` 的单位是英寸，最终图片的像素尺寸 = figsize × dpi。比如 `figsize=(12, 5)` 配合 `dpi=150`，图片就是 1800×750 像素。

### 4.2 准备颜色和标签

```python
colors = ['#2ecc71', '#e74c3c']
labels_bar = ['VIVO\n(Alive)', 'MORTO\n(Dead)']
```

- `colors`：两个十六进制颜色码。
  - `#2ecc71`：绿色（代表存活，积极）。
  - `#e74c3c`：红色（代表死亡，消极）。
- `labels_bar`：条形图的 x 轴标签。
  - `\n` 是换行符，让"VIVO"和"(Alive)"分两行显示，更紧凑。

> 💡 **小贴士**：颜色选择有讲究。在医学/生死场景中，绿色=存活、红色=死亡是符合直觉的配色。如果反过来用红色表示存活，会让读者困惑。数据可视化中，**颜色的语义要和数据的语义一致**。

### 4.3 绘制条形图

```python
bars = axes[0].bar(labels_bar, [target_counts.get(1, 0), target_counts.get(0, 0)],
                    color=colors, edgecolor='white', width=0.5)
```

#### `axes[0].bar()` 参数详解

`bar()` 是绘制条形图的函数，参数：

- 第一个参数 `labels_bar`：x 轴的标签（类别名）。
- 第二个参数 `[target_counts.get(1, 0), target_counts.get(0, 0)]`：每个类别对应的数值（条形高度）。
- `color=colors`：每个条形的填充颜色，传入颜色列表。
- `edgecolor='white'`：条形的**边框颜色**设为白色，让条形之间有清晰的分隔。
- `width=0.5`：条形的**宽度**，默认是 0.8。设为 0.5 让条形更细，留出更多空白，视觉上更清爽。

#### 返回值 `bars`

`bar()` 返回一个 `BarContainer` 对象，包含所有条形对象。后续可以用它来在条形上添加文字标签。

### 4.4 在条形上添加数值标签

```python
for bar, val in zip(bars, [target_counts.get(1, 0), target_counts.get(0, 0)]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5000,
                f'{val:,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
```

#### 这段代码的逻辑

用 `zip()` 把条形对象和对应的数值配对，然后遍历每个条形，在条形顶部上方添加文字。

#### `axes[0].text()` 参数详解

`text(x, y, s, ...)` 在坐标 `(x, y)` 处添加文字 `s`。

- **x 坐标**：`bar.get_x() + bar.get_width() / 2`
  - `bar.get_x()`：条形左边缘的 x 坐标。
  - `bar.get_width()`：条形的宽度。
  - 两者相加再除以 2，得到条形的**中心 x 坐标**。
- **y 坐标**：`bar.get_height() + 5000`
  - `bar.get_height()`：条形的高度（即数值）。
  - `+ 5000`：在条形顶部上方 5000 单位处放文字，避免文字和条形重叠。
- **文字内容**：`f'{val:,}'`，数值加千分位，如 `85,896`。
- `ha='center'`：**水平对齐**（horizontal alignment）为居中。其他选项：`'left'`、`'right'`。
- `va='bottom'`：**垂直对齐**（vertical alignment）为底部对齐。其他选项：`'top'`、`'center'`。
- `fontsize=11`：字体大小。
- `fontweight='bold'`：字体加粗。

> 💡 **小贴士**：在条形图上直接标注数值是好习惯，读者不需要对照 y 轴刻度去估算。`ha='center'` + `va='bottom'` 是最常见的组合：文字水平居中、垂直方向从条形顶部向上展开。

### 4.5 设置标题和轴标签

```python
axes[0].set_title('Target Distribution (Count)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Count')
```

- `set_title()`：设置子图标题。
- `set_ylabel()`：设置 y 轴标签。

### 4.6 隐藏多余边框（spines）

```python
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
```

#### 什么是 spines

**Spines** 是 matplotlib 中"边框"的术语，指的是子图四周的边界线。一个子图有 4 条 spines：

- `top`：上边框
- `bottom`：下边框（通常保留，因为有 x 轴）
- `left`：左边框（通常保留，因为有 y 轴）
- `right`：右边框

#### 为什么要隐藏 top 和 right

这是数据可视化的**美学原则**：

- 顶部和右侧的边框**没有承载信息**（没有刻度、没有数据），属于"chartjunk"（图表垃圾）。
- 隐藏它们可以让图表更简洁、更聚焦于数据本身。
- 这是 Edward Tufte 提出的"数据墨水比"（data-ink ratio）原则的体现：尽量减少非数据元素。

```python
# 隐藏前：四周都有边框（封闭的方框）
# 隐藏后：只有左和下有边框（L 形，更简洁）
```

> 💡 **小贴士**：隐藏 top 和 right spines 是数据可视化的"标配"操作。seaborn 库默认就会这样做，所以用 seaborn 画的图通常更美观。如果用纯 matplotlib，建议手动隐藏这两条 spines。

***

## 五、绘制饼图

```python
# 饼图
wedges, texts, autotexts = axes[1].pie(
    [target_counts.get(1, 0), target_counts.get(0, 0)],
    labels=['VIVO (Alive)', 'MORTO (Dead)'],
    autopct='%1.1f%%',
    colors=colors,
    startangle=90,
    explode=(0.03, 0.03),
    textprops={'fontsize': 11}
)
for autotext in autotexts:
    autotext.set_fontweight('bold')
axes[1].set_title('Target Distribution (Proportion)', fontsize=13, fontweight='bold')
```

### 5.1 `axes[1].pie()` 参数详解

`pie()` 是绘制饼图的函数，本教程用了多个参数：

#### 第一个参数：数值列表

```python
[target_counts.get(1, 0), target_counts.get(0, 0)]
```

每个数值决定对应扇形的大小（角度比例）。饼图会自动计算每个数值占总和的比例。

#### `labels`：每个扇形的标签

```python
labels=['VIVO (Alive)', 'MORTO (Dead)']
```

这些标签会显示在每个扇形的外侧。

#### `autopct='%1.1f%%'`：自动显示百分比

`autopct` 是"automatic percentage"的缩写，控制如何在扇形上显示百分比。

- `'%1.1f%%'` 是一个格式化字符串：
  - `%1.1f`：浮点数保留 1 位小数。
  - `%%`：转义后的百分号（单个 `%` 是格式化符号，`%%` 才表示字面意义的 `%`）。
- 所以 `40.9521%` 会显示为 `41.0%`（保留 1 位小数）。

如果设为 `autopct='%1.2f%%'`，则显示 2 位小数，如 `40.95%`。

#### `colors=colors`：扇形颜色

复用前面定义的 `colors = ['#2ecc71', '#e74c3c']`，绿色和红色。

#### `startangle=90`：起始角度

`startangle` 控制第一个扇形从哪个角度开始绘制。

- `startangle=90`：从**正上方**（12 点钟方向）开始，逆时针绘制。
- `startangle=0`（默认）：从正右方（3 点钟方向）开始。
- `startangle=180`：从正下方（6 点钟方向）开始。

设为 90 是惯例，让饼图从顶部开始，更符合阅读习惯。

#### `explode=(0.03, 0.03)`：扇形分离

`explode` 是一个元组，控制每个扇形**向外偏移**的距离。

- `(0.03, 0.03)`：两个扇形都向外偏移 0.03（半径的 3%）。
- `(0.1, 0)`：第一个扇形偏移 0.1，第二个不偏移。
- `(0, 0)`：不偏移（默认）。

本教程让两个扇形都轻微偏移（0.03），制造一种"分离"的视觉效果，让饼图更有层次感。

> 💡 **小贴士**：`explode` 常用于**强调某个类别**。比如你想突出"死亡"类，可以设 `explode=(0, 0.1)`，让死亡扇形向外突出。本教程两个都偏移是为了对称美观。

#### `textprops={'fontsize': 11}`：文字属性

`textprops` 是一个字典，设置所有文字（标签和百分比）的属性。

- `{'fontsize': 11}`：字体大小设为 11。
- 还可以设置 `{'fontsize': 11, 'color': 'white'}` 等。

### 5.2 返回值 `wedges, texts, autotexts`

`pie()` 返回三个对象：

- `wedges`：扇形对象列表（Wedge 对象）。
- `texts`：外侧标签文字对象列表（"VIVO (Alive)"、"MORTO (Dead)"）。
- `autotexts`：扇形内部百分比文字对象列表（"41.0%"、"59.0%"）。

### 5.3 加粗百分比文字

```python
for autotext in autotexts:
    autotext.set_fontweight('bold')
```

遍历所有百分比文字，调用 `set_fontweight('bold')` 加粗。这样百分比更醒目，而外侧标签保持普通字重，形成视觉层次。

### 5.4 设置饼图标题

```python
axes[1].set_title('Target Distribution (Proportion)', fontsize=13, fontweight='bold')
```

和条形图标题风格保持一致。

***

## 六、保存和关闭图形

```python
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "01_target_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [图] 01_target_distribution.png → 标签分布图已保存")
```

### 6.1 `plt.tight_layout()` 的作用

`tight_layout()` 会**自动调整子图间距和边距**，避免子图之间、子图与标题之间、子图与坐标轴标签之间重叠。

- 不调用时：matplotlib 用默认边距，可能导致标题和子图重叠、子图之间间距过大或过小。
- 调用后：自动计算最优布局，让图表更紧凑、更美观。

> 💡 **小贴士**：**只要画了多个子图，就应该调用** **`tight_layout()`**，这是几乎必须的步骤。它能让你的图表瞬间专业很多。

### 6.2 `plt.savefig()` 参数详解

`savefig()` 把当前图形保存到文件。

#### 文件路径

```python
os.path.join(IMG_DIR, "01_target_distribution.png")
```

用 `os.path.join()` 拼接路径，保存到 `img/01_target_distribution.png`。文件扩展名 `.png` 决定了输出格式，也可以用 `.jpg`、`.pdf`、`.svg` 等。

#### `dpi=150`：分辨率

`dpi` 是 **dots per inch**（每英寸点数），控制输出图片的分辨率。

- `dpi=72`：屏幕显示常用，文件小。
- `dpi=150`：本教程使用，适合屏幕显示和一般打印，文件大小适中。
- `dpi=300`：印刷质量，文件较大。

图片的像素尺寸 = `figsize × dpi`。本教程 `figsize=(12, 5)`，`dpi=150`，所以图片是 1800×750 像素。

> ⚠️ **常见问题**：dpi 设多少合适？
>
> - 论文/报告插图：300 dpi（印刷质量）
> - 网页/PPT 展示：150 dpi（足够清晰，文件不太大）
> - 快速预览：72 dpi（最小）
>
> 注意 dpi 越高，文件越大。本教程选 150 是平衡清晰度和文件大小的选择。

#### `bbox_inches='tight'`：紧凑裁剪

`bbox_inches='tight'` 让保存时**自动裁剪掉图形周围的空白**，只保留有内容的区域。

- 不设置时：保存的图片四周会有较大空白，尤其是有标题、标签溢出时。
- 设置 `'tight'`：图片紧贴内容，没有多余白边。

这和 `tight_layout()` 是配套使用的：`tight_layout()` 调整内部布局，`bbox_inches='tight'` 裁剪外部空白。

### 6.3 `plt.close()` 为什么要关闭图形

```python
plt.close()
```

`plt.close()` 关闭当前图形，释放内存。

**为什么要关闭**：

1. **释放内存**：matplotlib 的 Figure 对象会保留所有绘图数据，如果不关闭，内存会持续累积。在循环中画很多图时尤其重要。
2. **避免显示冲突**：在 Jupyter Notebook 中，不关闭的图形可能会被重复显示。
3. **良好习惯**：画完图就关闭，是 matplotlib 的最佳实践。

> ⚠️ **常见问题**：在循环中画图时，如果不调用 `plt.close()`，内存会暴涨甚至 OOM。正确写法：
>
> ```python
> for i in range(100):
>     plt.figure()
>     plt.plot(...)
>     plt.savefig(f"fig_{i}.png")
>     plt.close()  # 必须关闭！
> ```

***

## 七、运行结果

### 7.1 控制台输出

运行本模块代码后，控制台输出大致如下：

```
======================================================================
内容 1: 数据概况
======================================================================

  ▶ 样本数量 (Number of samples): 209,758
  ▶ 特征数量 (Number of features): 39
  ▶ 其中包含 38 个原始特征 + 1 个目标变量 (target)

  ▶ 标签分布 (Target Distribution):
      存活 (VIVO = 1):     85,896  (40.95%)
      死亡 (MORTO = 0):    123,862  (59.05%)

  ▶ 平衡性判断 (Balance Assessment): 数据轻度不平衡 (Mildly Imbalanced)
     存活/死亡 比例 = 0.693
  [图] 01_target_distribution.png → 标签分布图已保存
```

### 7.2 关键数字汇总

| 指标           | 数值                        |
| ------------ | ------------------------- |
| 样本数量         | 209,758                   |
| 特征数量         | 39（38 原始 + 1 target）      |
| 存活 (VIVO=1)  | 85,896 (40.95%)           |
| 死亡 (MORTO=0) | 123,862 (59.05%)          |
| 存活/死亡比例      | 0.693                     |
| 平衡性判定        | 轻度不平衡 (Mildly Imbalanced) |

### 7.3 标签分布图

![标签分布图](../img/01_target_distribution.png)

**图表解读**：

- **左图（条形图）**：展示每个类别的绝对数量。绿色条形（VIVO）高度为 85,896，红色条形（MORTO）高度为 123,862。可以直观看到死亡样本比存活样本多约 38,000。
- **右图（饼图）**：展示每个类别的比例。绿色扇形占 41.0%，红色扇形占 59.0%。可以直观看到死亡占多数，但差距不算极端。

> 💡 **小贴士**：条形图和饼图各有优势：
>
> - **条形图**：适合比较绝对数量，差异更直观。
> - **饼图**：适合展示比例关系，强调"部分占整体"。
>
> 本教程同时画两种图，让读者从两个角度理解数据分布。

***

## 八、本模块小结

### 核心知识点回顾

1. **数据形状**：`df.shape` 返回 `(行数, 列数)`，`df.shape[0]` 是行数，`df.shape[1]` 是列数。
2. **标签分布统计**：
   - `value_counts()` 统计频数。
   - `value_counts(normalize=True)` 统计比例。
   - `.get(key, default)` 防御性取值，避免 KeyError。
3. **f-string 格式化**：
   - `{:,}` 千分位
   - `{:.2f}` 两位小数
   - `{:>10,}` 右对齐 + 千分位
4. **平衡性判断**：
   - ratio = 正类比例 / 负类比例
   - 0.8–1.2：基本平衡
   - 0.5–0.8 或 1.2–2.0：轻度不平衡
   - 其他：严重不平衡
5. **matplotlib 绘图**：
   - `plt.subplots(1, 2, figsize=(12, 5))` 创建 1 行 2 列子图。
   - `bar()` 画条形图，`pie()` 画饼图。
   - `text()` 在图上添加文字标签。
   - 隐藏 `top` 和 `right` spines 提升美观度。
6. **图形保存**：
   - `tight_layout()` 调整内部布局。
   - `savefig(dpi=150, bbox_inches='tight')` 保存高分辨率紧凑图片。
   - `plt.close()` 释放内存。

### 本数据集的关键发现

- 数据集有 209,758 条记录，39 列（38 原始特征 + 1 target）。
- 标签分布：存活 40.95%，死亡 59.05%。
- 数据**轻度不平衡**，比例 0.693。
- 这种程度的不平衡在医学数据中常见，后续建模时可以用 `class_weight='balanced'` 处理。

### 下一模块预告

本模块完成了数据概况分析，知道了数据集的大小和标签分布。下一模块（**模块 2：缺失值分析**）将深入分析：

- 每个特征的缺失比例
- 缺失模式（MCAR/MAR/MNAR）
- 缺失值之间的相关性
- 缺失值的可视化

***

> 📌 **思考题**：
>
> 1. 如果数据集严重不平衡（比如 1:99），你会如何调整本模块的可视化代码？
> 2. 为什么 `ratio = target_props.get(1, 0) / target_props.get(0, 0)` 用比例相除，而不是用频数相除？两者结果一样吗？
> 3. 饼图在什么情况下不适合使用？条形图相比饼图有什么优势？
> 4. 如果要把这张图保存成 PDF 格式用于论文，需要修改哪些参数？

