# 模块 3：分布分析

> **核心思想**：在建模之前，我们必须先"认识"每一个数值特征。它长什么样？是钟形对称的，还是歪向一边？尾巴有多长？有没有离群点？这些信息决定了我们后续是否需要做对数变换、是否需要标准化、是否能用基于正态假设的模型。本模块带你一步步完成分布分析的全流程。

---

## 一、学习目标

学完本模块后，你将能够：

1. **理解** `df.select_dtypes()` 按数据类型筛选列的机制，并能用列表推导式排除不需要的列。
2. **计算并解释** 五大描述性统计量：均值、中位数、标准差、偏度、峰度，并写出它们的公式直觉。
3. **判断** 一个特征是近似正态、轻度偏态还是显著偏态，并知道每种情况对建模的影响。
4. **绘制并解读** 三种核心分布图：直方图 + KDE、Q-Q 图、分组箱线图。
5. **理解** 核密度估计（KDE）的原理，以及正态分布概率密度函数（PDF）公式的每个符号含义。
6. **识别** "数值陷阱"——即那些看起来是数值、本质却是类别的变量（如职业代码、形态学代码）。
7. **运用** Matplotlib 的关键参数（`density`、`alpha`、`patch_artist` 等）自定义图表外观。

---

## 二、背景回顾

本教程使用的是**巴西癌症登记数据**（Brazilian Cancer Registry）。在模块 1 中，我们已经过滤掉 `Status.Vital`（生存状态）缺失的样本，剩余 **209,758 条**记录。

经过模块 2 的缺失值分析后，数据集中被 pandas 识别为数值型（`np.number`）的特征共有 4 个：

| 特征名 | 含义 | 备注 |
|---|---|---|
| `Age` | 患者诊断时的年龄（岁） | 真正的连续型数值 |
| `Code.Profession` | 职业分类代码 | **本质是类别变量**（数值陷阱） |
| `Code.of.Morphology` | 肿瘤形态学代码（ICD-O 标准） | **本质是类别变量**（数值陷阱） |
| `year` | 诊断年份（如 2008、2013） | 离散型时间变量 |

> ⚠️ **重要提醒**：被 pandas 识别为"数值"≠ 真正的连续型数值变量。我们会在本模块末尾详细讨论"数值陷阱"问题。

---

## 三、第一步：选取数值型特征

### 3.1 用 `select_dtypes` 按数据类型筛选列

```python
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
```

**逐层拆解：**

- `df.select_dtypes(include=[np.number])`：这是 pandas DataFrame 的一个方法，作用是**按列的数据类型筛选**，返回一个新的 DataFrame，只包含符合类型的列。
  - `include=[np.number]` 表示"只保留数值类型的列"。`np.number` 是 NumPy 中所有数值类型的"总称"，它涵盖了：
    - `int8 / int16 / int32 / int64`（整数）
    - `float16 / float32 / float64`（浮点数）
    - `uint8 / uint16 / uint32 / uint64`（无符号整数）
    - `complex64 / complex128`（复数，极少见）
  - 如果你想筛选别的类型，可以这样用：
    - `include=['object']` → 筛选字符串/混合类型列
    - `include=['datetime64']` → 筛选日期时间列
    - `include=['category']` → 筛选类别型列
    - `exclude=[np.number]` → 排除数值列，保留其他所有列
- `.columns`：取出筛选后 DataFrame 的列名（一个 `Index` 对象）。
- `.tolist()`：把 `Index` 对象转换成普通的 Python 列表（`list`），方便后续用列表推导式操作。

> 💡 **小贴士**：为什么用 `np.number` 而不是 `'number'`？两者其实等价，pandas 都支持。但用 `np.number` 更明确地表明我们引用的是 NumPy 的类型系统，在团队协作时代码可读性更好。

### 3.2 用列表推导式排除不需要的列

```python
exclude_cols = ['target', 'Patient.Code']
numeric_features = [c for c in numeric_cols if c not in exclude_cols]
```

**列表推导式（List Comprehension）语法解析：**

```python
[<表达式> for <变量> in <可迭代对象> if <条件>]
```

读作："对于 `numeric_cols` 中的每一个列名 `c`，如果 `c` 不在 `exclude_cols` 列表里，就把 `c` 放进新列表。"

**为什么要排除这两列？**

- `target`：这是我们要预测的目标变量（生存状态）。在分布分析阶段，我们只关心**特征**的分布，不关心目标本身。把目标混进来会污染分析结果。
- `Patient.Code`：这是患者的唯一标识符（ID），本质上是一个"行号"，没有任何统计意义。它的均值、标准差都是无意义的数字。

> 📌 **常见问题**：为什么不直接在 `select_dtypes` 里排除？
> 答：`select_dtypes` 只能按**数据类型**筛选，不能按**列名**筛选。所以需要先用 `select_dtypes` 拿到所有数值列，再用列表推导式按名字剔除特定列。这是 pandas 工作流中非常常见的"两步走"模式。

运行后，`numeric_features` 的值为：

```python
['Age', 'Code.Profession', 'Code.of.Morphology', 'year']
```

---

## 四、第二步：计算描述性统计量

接下来，我们对每一个数值特征计算 5 个核心统计量。先看完整代码：

```python
dist_results = []
for col in numeric_features:
    data = df[col].dropna()        # 丢弃缺失值
    if len(data) < 50:             # 样本太少则跳过
        continue

    mean_val = data.mean()         # 均值
    median_val = data.median()     # 中位数
    std_val = data.std()           # 标准差
    skewness = data.skew()         # 偏度
    kurtosis_val = data.kurtosis() # 峰度
    ...
```

### 4.1 `data.dropna()`：丢弃缺失值

在计算统计量之前，必须先丢弃缺失值（`NaN`）。否则 `mean()`、`std()` 等函数虽然能自动跳过 `NaN`，但 `len(data)` 会包含缺失值的计数，导致样本量统计错误。`dropna()` 返回一个不含 `NaN` 的新 Series。

### 4.2 `if len(data) < 50: continue`：样本量门槛

如果某个特征的有效样本少于 50 条，统计量（尤其是偏度和峰度）会非常不稳定，没有参考价值。所以直接 `continue` 跳过。

### 4.3 五大统计量详解

#### (1) `data.mean()` —— 均值（Mean）

**含义**：所有数据的算术平均值，反映数据的"中心位置"。

**公式直觉**：

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}
$$

**特点**：
- 优点：用了所有数据的信息，数学性质好（如最小二乘法的核心）。
- 缺点：**对离群值极其敏感**。一个亿万富翁走进酒吧，所有顾客的"平均资产"瞬间暴涨。

#### (2) `data.median()` —— 中位数（Median）

**含义**：把数据从小到大排序后，位于正中间的那个数。如果数据量是偶数，则取中间两个数的平均值。

**公式直觉**：

$$
\text{Median} = \begin{cases}
x_{(n+1)/2} & n \text{ 为奇数} \\
\frac{1}{2}\left(x_{n/2} + x_{n/2+1}\right) & n \text{ 为偶数}
\end{cases}
$$

其中 $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$ 是排序后的数据。

**特点**：
- 优点：**对离群值鲁棒（robust）**。亿万富翁走进酒吧，中位数几乎不变。
- 缺点：只用了中间一两个数据点的信息，忽略了大部分数据。

> 💡 **小贴士**：比较均值和中位数，是判断分布偏态的"快速肉眼法"：
> - 均值 ≈ 中位数 → 分布大致对称
> - 均值 > 中位数 → 右偏（正偏），右侧有长尾
> - 均值 < 中位数 → 左偏（负偏），左侧有长尾

#### (3) `data.std()` —— 标准差（Standard Deviation）

**含义**：衡量数据围绕均值的"离散程度"。标准差越大，数据越分散；越小，数据越集中。

**公式直觉**：

$$
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}
$$

**关键点**：
- 分母是 $n-1$ 而不是 $n$，这叫**贝塞尔校正（Bessel's correction）**。原因是：我们用样本均值 $\bar{x}$ 代替了总体均值 $\mu$，这会低估方差，所以除以 $n-1$ 来修正。pandas 的 `std()` 默认 `ddof=1`，即用 $n-1$。
- 标准差的单位与原始数据相同（如 Age 的标准差单位是"岁"），所以比方差（单位是"岁的平方"）更直观。

#### (4) `data.skew()` —— 偏度（Skewness）

**含义**：衡量分布的**不对称程度**。这是本模块最核心的统计量之一。

**公式直觉**：

$$
\text{Skewness} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^3}{\left(\sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2}\right)^3}
$$

**直觉理解**：分子是"三阶中心矩"，衡量数据偏离均值后的"立方和"。如果右侧（大于均值）的尾巴长，正的偏离值在立方后会主导，偏度为正；反之偏度为负。

**判断标准（本教程采用）：**

| 偏度绝对值 | 分布类型 | 说明 |
|---|---|---|
| `abs(skewness) < 0.5` | 近似正态 (Approx. Normal) | 对称性较好，可视为正态 |
| `0.5 ≤ abs(skewness) < 1` | 轻度偏态 (Mildly Skewed) | 有点歪，但不太严重 |
| `abs(skewness) ≥ 1` | 显著偏态 (Highly Skewed) | 严重不对称，需要变换 |

**偏度符号的含义：**
- 偏度 > 0：**右偏（正偏）**，右侧有长尾，均值 > 中位数。
- 偏度 < 0：**左偏（负偏）**，左侧有长尾，均值 < 中位数。
- 偏度 = 0：完全对称（如正态分布）。

> ⚠️ **常见误区**：很多人以为"右偏"就是"峰在右边"。**恰恰相反！** 右偏分布的峰在**左边**，长尾在**右边**。记忆口诀："偏"指的是**尾巴的方向**，不是峰的方向。

对应代码：

```python
if abs(skewness) < 0.5:
    dist_type = "近似正态 (Approx. Normal)"
elif abs(skewness) < 1:
    dist_type = "轻度偏态 (Mildly Skewed)"
else:
    dist_type = "显著偏态 (Highly Skewed)"
```

#### (5) `data.kurtosis()` —— 峰度（Kurtosis）

**含义**：衡量分布的"尖峭程度"和"尾巴厚度"。

**公式直觉**：

$$
\text{Kurtosis} = \frac{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^4}{\left(\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2\right)^2} - 3
$$

**注意**：pandas 的 `kurtosis()` 默认返回的是**超额峰度（excess kurtosis）**，即"原始峰度减 3"。这样做的目的是让正态分布的峰度等于 0，方便比较。

**判断标准：**

| 超额峰度 | 分布形态 | 说明 |
|---|---|---|
| > 0 | **尖峰厚尾（Leptokurtic）** | 比正态分布更尖、尾巴更厚，极端值更多 |
| = 0 | **正态（Mesokurtic）** | 与正态分布相同 |
| < 0 | **平峰薄尾（Platykurtic）** | 比正态分布更平、尾巴更薄，极端值更少 |

> 💡 **小贴士**：峰度大意味着"厚尾"，这在金融风控中极其重要——厚尾分布意味着"黑天鹅"事件（极端损失）比正态分布预测的更频繁。2008 年金融危机的部分原因，就是风险模型错误假设了正态分布（薄尾）。

### 4.4 把结果存入 DataFrame

```python
dist_results.append({
    'Feature': col,
    'Count': len(data),
    'Mean': mean_val,
    'Median': median_val,
    'Std': std_val,
    'Skewness': skewness,
    'Kurtosis': kurtosis_val,
    'Distribution': dist_type
})

dist_df = pd.DataFrame(dist_results)
```

这里我们用一个列表 `dist_results` 收集每个特征的统计字典，最后一次性转成 DataFrame。这比"逐行 append 到 DataFrame"高效得多，是 pandas 的最佳实践。

---

## 五、第三步：图 3a —— 直方图 + KDE + 正态曲线

### 5.1 代码全貌

```python
plot_features = ['Age'] if 'Age' in numeric_features else numeric_features[:2]

fig, axes = plt.subplots(1, len(plot_features), figsize=(6 * len(plot_features), 5))
if len(plot_features) == 1:
    axes = [axes]

for ax, col in zip(axes, plot_features):
    data = df[col].dropna()
    if len(data) > 50000:
        data_sample = data.sample(50000, random_state=42)
    else:
        data_sample = data

    ax.hist(data_sample, bins=80, density=True, alpha=0.6,
            color='#3498db', edgecolor='white', label='Histogram')
    data_sample.plot.kde(ax=ax, color='#e74c3c', linewidth=2.5, label='KDE')
    x_range = np.linspace(data_sample.min(), data_sample.max(), 1000)
    normal_pdf = (1 / (data_sample.std() * np.sqrt(2 * np.pi))) * \
                 np.exp(-0.5 * ((x_range - data_sample.mean()) / data_sample.std()) ** 2)
    ax.plot(x_range, normal_pdf, '--', color='#2c3e50', linewidth=1.5,
            label=f'N(μ={data_sample.mean():.1f}, σ={data_sample.std():.1f})')
```

### 5.2 三元表达式选取绘图特征

```python
plot_features = ['Age'] if 'Age' in numeric_features else numeric_features[:2]
```

这是一个**三元条件表达式**：如果 `Age` 在数值特征里，就只画 `Age`；否则取前两个数值特征。优先选 `Age` 是因为它是本数据集中**唯一真正的连续型数值变量**，最适合展示分布分析。

### 5.3 处理"单子图"的陷阱

```python
fig, axes = plt.subplots(1, len(plot_features), figsize=(6 * len(plot_features), 5))
if len(plot_features) == 1:
    axes = [axes]
```

**为什么要 `axes = [axes]`？**

当 `plt.subplots(1, 1)` 创建 1×1 子图时，返回的 `axes` 是**单个 Axes 对象**，不是列表。但 `for ax, col in zip(axes, plot_features)` 需要 `axes` 是可迭代的。所以用 `axes = [axes]` 把它包成列表，保证循环能正常工作。这是 Matplotlib 中极其常见的"坑"。

### 5.4 大数据集抽样

```python
if len(data) > 50000:
    data_sample = data.sample(50000, random_state=42)
else:
    data_sample = data
```

**为什么要抽样？**
- `Age` 有 209,443 条数据，全部画直方图会非常慢，KDE 计算更是 $O(n^2)$ 复杂度。
- 50,000 条已经足够稳定地估计分布形状。
- `random_state=42` 保证每次抽样结果一致，便于复现。42 是程序员的"梗"（来自《银河系漫游指南》中"生命、宇宙和一切的答案"）。

### 5.5 `ax.hist()` 参数详解

```python
ax.hist(data_sample, bins=80, density=True, alpha=0.6,
        color='#3498db', edgecolor='white', label='Histogram')
```

| 参数 | 值 | 作用 |
|---|---|---|
| `data_sample` | 输入数据 | 要绘制直方图的数据 |
| `bins` | `80` | 把数据范围等分成 80 个区间（"桶"）。`bins` 越多，直方图越细，但太大会被噪声主导；太少则掩盖结构。经验法则：`bins ≈ √n`，这里 √50000 ≈ 224，但 80 已经足够清晰 |
| `density` | `True` | **关键参数！** 把纵轴从"频数（count）"归一化为"概率密度（density）"，使直方图总面积 = 1。这样才能和 KDE 曲线、正态 PDF 曲线画在同一坐标系下比较 |
| `alpha` | `0.6` | 透明度，0=全透明，1=不透明。设为 0.6 是为了能透过直方图看到下方的曲线 |
| `color` | `'#3498db'` | 直方图的填充色（蓝色，十六进制 RGB） |
| `edgecolor` | `'white'` | 每个柱子的边框颜色。白色边框让相邻柱子视觉上更易区分 |
| `label` | `'Histogram'` | 图例标签 |

> 📌 **常见问题：为什么用 `density=True` 而不是 `frequency`（频数）？**
>
> 因为我们要把直方图、KDE 曲线、正态 PDF 曲线**画在同一张图上比较**。后两者都是"概率密度"，单位是"1/单位"。如果直方图用频数（如 500、1000），纵轴量纲不一致，曲线会被压扁到几乎看不见。`density=True` 把直方图归一化为"概率密度"，三者量纲统一，可以直接叠加。

### 5.6 KDE（核密度估计）原理详解

```python
data_sample.plot.kde(ax=ax, color='#e74c3c', linewidth=2.5, label='KDE')
```

**什么是 KDE（Kernel Density Estimation，核密度估计）？**

KDE 是一种**非参数**方法，用于从一组样本数据中估计其背后的概率密度函数（PDF）。直方图是"分桶计数"，KDE 则是"每个数据点贡献一个小山丘，叠加起来形成平滑曲线"。

**数学公式：**

$$
\hat{f}(x) = \frac{1}{n h} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
$$

**逐项解释：**
- $\hat{f}(x)$：在点 $x$ 处估计的概率密度。
- $n$：样本量。
- $h$：**带宽（bandwidth）**，控制"小山丘"的宽度。$h$ 越大，曲线越平滑但可能掩盖细节；$h$ 越小，曲线越尖锐但可能过拟合噪声。
- $K(\cdot)$：**核函数（kernel function）**，通常用高斯核（正态分布的形状）。它决定了每个数据点贡献的"小山丘"长什么样。
- $x_i$：第 $i$ 个数据点。

**直觉理解：**
想象每个数据点都"点亮"一盏小灯，灯光呈高斯分布向四周衰减。所有灯的光叠加起来，就形成了一条平滑的密度曲线。带宽 $h$ 就是"灯的亮度范围"。

**KDE vs 直方图：**
| 特性 | 直方图 | KDE |
|---|---|---|
| 形态 | 阶梯状（分桶） | 平滑曲线 |
| 依赖参数 | `bins`（桶数） | `bandwidth`（带宽） |
| 边界效应 | 有明显的桶边界 | 无桶边界，但边界处可能泄漏 |
| 计算成本 | 低 | 较高（$O(n^2)$） |

### 5.7 正态分布 PDF 公式详解

```python
x_range = np.linspace(data_sample.min(), data_sample.max(), 1000)
normal_pdf = (1 / (data_sample.std() * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((x_range - data_sample.mean()) / data_sample.std()) ** 2)
ax.plot(x_range, normal_pdf, '--', color='#2c3e50', linewidth=1.5,
        label=f'N(μ={data_sample.mean():.1f}, σ={data_sample.std():.1f})')
```

**正态分布概率密度函数（PDF）公式：**

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$`

**每个符号的含义：**

| 符号 | 含义 | 代码对应 |
|---|---|---|
| $x$ | 自变量（横轴的值） | `x_range` |
| $\mu$ | 均值（决定钟形曲线的中心位置） | `data_sample.mean()` |
| $\sigma$ | 标准差（决定钟形曲线的宽窄） | `data_sample.std()` |
| $\pi$ | 圆周率 ≈ 3.14159 | `np.pi` |
| $\exp(\cdot)$ | 自然指数函数 $e^{(\cdot)}$ | `np.exp(...)` |
| $\frac{1}{\sigma\sqrt{2\pi}}$ | 归一化常数，保证曲线下面积 = 1 | `1 / (data_sample.std() * np.sqrt(2 * np.pi))` |
| $-\frac{(x-\mu)^2}{2\sigma^2}$ | 指数部分，决定钟形 | `-0.5 * ((x_range - mean) / std) ** 2` |

**直觉理解：**
- $\mu$ 决定"钟"的中心在哪里。
- $\sigma$ 决定"钟"有多宽。$\sigma$ 越大，钟越矮胖；$\sigma$ 越小，钟越瘦高。
- 指数部分 $-\frac{(x-\mu)^2}{2\sigma^2}$ 始终 ≤ 0，所以 $\exp$ 后 ≤ 1。当 $x = \mu$ 时取最大值 1，对应峰值 $\frac{1}{\sigma\sqrt{2\pi}}$。
- 我们画这条曲线，是为了**直观对比**：如果数据真的服从正态分布，KDE 曲线应该和这条虚线重合。偏离越多，说明越不正态。

### 5.8 `np.linspace` 生成等差数列

```python
x_range = np.linspace(data_sample.min(), data_sample.max(), 1000)
```

`np.linspace(start, stop, num)` 生成从 `start` 到 `stop` 的 `num` 个**等间距**点（包含两端）。

**示例：**
```python
np.linspace(0, 10, 5)  # → array([ 0. ,  2.5,  5. ,  7.5, 10. ])
```

**为什么用 1000 个点？**
画曲线需要足够多的点才能显得平滑。1000 个点足以让正态 PDF 曲线看起来非常顺滑，而计算成本可以忽略。

> 💡 **小贴士**：`np.linspace` 和 `np.arange` 的区别：
> - `linspace(0, 10, 5)` → 指定**点数**，自动算间距，**包含终点**。
> - `arange(0, 10, 2.5)` → 指定**步长**，自动算点数，**不包含终点**。
> 画曲线时优先用 `linspace`，因为你能精确控制采样密度。

### 5.9 结果图

![分布直方图+KDE](../img/03a_distribution_hist_kde.png)

**如何解读这张图？**
- **蓝色直方图**：Age 的实际分布。可以看到峰值在 65 岁左右，左侧（低龄）上升较缓，右侧（高龄）下降较快，整体略微左偏。
- **红色 KDE 曲线**：平滑后的密度估计，与直方图形状吻合。
- **黑色虚线**：以样本均值和标准差为参数的正态分布曲线。对比可见，Age 分布比正态分布略尖，左尾稍长，印证了偏度 = -0.683（轻度左偏）的结论。

---

## 六、第四步：图 3b —— Q-Q 图（分位数图）

### 6.1 代码全貌

```python
from scipy import stats
fig, axes = plt.subplots(1, len(plot_features), figsize=(6 * len(plot_features), 5))
if len(plot_features) == 1:
    axes = [axes]

for ax, col in zip(axes, plot_features):
    data = df[col].dropna()
    if len(data) > 5000:
        data_qq = data.sample(5000, random_state=42)
    else:
        data_qq = data

    stats.probplot(data_qq, dist="norm", plot=ax)
    ax.get_lines()[0].set_markerfacecolor('#3498db')
    ax.get_lines()[1].set_color('#e74c3c')
```

### 6.2 Q-Q 图原理详解

**Q-Q 图（Quantile-Quantile Plot，分位数-分位数图）** 是判断数据是否服从某分布（通常是正态分布）的最直观工具。

**核心思想：分位数对分位数。**

具体步骤：
1. 把样本数据从小到大排序：$x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$。
2. 对每个 $x_{(i)}$，计算它的"经验分位数"位置 $p_i = \frac{i - 0.5}{n}$（用 0.5 是避免 $p=0$ 或 $p=1$ 时理论分位数为无穷大）。
3. 在标准正态分布 $N(0,1)$ 中，找到分位数 $z_i = \Phi^{-1}(p_i)$（即标准正态分布的 $p_i$ 分位数）。
4. 在坐标系中画散点：横轴是 $z_i$（理论分位数），纵轴是 $x_{(i)}$（样本分位数）。

**如何判断正态性？**
- **如果数据服从正态分布**：所有点会落在一条直线上（这条直线就是"参考线"）。
- **如果数据有重尾（厚尾）**：两端的点会偏离参考线，呈"S 形"或"反 S 形"。
- **如果数据右偏**：右上端的点会向上翘起（高于参考线）。
- **如果数据左偏**：左下端的点会向下垂落（低于参考线）。

> 💡 **小贴士**：Q-Q 图比偏度数值更直观。偏度只给一个数字，Q-Q 图能让你看到"哪里偏离了正态"——是中间、是两端、还是某一侧。

### 6.3 `stats.probplot` 参数详解

```python
stats.probplot(data_qq, dist="norm", plot=ax)
```

| 参数 | 值 | 作用 |
|---|---|---|
| `data_qq` | 输入数据 | 要检验的样本数据（一维数组） |
| `dist` | `"norm"` | 假设的理论分布。`"norm"` 是正态分布；也可以是 `"expon"`（指数）、`"lognorm"`（对数正态）、`"uniform"`（均匀）等 |
| `plot` | `ax` | 把图绘制到指定的 Matplotlib Axes 对象上。如果不传，则不画图，只返回计算结果 |

**返回值**：`probplot` 会返回一个元组 `(osm, osr)`，其中 `osm` 是理论分位数，`osr` 是样本分位数。但因为我们传了 `plot=ax`，它会自动画图。

### 6.4 `ax.get_lines()[0]` 和 `[1]` 分别是什么？

```python
ax.get_lines()[0].set_markerfacecolor('#3498db')  # 数据点
ax.get_lines()[1].set_color('#e74c3c')            # 参考线
```

`stats.probplot` 在 Axes 上画了**两条 Line2D 对象**：

| 索引 | 对应内容 | 默认外观 |
|---|---|---|
| `ax.get_lines()[0]` | **样本数据点**（散点） | 蓝色圆点 |
| `ax.get_lines()[1]` | **参考线**（理论直线） | 红色实线 |

**为什么是这两条？**
- 数据点：每个点代表"样本分位数 vs 理论分位数"。
- 参考线：如果数据完全服从正态分布，所有数据点都会落在这条线上。这条线通常用数据的均值和标准差（或最小二乘拟合）确定。

我们通过 `set_markerfacecolor` 把数据点改成蓝色，通过 `set_color` 把参考线改成红色，让配色和图 3a 保持一致。

> 📌 **常见问题**：为什么 Q-Q 图抽样只用 5000 个点，而直方图用 50000 个？
> 答：Q-Q 图是散点图，点太多会糊成一团，反而看不清偏离模式。5000 个点足够展示分布形状，又不会过于密集。直方图是分桶统计，需要更多数据才能让每个桶内的计数稳定。

### 6.5 结果图

![Q-Q图](../img/03b_qq_plot.png)

**如何解读这张 Q-Q 图？**
- 如果 Age 完全服从正态分布，所有蓝色点都会落在红色参考线上。
- 实际观察：中间部分大致贴合参考线，但**左下端**（低龄端）的点偏离参考线向下，**右上端**（高龄端）的点偏离参考线向下。这种"两端都向下弯"的形态，说明 Age 分布比正态分布**尾部更轻**（薄尾），同时整体略左偏，与偏度 = -0.683 的结论一致。

---

## 七、第五步：图 3c —— 按目标分组的箱线图

### 7.1 代码全貌

```python
if 'Age' in df.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    age_data = df[['Age', 'target']].dropna()
    if len(age_data) > 50000:
        age_data = age_data.sample(50000, random_state=42)

    bp = ax.boxplot([age_data.loc[age_data['target'] == 1, 'Age'].values,
                     age_data.loc[age_data['target'] == 0, 'Age'].values],
                    tick_labels=['VIVO (Alive)', 'MORTO (Dead)'],
                    patch_artist=True,
                    widths=0.4)

    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
```

### 7.2 箱线图（Boxplot）构成详解

箱线图由五数概括（Five-number summary）构成，能一图展示分布的中心、离散、偏态和离群值。

**箱线图的五大组成部分：**

```
          ╎ ← 上须线（upper whisker）
          ╎
        ┌─┴─┐
        │   │ ← 箱体上沿 = Q3（75% 分位数）
        │   │
        │───│ ← 中位线 = Q2（50% 分位数，即中位数）
        │   │
        │   │ ← 箱体下沿 = Q1（25% 分位数）
        └─┬─┘
          ╎
          ╎ ← 下须线（lower whisker）
          ╎
        ○     ← 异常点（flier，超出须线范围）
```

**各部分含义：**

| 部分 | 含义 | 计算方式 |
|---|---|---|
| **箱体（box）** | 中间 50% 数据的范围 | 从 Q1 到 Q3 |
| **箱体高度** | 四分位距 IQR | IQR = Q3 − Q1 |
| **中位线（median）** | 数据的中位数 | Q2 |
| **上须线（upper whisker）** | "正常"数据的上界 | min(Q3 + 1.5×IQR, 数据最大值) |
| **下须线（lower whisker）** | "正常"数据的下界 | max(Q1 − 1.5×IQR, 数据最小值) |
| **异常点（fliers）** | 超出须线范围的点 | < Q1−1.5×IQR 或 > Q3+1.5×IQR |

> 💡 **小贴士**：1.5×IQR 是 Matplotlib 默认的异常点判定阈值。这个阈值来自 John Tukey（箱线图发明者）的经验：对于正态分布，1.5×IQR 大约对应 ±2.7σ，覆盖了 99.3% 的数据，剩下的 0.7% 被标记为异常点。

### 7.3 `patch_artist=True` 参数

```python
bp = ax.boxplot([...], patch_artist=True, widths=0.4)
```

| 参数 | 作用 |
|---|---|
| `patch_artist` | 默认 `False`，箱体只有边框，没有填充色。设为 `True` 后，箱体变成 `Patch` 对象，**可以填充颜色**。这是自定义箱线图外观的关键开关 |
| `widths` | 箱体的宽度（0~1 之间）。0.4 是一个适中值，既不会太胖显得笨重，也不会太瘦难以看清 |
| `tick_labels` | 每个箱子的标签（之前版本叫 `labels`，新版已更名） |

### 7.4 `bp` 字典的五个键

`ax.boxplot()` 返回一个字典 `bp`，包含箱线图的所有图形组件：

| 键 | 对应部分 | 类型 | 数量 |
|---|---|---|---|
| `bp['boxes']` | **箱体**（Q1 到 Q3 的矩形） | Line2D/Patch 列表 | 每组 1 个 |
| `bp['whiskers']` | **须线**（上下两条） | Line2D 列表 | 每组 2 个（上须、下须） |
| `bp['caps']` | **须线端帽**（须线末端的横线） | Line2D 列表 | 每组 2 个（上帽、下帽） |
| `bp['medians']` | **中位线**（箱体内的横线） | Line2D 列表 | 每组 1 个 |
| `bp['fliers']` | **异常点**（超出须线的点） | Line2D 列表 | 每组 1 个（包含所有异常点） |

**对应代码：**

```python
bp['boxes'][0].set_facecolor('#2ecc71')   # 第一个箱体（VIVO）填充绿色
bp['boxes'][1].set_facecolor('#e74c3c')   # 第二个箱体（MORTO）填充红色
for whisker in bp['whiskers']:
    whisker.set_color('gray')             # 所有须线设为灰色
for cap in bp['caps']:
    cap.set_color('gray')                 # 所有端帽设为灰色
for median in bp['medians']:
    median.set_color('black')             # 所有中位线设为黑色
    median.set_linewidth(2)               # 中位线加粗
```

**配色逻辑：**
- 绿色（`#2ecc71`）→ VIVO（存活），象征"生"。
- 红色（`#e74c3c`）→ MORTO（死亡），象征"亡"。
- 这种"语义化配色"让读者一眼就能理解颜色含义，比随机配色专业得多。

### 7.5 布尔索引 + `.values`

```python
age_data.loc[age_data['target'] == 1, 'Age'].values
```

**逐层拆解：**

1. `age_data['target'] == 1`：这是一个**布尔 Series**，对每一行判断 `target` 是否等于 1。结果形如 `[True, False, True, ...]`。
2. `age_data.loc[布尔Series, 'Age']`：`.loc[行选择, 列选择]` 是 pandas 的标签索引。这里"行选择"是布尔 Series，表示"只保留 True 对应的行"；"列选择"是 `'Age'`，表示"只取 Age 列"。结果是一个 Series，包含所有 `target==1` 患者的年龄。
3. `.values`：把 Series 转换成 NumPy 数组。`ax.boxplot()` 需要传入数组或列表的列表，所以用 `.values` 取出底层数据。

> 📌 **常见问题**：`.loc` 和 `.iloc` 的区别？
> - `.loc`：按**标签**索引，如 `df.loc[行标签, 列标签]`。
> - `.iloc`：按**位置**索引，如 `df.iloc[0:5, 2]`（取前 5 行第 3 列）。
> 布尔索引通常用 `.loc`，因为它能精确匹配标签语义。

### 7.6 结果图

![年龄分组箱线图](../img/03c_age_by_target_boxplot.png)

**如何解读这张箱线图？**
- **VIVO（存活）组**：中位数偏低，箱体偏下，说明存活患者整体更年轻。
- **MORTO（死亡）组**：中位数偏高，箱体偏上，说明死亡患者整体更年长。
- 这与医学常识一致：**年龄是癌症生存的重要预后因素**，年龄越大，死亡风险越高。
- 两组箱体有重叠，说明 Age 单独不足以完全区分生死，但确实有显著趋势。后续模块的统计检验会量化这种差异。

---

## 八、实际运行结果

下表是 4 个数值特征的实际统计结果（数据集已过滤掉 `Status.Vital` 缺失样本，剩余 209,758 条记录）：

| 特征 | 样本量 | 均值 | 中位数 | 标准差 | 偏度 | 峰度 | 分布类型 |
|---|---:|---:|---:|---:|---:|---:|---|
| `Age` | 209,443 | 63.79 | 65.00 | 16.55 | −0.683 | 0.639 | 轻度偏态 |
| `Code.Profession` | 193,966 | 218.34 | 0.00 | 289.05 | 0.940 | −0.581 | 轻度偏态 |
| `Code.of.Morphology` | 209,758 | 82,428.77 | 80,903.00 | 4,468.45 | 2.634 | 6.027 | 显著偏态 |
| `year` | 209,758 | 2,012.24 | 2,013.00 | 3.92 | −1.004 | 0.786 | 显著偏态 |

### 8.1 逐特征解读

#### Age（年龄）
- **样本量** 209,443（比总数少 315，说明有少量缺失）。
- **均值 63.79 < 中位数 65.00** → 左偏（负偏），与偏度 −0.683 一致。
- **偏度绝对值 0.683** 落在 [0.5, 1) 区间 → 轻度偏态。
- **峰度 0.639 > 0** → 比正态分布略尖，有轻微厚尾。
- **结论**：Age 接近正态但略左偏，多数模型可以直接使用，无需变换。

#### Code.Profession（职业代码）
- **样本量** 193,966（缺失约 15,792 条，缺失率 7.5%）。
- **中位数 = 0**，说明超过一半患者的职业代码是 0（很可能是"未填写"或"无业"的占位符）。
- **均值 218.34 >> 中位数 0** → 强烈右偏。
- **偏度 0.940** 接近 1 → 轻度偏态（临界显著偏态）。
- **⚠️ 数值陷阱！** 这是职业分类代码，不是连续数值。它的"均值 218.34"毫无意义——你不能说"平均职业是 218 号"。详见第八节。

#### Code.of.Morphology（形态学代码）
- **样本量** 209,758（无缺失）。
- **均值 82,428.77 > 中位数 80,903** → 右偏。
- **偏度 2.634** 远超 1 → **显著偏态**。
- **峰度 6.027** 远大于 0 → **尖峰厚尾**，存在极端值。
- **⚠️ 数值陷阱！** 这是 ICD-O（国际疾病分类-肿瘤学）的形态学代码，如 8090/3 表示"基底细胞癌，恶性"。代码本身是数字，但数字大小没有意义，8090 并不"大于"8080。

#### year（诊断年份）
- **样本量** 209,758（无缺失）。
- **均值 2012.24 < 中位数 2013** → 左偏。
- **偏度 −1.004** 刚好超过 1 → 显著偏态。
- **峰度 0.786** → 略尖。
- **解读**：数据集中在 2008–2017 年，但 2013 年后样本更多，所以左偏（左侧 2008 年附近的尾巴更长）。year 是离散时间变量，分布偏态本身不影响建模，但要注意**时间漂移**问题（未来年份的分布可能变化）。

---

## 九、关键概念：数值陷阱（Numeric Trap）

> **数值陷阱**：某些变量在数据表中被存储为数字（int 或 float），pandas 会自动把它们识别为数值型，但它们**本质上是类别变量**。对它们计算均值、标准差、偏度是**毫无意义**的，甚至会产生误导。

### 9.1 本数据集中的两个"陷阱"

#### (1) `Code.Profession`（职业代码）
- 这是一个**分类编码**，每个数字代表一种职业类别（如 0=未填写，100=医生，200=工程师……）。
- 数字大小没有顺序意义——"200 号职业"并不比"100 号职业"多两倍什么。
- **正确处理**：当作类别变量，用 `astype('category')` 或 One-Hot 编码。
- **错误处理**：当作连续数值喂给线性模型，模型会强行拟合一个"职业代码每增加 1，风险变化 X"的关系，这完全是胡说八道。

#### (2) `Code.of.Morphology`（形态学代码）
- 这是 **ICD-O-3**（International Classification of Diseases for Oncology, 3rd Edition）的形态学编码。
- 编码格式通常是 `M-XXXX/Y`，如 `M-8090/3`（基底细胞癌，恶性）。在数据表中常被存为数字 80903。
- 数字大小没有意义——8090 并不"大于"8080，它们只是不同的肿瘤类型标签。
- **正确处理**：当作类别变量。由于类别数很多（数百个），可以考虑：
  - 只保留高频类别，低频合并为"其他"。
  - 用目标编码（Target Encoding）。
  - 用嵌入层（Embedding，适用于深度学习）。

### 9.2 如何识别数值陷阱？

| 检查项 | 真正的数值变量 | 伪装的类别变量 |
|---|---|---|
| **单位** | 有明确单位（岁、kg、元） | 无单位，只是代码 |
| **大小关系** | 5 < 10 < 20 有物理意义 | 5 和 10 只是不同标签 |
| **小数** | 可以有小数（如 63.5 岁） | 通常是整数 |
| **均值意义** | "平均年龄 63.79 岁"有意义 | "平均职业 218 号"无意义 |
| **取值数量** | 通常很多（连续） | 可能有限（离散类别） |
| **领域知识** | 物理量、统计量 | 编码、ID、等级 |

> 💡 **小贴士**：在 EDA 阶段，**永远先看数据字典（data dictionary）和领域知识**，不要盲目相信 pandas 的 `select_dtypes`。一个负责任的数据科学家会把 `Code.Profession` 和 `Code.of.Morphology` 从数值特征列表中剔除，移到类别特征列表中。

### 9.3 修正后的特征分类

| 特征 | pandas 类型 | 真实类型 | 正确处理方式 |
|---|---|---|---|
| `Age` | float | 连续数值 | 标准化、可直接用于线性模型 |
| `year` | int | 离散数值/时间 | 当作序数或时间特征 |
| `Code.Profession` | float | **类别** | One-Hot / 目标编码 |
| `Code.of.Morphology` | float | **类别** | 高基数类别编码 |

---

## 十、常见问题（FAQ）

### Q1：偏度和峰度的阈值是怎么定的？为什么是 0.5 和 1？

A：这些是经验阈值，不是严格的统计标准。不同教材和领域略有差异：
- 本教程采用：`|skew|<0.5` 近似正态，`<1` 轻度偏态，`≥1` 显著偏态。
- 有的文献用更严格的标准：`|skew|<0.5` 才算对称。
- 有的用更宽松：`|skew|<2` 都算"可接受"。
关键不是阈值本身，而是**根据偏度决定后续操作**：显著偏态的特征可能需要对数变换或 Box-Cox 变换。

### Q2：为什么 `density=True` 后直方图纵轴超过 1？概率密度不是应该 ≤ 1 吗？

A：**概率密度（density）和概率（probability）是两回事！**
- 概率密度 $f(x)$ 可以大于 1，只要 $\int f(x)dx = 1$（总面积为 1）。
- 例如，均匀分布 U(0, 0.1) 的密度是 $f(x) = 10$（在 [0, 0.1] 区间内），但概率不会超过 1。
- 直方图用 `density=True` 后，纵轴是"每单位的概率"，柱子的高度 × 柱子的宽度 = 该区间的概率。

### Q3：Q-Q 图上的点不呈直线，但偏度接近 0，这是为什么？

A：偏度只衡量**三阶矩**（不对称性），Q-Q 图展示的是**整体分布形状**。可能的情况：
- 分布对称但**厚尾**（如 t 分布）：偏度 ≈ 0，但 Q-Q 图两端会偏离参考线。
- 分布对称但**双峰**：偏度 ≈ 0，但 Q-Q 图会呈"S 形"。
- 所以 Q-Q 图比单一偏度数值提供更多信息，两者应结合使用。

### Q4：`data.skew()` 和 `data.kurtosis()` 默认会自动跳过 NaN 吗？

A：**会的**。pandas 的大多数统计方法（`mean`、`std`、`skew`、`kurtosis`）默认 `skipna=True`，会自动忽略 NaN。但为了**精确控制样本量**（如 `len(data)`），本教程先 `dropna()` 再计算，这样 `len(data)` 反映的是有效样本数。

### Q5：箱线图的异常点和模块 4 的 IQR 离群值是一回事吗？

A：**基本是一回事**。箱线图默认用 1.5×IQR 规则标记异常点，这与模块 4 的 IQR 方法一致。但箱线图只是"标记"异常点，不做删除；模块 4 会进一步讨论如何处理（删除、盖帽、变换等）。

### Q6：为什么 `Code.Profession` 的中位数是 0？0 是有效值还是缺失？

A：在本数据集中，0 很可能是**"未填写"或"无业"的占位符**。这体现了真实数据的复杂性：缺失值不一定表现为 `NaN`，也可能用 0、−1、999 等特殊值表示。在 EDA 阶段要结合领域知识和数据字典判断。

---

## 十一、小贴士汇总

> 💡 **小贴士 1**：用 `df.select_dtypes(include=[np.number])` 筛选数值列后，**一定要人工复核**，剔除伪装成数值的类别变量（如 ID、编码）。

> 💡 **小贴士 2**：比较均值和中位数是判断偏态的"快速肉眼法"：均值 > 中位数 → 右偏；均值 < 中位数 → 左偏。

> 💡 **小贴士 3**：画直方图时，`bins` 不是越多越好。经验法则 `bins ≈ √n`，或用 Freedman-Diaconis 规则。pandas 的 `hist()` 默认用 30，可手动调整。

> 💡 **小贴士 4**：KDE 的带宽 `h` 对结果影响巨大。pandas 默认用 Scott 规则。如果曲线太平滑或太抖，可以手动指定 `bw_method`。

> 💡 **小贴士 5**：Q-Q 图抽样时，5000 个点足够。点太多会糊成一团，反而看不清偏离模式。

> 💡 **小贴士 6**：箱线图的 `patch_artist=True` 是自定义配色的关键开关，默认 `False` 时箱体无法填充颜色。

> 💡 **小贴士 7**：**永远不要**对类别编码（如职业代码、邮编）计算均值和标准差。在 EDA 报告中，这类变量应移到"类别特征"章节分析。

> 💡 **小贴士 8**：`random_state=42` 不是必须的，但保证结果可复现。在教程和论文中，固定随机种子是基本规范。

---

## 十二、本模块小结

本模块我们完成了：

1. **选取数值特征**：用 `select_dtypes` 按类型筛选，用列表推导式排除 `target` 和 `Patient.Code`。
2. **计算五大统计量**：均值、中位数、标准差、偏度、峰度，并理解了每个统计量的公式直觉和适用场景。
3. **判断分布类型**：基于偏度绝对值，分为近似正态、轻度偏态、显著偏态三类。
4. **绘制三种核心分布图**：
   - **直方图 + KDE + 正态曲线**：直观对比实际分布与正态分布。
   - **Q-Q 图**：通过分位数对分位数，精确判断正态性。
   - **分组箱线图**：按目标变量分组，观察特征在不同类别下的分布差异。
5. **识别数值陷阱**：`Code.Profession` 和 `Code.of.Morphology` 虽然是数值类型，但本质是类别变量，不能当作连续数值处理。

**下一步**：在模块 4 中，我们将基于本模块的分布分析，进一步用 IQR 和 Z-score 方法识别和处理离群值。

---

> 📚 **延伸阅读**：
> - pandas 官方文档：[`select_dtypes`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html)
> - scipy 官方文档：[`stats.probplot`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html)
> - Matplotlib 官方文档：[`boxplot`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
> - 维基百科：[核密度估计](https://en.wikipedia.org/wiki/Kernel_density_estimation)、[Q-Q 图](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
