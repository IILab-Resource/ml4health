# 模块 2：缺失值分析

## 学习目标

完成本模块后，你将能够：

1. **理解缺失值的概念**：知道什么是缺失值、为什么真实数据中会普遍存在缺失。
2. **掌握 Pandas 缺失值检测方法**：熟练使用 `df.isnull()`、`.sum()`、`.astype()` 等方法统计缺失情况。
3. **构建缺失值汇总表**：能用 `pd.DataFrame({...})` 把 Series 整理成可读的表格，并用 `sort_values` 排序。
4. **编写分档函数**：会用 `def` 定义函数、用 `if-elif-else` 实现分档逻辑，并通过 `.apply()` 应用到 Series。
5. **绘制水平条形图**：理解 `ax.barh()` 与 `ax.bar()` 的区别，掌握 `invert_yaxis()`、`ax.text()` 标签定位技巧。
6. **绘制缺失模式热力图**：理解 `sns.heatmap()` 的核心参数（`cmap`、`cbar_kws`、`linewidths`），知道为什么要抽样。
7. **绘制缺失相关性热力图**：理解 `missing_matrix.corr()` 的含义，掌握上三角掩码 `np.triu()` 的原理与用法。
8. **区分三种缺失机制**：能区分 MCAR / MAR / MNAR，并能结合本数据集给出合理示例。
9. **解读分析结果**：能从图中读出"哪些列缺失最严重"、"哪些列的缺失模式高度相关"等关键信息。

> **前置知识**：Python 基础语法、Pandas 的 DataFrame/Series 概念、Matplotlib 的 `subplots` 基本用法、NumPy 数组基础操作。

---

## 一、为什么必须做缺失值分析？

真实世界的医学数据几乎从不"干净"。在巴西癌症登记数据集中，我们已经过滤掉 `Status.Vital` 缺失的样本，剩余 **209,758 条记录、39 列**（38 个原始特征 + 1 个目标变量）。但即便如此，许多特征列仍然存在大量缺失。

缺失值分析要回答三个核心问题：

1. **缺失有多严重？** —— 哪些列缺失多到几乎不可用？
2. **缺失有没有规律？** —— 缺失是随机发生的，还是和某些变量有关？
3. **缺失的机制是什么？** —— 是 MCAR、MAR 还是 MNAR？这决定了我们能否直接删除、能否用均值填补。

> **重要概念**：缺失值分析不是"清理数据"的附属步骤，而是理解数据生成过程（Data Generating Process）的关键窗口。**缺失本身往往携带着信息**——例如"病情严重的患者更可能缺失教育信息"，这种规律本身就是建模时可以利用的信号。

---

## 二、计算各列缺失比例

### 2.1 代码

```python
# 计算各列缺失比例
missing_series = df.isnull().sum()
missing_pct = (missing_series / len(df)) * 100
```

### 2.2 逐行解析

#### `df.isnull()` —— 返回一个布尔 DataFrame

`df.isnull()`（等价于 `df.isna()`）会对 DataFrame 的**每一个单元格**进行判断：如果该单元格是 `NaN`、`None`、`NaT` 等"缺失值"，就填 `True`；否则填 `False`。

> **关键理解**：`df.isnull()` 返回的不是一个数字表，而是一个**与原 DataFrame 形状完全相同的布尔 DataFrame**（每个元素都是 `True` 或 `False`）。

举例：假设原数据是

| Name  | Age  | City  |
| ----- | ---- | ----- |
| Alice | 25   | NY    |
| Bob   | NaN  | LA    |
| NaN   | 30   | NaN   |

那么 `df.isnull()` 返回：

| Name  | Age   | City  |
| ----- | ----- | ----- |
| False | False | False |
| False | True  | False |
| True  | False | True  |

#### `.sum()` —— 默认按列求和

在 Pandas 中，`DataFrame.sum()` 默认参数是 `axis=0`，也就是**按列求和**（把每一列的所有行加起来）。

由于布尔值在求和时会被当作整数（`True = 1`，`False = 0`），所以 `df.isnull().sum()` 的结果就是：**每一列有多少个缺失值**。

返回结果是一个 **Series**，索引是列名，值是该列的缺失计数：

```
Age       1
City      1
Name      1
dtype: int64
```

> **小贴士**：如果你想按行统计每个样本缺失了多少个字段，用 `df.isnull().sum(axis=1)`。这在检测"几乎全空"的样本时很有用。

#### `missing_pct = (missing_series / len(df)) * 100`

- `len(df)` 返回 DataFrame 的行数（这里是 209,758）。
- `missing_series / len(df)` 是 Pandas 的**广播运算**：Series 的每个元素都除以同一个数。
- `* 100` 把比例转成百分比。

最终 `missing_pct` 也是一个 Series，索引与 `missing_series` 相同，值是每列的缺失百分比。

---

## 三、构建缺失值汇总表

### 3.1 代码

```python
missing_df = pd.DataFrame({
    'Column': missing_series.index,
    'Missing_Count': missing_series.values,
    'Missing_Pct': missing_pct.values
}).sort_values('Missing_Pct', ascending=False)
```

### 3.2 逐行解析

#### `pd.DataFrame({...})` —— 用字典构造 DataFrame

`pd.DataFrame()` 有多种构造方式，最常用的一种是**传入一个字典**：字典的 key 成为列名，value（通常是列表或数组）成为该列的数据。

这里我们传入三个 key-value 对：

| 字典 key          | 字典 value                    | 含义                |
| ---------------- | --------------------------- | ----------------- |
| `'Column'`       | `missing_series.index`      | 列名（Series 的索引）    |
| `'Missing_Count'`| `missing_series.values`     | 缺失计数（Series 的值数组） |
| `'Missing_Pct'`  | `missing_pct.values`        | 缺失百分比             |

> **为什么用 `.values`？** Series 的 `.values` 属性返回底层的 NumPy 数组（去掉索引）。如果我们直接传 Series，Pandas 会尝试按索引对齐，虽然结果通常一样，但显式用 `.values` 更清晰、避免对齐歧义。

#### `.sort_values('Missing_Pct', ascending=False)` —— 按缺失比例降序排列

`sort_values()` 是 DataFrame 的排序方法，关键参数：

| 参数           | 说明                                       |
| ------------ | ---------------------------------------- |
| `by`         | 排序依据的列名（或列名列表）。这里是 `'Missing_Pct'`。      |
| `ascending`  | 是否升序。`True`（默认）从小到大；`False` 从大到小。我们想要缺失最多的在最上面，所以用 `False`。 |
| `inplace`    | 是否原地修改。默认 `False`，返回新 DataFrame。         |
| `na_position`| 缺失值放在开头还是末尾，默认 `'last'`。                |

排序后，`missing_df` 的第一行就是缺失比例最高的列。

---

## 四、过滤出"有缺失"的列

### 4.1 代码

```python
# 只显示缺失 > 0 的列
missing_nonzero = missing_df[missing_df['Missing_Pct'] > 0]
```

### 4.2 原理：布尔索引（Boolean Indexing）

这是 Pandas 最常用的筛选技巧，分两步理解：

**第一步**：`missing_df['Missing_Pct'] > 0` 会返回一个**布尔 Series**（与 `missing_df` 行数相同），每一行是 `True` 或 `False`，表示该列的缺失比例是否大于 0。

```
0     True
1     True
2     True
...
36    False    # 这一列没有缺失
37    False
Name: Missing_Pct, dtype: bool
```

**第二步**：`missing_df[布尔Series]` 会**只保留布尔值为 `True` 的行**，丢弃 `False` 的行。这就像用一个"掩码"盖在 DataFrame 上，只露出我们想要的部分。

> **小贴士**：布尔索引可以组合多个条件，用 `&`（与）、`|`（或）、`~`（非）连接，每个条件必须用括号括起来。例如：
> ```python
> # 缺失比例在 5% 到 50% 之间的列
> missing_df[(missing_df['Missing_Pct'] >= 5) & (missing_df['Missing_Pct'] < 50)]
> ```

---

## 五、缺失值分档统计

### 5.1 代码

```python
# 缺失值分档统计
def missing_category(pct):
    if pct == 0:
        return "无缺失"
    elif pct < 5:
        return "轻度 (< 5%)"
    elif pct < 20:
        return "中度 (5%-20%)"
    elif pct < 50:
        return "高度 (20%-50%)"
    else:
        return "严重 (≥ 50%)"

missing_df['Category'] = missing_df['Missing_Pct'].apply(missing_category)
```

### 5.2 逐行解析

#### `def missing_category(pct):` —— 函数定义

`def` 是 Python 定义函数的关键字，语法是：

```python
def 函数名(参数1, 参数2, ...):
    函数体
    return 返回值
```

这里定义了一个名为 `missing_category` 的函数，接收一个参数 `pct`（百分比），返回一个字符串（分档标签）。

#### `if-elif-else` —— 多分支条件判断

Python 的条件判断结构：

- `if 条件1:` —— 如果条件1为真，执行这块代码。
- `elif 条件2:` —— 否则如果条件2为真，执行这块代码（elif = else if）。
- `else:` —— 以上条件都不满足时执行。

**关键理解**：`if-elif-else` 是**从上到下依次判断，一旦某个条件为真就执行对应代码块，然后跳出整个结构**，不会再检查后面的条件。所以条件的顺序很重要。

本函数的分档逻辑：

| 输入 `pct`       | 命中分支              | 返回值              |
| --------------- | ------------------ | ----------------- |
| `pct == 0`      | 第一个 `if`           | `"无缺失"`          |
| `0 < pct < 5`   | 第一个 `elif`         | `"轻度 (< 5%)"`    |
| `5 ≤ pct < 20`  | 第二个 `elif`         | `"中度 (5%-20%)"`  |
| `20 ≤ pct < 50` | 第三个 `elif`         | `"高度 (20%-50%)"` |
| `pct ≥ 50`      | `else`             | `"严重 (≥ 50%)"`   |

> **常见问题：为什么用 `elif` 而不是多个独立的 `if`？**
>
> 如果写成多个独立的 `if`，每个条件都会被独立判断。例如 `pct = 3` 时，`pct < 5` 为真，`pct < 20` 也为真，会导致逻辑混乱。`elif` 保证了"互斥"——只有一个分支会被执行。

#### `.apply(missing_category)` —— 把函数应用到 Series 的每个元素

`Series.apply(func)` 是 Pandas 的核心方法之一，它的作用是：**对 Series 中的每一个元素，调用 `func` 函数，并把所有返回值组成一个新的 Series**。

可以理解为等价的循环：

```python
# apply 的等价写法（伪代码）
result = []
for value in missing_df['Missing_Pct']:
    result.append(missing_category(value))
missing_df['Category'] = result
```

但 `apply` 用的是底层 C 实现，比显式循环快得多，而且代码更简洁。

> **小贴士**：`apply` 也可以用在 DataFrame 上（`df.apply(func, axis=0/1)`），`axis=0` 对每列应用函数，`axis=1` 对每行应用函数。

---

## 六、图 2a：缺失比例条形图（Top 30）

### 6.1 代码

```python
# 图 2a: 缺失比例条形图 (Top 30)
plot_missing = missing_nonzero.head(30)
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(range(len(plot_missing)), plot_missing['Missing_Pct'].values,
               color='#3498db', edgecolor='white')
ax.set_yticks(range(len(plot_missing)))
ax.set_yticklabels(plot_missing['Column'].values, fontsize=9)
ax.set_xlabel('Missing Percentage (%)', fontsize=12)
ax.set_title('Top 30 Features by Missing Value Percentage', fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 添加数值标签
for bar, pct in zip(bars, plot_missing['Missing_Pct'].values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=8)
```

### 6.2 逐行解析

#### `ax.barh()` vs `ax.bar()` —— 水平条形图 vs 垂直条形图

| 方法        | 方向   | x 轴含义   | y 轴含义   | 适用场景              |
| --------- | ---- | ------- | ------- | ----------------- |
| `ax.bar()`  | 垂直   | 类别（柱子横向排列） | 数值（柱子高度） | 类别较少、类别名短         |
| `ax.barh()` | 水平   | 数值（柱子长度） | 类别（柱子纵向排列） | **类别多、类别名长**（本例 30 个长列名） |

**为什么本例用 `barh`？** 因为我们要展示 30 个特征，且列名（如 `Code.of.Disease.Adult.Young.`）很长。如果用垂直柱状图，x 轴标签会挤在一起或需要旋转 90°，可读性很差。水平条形图让类别名在 y 轴水平排列，天然适合这种场景。

`barh` 的参数：

| 参数                          | 说明                          |
| --------------------------- | --------------------------- |
| 第 1 个位置参数（`y`）              | 柱子的 y 坐标位置，这里是 `range(30)`，即 0,1,2,...,29 |
| 第 2 个位置参数（`width`）          | 柱子的长度（数值），这里是缺失百分比          |
| `color`                     | 柱子填充颜色，`'#3498db'` 是十六进制颜色码（蓝色） |
| `edgecolor`                 | 柱子边框颜色，`'white'` 让柱子之间有清晰分隔 |

返回值 `bars` 是一个 `BarContainer` 对象，包含所有柱子的引用，后面添加标签时要用到。

#### `ax.invert_yaxis()` —— 反转 y 轴

默认情况下，Matplotlib 的 y 轴是**从下到上递增**的（0 在底部，29 在顶部）。这意味着 `barh` 画的第一个柱子（y=0）会在图的**底部**。

但我们习惯从上往下读，希望缺失最多的列（排序后第一行）显示在**顶部**。`ax.invert_yaxis()` 把 y 轴反转，让 0 在顶部、29 在底部，这样第一个条目就显示在最上面。

> **常见问题**：如果不加 `invert_yaxis()`，图会"倒过来"——缺失最多的列在底部，看起来很别扭。这是画水平条形图时几乎必加的一行。

#### `ax.spines['top'].set_visible(False)` —— 隐藏顶部和右侧边框

Matplotlib 默认会给子图加一个"方框"（四条边：top、bottom、left、right）。现代数据可视化倾向于"简洁"风格，通常会隐藏 top 和 right 两条边，只保留 left 和 bottom 形成一个 "L" 形坐标轴。

```python
ax.spines['top'].set_visible(False)    # 隐藏顶部边框
ax.spines['right'].set_visible(False)  # 隐藏右侧边框
```

#### `ax.text()` —— 在条形上添加数值标签

这是最需要理解的细节之一。`ax.text(x, y, s)` 在坐标 `(x, y)` 处绘制字符串 `s`。

在循环中：

```python
for bar, pct in zip(bars, plot_missing['Missing_Pct'].values):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{pct:.1f}%', va='center', fontsize=8)
```

- `zip(bars, pct_values)` 把柱子对象和对应的百分比值"配对"成元组，方便同时遍历。
- `bar.get_width()` —— 返回柱子的长度（即缺失百分比数值）。
- `bar.get_y()` —— 返回柱子左下角的 y 坐标。
- `bar.get_height()` —— 返回柱子的"高度"（在水平条形图中，这是柱子的粗细程度）。

**坐标计算逻辑**：

| 坐标                                  | 计算                          | 含义              |
| ----------------------------------- | --------------------------- | --------------- |
| `x`                                 | `bar.get_width() + 0.3`     | 柱子末端再往右偏移 0.3 个单位，让标签不贴着柱子 |
| `y`                                 | `bar.get_y() + bar.get_height() / 2` | 柱子的**垂直中心**（柱子起点 + 一半高度） |
| `s`                                 | `f'{pct:.1f}%'`             | 格式化字符串，保留 1 位小数 |

`va='center'` 是 vertical alignment（垂直对齐）参数，让文字的垂直中心对齐到指定的 y 坐标。如果不加，文字会默认底部对齐，看起来偏上。

> **小贴士**：`f'{pct:.1f}%'` 是 f-string 格式化语法，`:.1f` 表示"浮点数，保留 1 位小数"。例如 `pct = 77.15` 会渲染成 `"77.2%"`。

### 6.3 结果图

![缺失比例图](../img/02a_missing_percentage.png)

从图中可以清晰看到：

- **`Child.Illness.Code` 和 `Child.Illness.Description`** 缺失率最高，达到 **98.69%**——这两个字段是"儿童疾病"相关，绝大多数成年癌症患者自然没有这些信息，所以几乎全空。
- **`TNM` 缺失 77.15%**、**`Extension` 缺失 60.58%**——这两个是肿瘤临床分期关键字段，缺失率偏高，说明很多病例的分期记录不完整。
- **`Age`、`Date.of.Birth`** 缺失率很低（< 0.3%），属于基础人口学信息，记录比较完整。

---

## 七、图 2b：缺失模式热力图

### 7.1 代码

```python
# 图 2b: 缺失矩阵热力图 (选取关键列)
key_cols_candidates = ['Age', 'Gender', 'Raca.Color', 'Degree.of.Education',
                       'State.Civil', 'TNM', 'Extension', 'Laterality',
                       'Diagnostic.means', 'Type.of.Death', 'Distant.metastasis']
key_cols = [c for c in key_cols_candidates if c in df.columns]
missing_matrix = df[key_cols].isnull().astype(int)

# 如果数据太大，抽样显示
sample_size = min(5000, len(df))
if len(df) > sample_size:
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    missing_matrix_sample = missing_matrix.iloc[sample_idx]
else:
    missing_matrix_sample = missing_matrix

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(missing_matrix_sample.T, cmap=['#2ecc71', '#e74c3c'],
            cbar_kws={'label': 'Missing (1) / Present (0)', 'ticks': [0, 1]},
            ax=ax, linewidths=0)
```

### 7.2 逐行解析

#### 列表推导式：`key_cols = [c for c in key_cols_candidates if c in df.columns]`

这是一种 Pythonic 的写法，等价于：

```python
key_cols = []
for c in key_cols_candidates:
    if c in df.columns:
        key_cols.append(c)
```

作用是：从候选列名列表中，**只保留实际存在于 DataFrame 中的列**，避免因列名拼写错误或数据版本不同而报错。

#### `df[key_cols].isnull().astype(int)` —— 为什么要把布尔转成 int？

分两步：

1. `df[key_cols].isnull()` 返回一个**布尔 DataFrame**（`True`/`False`）。
2. `.astype(int)` 把布尔值转换成整数：`True → 1`，`False → 0`。

**为什么要转换？** 因为 `sns.heatmap()` 需要的是**数值矩阵**——它根据每个单元格的数值映射到颜色。如果传入布尔值，虽然 Seaborn 内部可能也能处理，但显式转成 `int` 更规范、更可控，也方便后续计算相关性（`corr()` 要求数值类型）。

转换后，`1` 表示"缺失"，`0` 表示"存在"，这样热力图就能用颜色直观区分。

#### `np.random.seed(42)` 和 `np.random.choice()` —— 随机抽样

```python
sample_size = min(5000, len(df))
if len(df) > sample_size:
    np.random.seed(42)
    sample_idx = np.random.choice(len(df), sample_size, replace=False)
    missing_matrix_sample = missing_matrix.iloc[sample_idx]
```

**为什么要抽样？** 我们有 209,758 行数据。如果全部画在热力图上，每个像素代表一行，图会非常巨大且密集，根本看不出模式。抽样 5000 行既能保留整体的缺失模式，又能让图清晰可读。

**`np.random.seed(42)`** —— 设置随机数种子。计算机的"随机数"其实是伪随机（由算法生成），给定相同的种子，每次运行生成的随机序列都**完全相同**。这样：

- 结果**可复现**：别人运行你的代码能得到一样的图。
- 调试方便：如果图有问题，每次都能复现同样的样本。

> **为什么是 42？** 这是一个程序员圈的"梗"——来自《银河系漫游指南》中"生命、宇宙及一切的终极答案是 42"。用 42 只是约定俗成，用 0、1、123 都可以，关键是**固定一个值**。

**`np.random.choice(a, size, replace)`** —— 从数组中随机抽样：

| 参数        | 说明                          |
| --------- | --------------------------- |
| `a`       | 抽样的总体。这里是 `len(df)`，等价于从 `0, 1, 2, ..., 209757` 中抽 |
| `size`    | 抽多少个。这里是 `sample_size = 5000` |
| `replace` | 是否有放回。`False` 表示**无放回抽样**（同一个样本不会被抽两次） |

返回的是 5000 个随机索引，然后用 `missing_matrix.iloc[sample_idx]` 按这些索引取行，得到抽样后的子矩阵。

#### `sns.heatmap()` —— 热力图核心函数

```python
sns.heatmap(missing_matrix_sample.T, cmap=['#2ecc71', '#e74c3c'],
            cbar_kws={'label': 'Missing (1) / Present (0)', 'ticks': [0, 1]},
            ax=ax, linewidths=0)
```

**`.T`** —— 转置矩阵。原矩阵是"行=样本，列=特征"，转置后变成"行=特征，列=样本"，这样特征名显示在 y 轴（垂直排列，更易读），样本在 x 轴。

**关键参数详解**：

| 参数           | 说明                                       |
| ------------ | ---------------------------------------- |
| `data`       | 第一个位置参数，要绘制的二维数值矩阵                       |
| `cmap`       | 颜色映射。**传列表的含义**：当数据只有 0 和 1 两个离散值时，传 `['#2ecc71', '#e74c3c']`（绿色、红色）表示 0 用第一种颜色（绿=存在），1 用第二种颜色（红=缺失）。这种"红绿配色"非常直观——红色一眼就能看出缺失区域 |
| `cbar_kws`   | colorbar kwargs 的缩写，是一个字典，用来配置右侧的颜色条。`'label'` 设置颜色条的标题，`'ticks'` 设置颜色条上显示的刻度位置（这里只显示 0 和 1 两个刻度） |
| `ax`         | 在哪个子图上绘制                                  |
| `linewidths` | 单元格之间的间隔线宽度。`0` 表示无间隔，让色块紧密相连，适合大规模矩阵      |

> **小贴士**：`cmap` 传列表的方式只适合离散值（如 0/1）。如果是连续值，应该传 colormap 名称（如 `'Blues'`、`'viridis'`）。

### 7.3 结果图

![缺失模式热力图](../img/02b_missing_pattern.png)

**如何解读这张图？**

- **每一行**是一个特征，**每一列**是一个样本（共 5000 个抽样样本）。
- **绿色 = 数据存在（0）**，**红色 = 数据缺失（1）**。
- `Age`、`Gender` 几乎全绿 → 记录非常完整。
- `Distant.metastasis`、`TNM`、`Extension`、`Type.of.Death` 有大片红色 → 缺失严重。
- 观察"红色条纹是否对齐"：如果多个特征的红色出现在**同一批样本**上，说明这些特征的缺失是**相关**的（可能是同一类患者整体记录不全）。

---

## 八、图 2c：缺失值相关性热力图

### 8.1 代码

```python
# 图 2c: 缺失值相关性热力图
missing_corr = missing_matrix.corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(missing_corr, dtype=bool), k=1)
sns.heatmap(missing_corr, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', square=True,
            linewidths=0.5, cbar_kws={'label': 'Correlation'},
            ax=ax, annot_kws={'fontsize': 8})
```

### 8.1 逐行解析

#### `missing_matrix.corr()` —— 计算的是什么相关性？

`missing_matrix` 是一个 0/1 矩阵（1=缺失，0=存在），`corr()` 计算**列与列之间的皮尔逊相关系数**。

**关键理解**：这里计算的不是"原始特征值"之间的相关性，而是**"缺失模式"之间的相关性**——即"特征 A 缺失"这件事，与"特征 B 缺失"这件事，是否倾向于同时发生。

- 相关系数 **接近 +1**：两个特征倾向于**同时缺失**（同生共死）。
- 相关系数 **接近 -1**：一个特征缺失时，另一个倾向于**不缺失**（此消彼长）。
- 相关系数 **接近 0**：两个特征的缺失**互不影响**。

> **重要概念**：缺失相关性是判断缺失机制的重要线索。如果多个临床字段（如 TNM、Extension、Laterality）的缺失高度正相关，往往说明这些字段来自同一套检查流程——没做这套检查的患者，这些字段会一起缺失。

#### `np.triu(np.ones_like(missing_corr, dtype=bool), k=1)` —— 生成上三角掩码

这是 NumPy 的一个经典技巧，分两步理解：

**第一步：`np.ones_like(missing_corr, dtype=bool)`**

生成一个与 `missing_corr` 形状相同的全 `True` 矩阵（因为 `dtype=bool`，所以 1 变成 `True`）。假设 `missing_corr` 是 11×11，这就是一个 11×11 的全 True 矩阵。

**第二步：`np.triu(..., k=1)`**

`triu` 是 "triangle upper" 的缩写，返回矩阵的**上三角部分**：

- `k=0`：保留主对角线及以上的部分（主对角线 + 上三角）。
- `k=1`：保留**主对角线以上**的部分（**不包括主对角线**），主对角线及以下全变 `False`。
- `k=2`：从主对角线再往上偏移一行。

举例（4×4 矩阵，`k=1`）：

```
原矩阵（全 True）        triu(k=1) 后
T T T T                F T T T
T T T T    →           F F T T
T T T T                F F F T
T T T T                F F F F
```

**为什么要生成这个掩码？** 因为相关性矩阵是**对称的**（`corr(A,B) == corr(B,A)`），而且主对角线永远是 1（`corr(A,A) == 1`，没意义）。如果全画出来，一半的信息是冗余的。用上三角掩码可以**只保留下三角（含或不含对角线）**，让图更简洁。

#### `mask=mask` —— 如何隐藏上三角？

`sns.heatmap()` 的 `mask` 参数接收一个**布尔矩阵**，规则是：**`True` 的位置会被隐藏（不绘制）**。

我们生成的 `mask` 中，上三角是 `True`，所以上三角被隐藏；下三角是 `False`，所以下三角正常显示。最终图上只显示下三角部分。

#### `sns.heatmap()` 的其他参数详解

```python
sns.heatmap(missing_corr, mask=mask, cmap='RdBu_r', center=0,
            annot=True, fmt='.2f', square=True,
            linewidths=0.5, cbar_kws={'label': 'Correlation'},
            ax=ax, annot_kws={'fontsize': 8})
```

| 参数            | 说明                                       |
| ------------- | ---------------------------------------- |
| `cmap`        | `'RdBu_r'` = Red-Blue-reversed。`RdBu` 是"红-白-蓝"渐变，加 `_r` 表示反转，变成"蓝-白-红"。这样**负相关=蓝色，0=白色，正相关=红色**，非常直观 |
| `center`      | 颜色映射的中心值。`center=0` 让 0 对应白色（中间色），负值偏蓝、正值偏红。如果不设 `center`，颜色会以数据的中位数为中心，可能导致 0 不是白色，误导解读 |
| `annot`       | annotation 的缩写，`True` 表示在每个格子里**标注数值**    |
| `fmt`         | format string，`'.2f'` 表示保留 2 位小数。只有 `annot=True` 时才生效 |
| `square`      | `True` 让每个单元格是正方形（而不是根据数据范围自动拉伸）。相关性矩阵通常用正方形更美观 |
| `linewidths`  | 单元格之间的间隔线宽度，`0.5` 让格子之间有细线分隔，更易读          |
| `cbar_kws`    | 颜色条配置，`'label'` 设置颜色条标题                  |
| `annot_kws`   | annotation kwargs，标注文字的样式。`{'fontsize': 8}` 设置字号为 8，避免格子装不下 |

> **小贴士**：`cmap='RdBu_r'` + `center=0` 是绘制相关性矩阵的"黄金组合"——红色代表正相关、蓝色代表负相关、白色代表无相关，一目了然。记住这个搭配，几乎所有相关性热力图都适用。

### 8.3 结果图

![缺失相关性热力图](../img/02c_missing_correlation.png)

**如何解读这张图？**

- 颜色越**红**，两个特征的缺失越倾向于**同时发生**。
- 颜色越**蓝**，一个缺失时另一个越倾向于**不缺失**。
- 颜色越**白**，两个特征的缺失**互不相关**。

典型观察：`TNM`、`Extension`、`Laterality` 这些临床分期相关字段，往往呈现较强的正相关——它们来自同一套肿瘤分期评估流程，没做评估的患者这些字段会一起缺失。

---

## 九、实际运行结果

### 9.1 缺失概况

```
存在缺失值的列数: 24 / 39
完全完整的列数:   15 / 39
```

> **关键数字**：**24/39 列存在缺失**（约 62% 的列不完整），其中 **10 列缺失 ≥ 50%**。这说明本数据集的缺失问题相当严重，建模时必须谨慎处理。

### 9.2 各列缺失比例（按严重程度分档）

| 程度             | 列数 | 代表列（缺失率）                                  |
| -------------- | -- | ----------------------------------------- |
| **严重 (≥ 50%)** | 10 | `Child.Illness.Code` (98.69%)、`TNM` (77.15%)、`Extension` (60.58%) |
| **高度 (20%-50%)** | 5  | `Laterality` (49.65%)、`Type.of.Death` (42.58%)、`Degree.of.Education` (32.86%) |
| **中度 (5%-20%)** | 3  | `Raca.Color` (15.22%)、`Name.Occupation` (7.53%) |
| **轻度 (< 5%)**  | 6  | `Nationality` (3.03%)、`Age` (0.15%)       |
| **无缺失**         | 15 | `Gender`、`year`、`Status.Vital` 等          |

### 9.3 重点字段缺失率

| 字段                  | 缺失率     | 临床意义               | 处理建议                  |
| ------------------- | ------- | ------------------ | --------------------- |
| `TNM`               | 77.15%  | 肿瘤分期（T/N/M 组合），核心预后因素 | 缺失太多，直接删除或建"缺失"指示变量   |
| `Extension`         | 60.58%  | 肿瘤扩展程度             | 同上                    |
| `Distant.metastasis`| 94.97%  | 远处转移               | 几乎全空，建议删除或仅作"有无记录"指示 |
| `Age`               | 0.15%   | 年龄                 | 缺失极少，可直接删除缺失行或中位数填补  |
| `Gender`            | 0%      | 性别                 | 完整，无需处理              |

> **重要发现**：`TNM`（77.15%）和 `Extension`（60.58%）是肿瘤学中最重要的两个分期字段，但缺失率都偏高。这提醒我们：**临床价值高的字段不一定数据质量好**，建模时要在"信息量"和"可用样本量"之间权衡。

---

## 十、三种缺失机制：MCAR / MAR / MNAR

缺失值的处理策略**高度依赖于缺失机制**。统计学家把缺失机制分为三类，理解它们是做好缺失值处理的前提。

### 10.1 MCAR（Missing Completely At Random，完全随机缺失）

> **定义**：缺失的发生与**任何变量**（无论已观测还是未观测）都**无关**，完全是随机的。

**特点**：

- 缺失就像"掷骰子"决定的，没有任何规律。
- 在这种情况下，直接删除缺失样本（listwise deletion）不会引入偏差。
- 现实中**很少见**——真实数据几乎不会是严格的 MCAR。

**本数据集示例**：`Age` 缺失 0.15%（315 条），可能是极少数录入时手滑漏填，接近 MCAR。

### 10.2 MAR（Missing At Random，条件随机缺失）

> **定义**：缺失的发生与**已观测的其他变量**有关，但与**缺失值本身**无关。

**注意**："At Random" 这个名字很容易误导——MAR **不是**完全随机的！它的意思是"在给定已观测变量的条件下，缺失是随机的"。

**特点**：

- 缺失有规律，但规律可以被已观测变量"解释"。
- 可以用已观测变量来预测缺失值（如多重插补 Multiple Imputation）。
- 现实中最常见的假设，也是大多数统计方法的默认假设。

**本数据集示例**：`Degree.of.Education`（教育程度）缺失 32.86%。高龄患者可能更倾向于缺失教育信息（年代久远、记录不全）——如果"年龄"是已观测的，那么教育程度的缺失就属于 MAR。

### 10.3 MNAR（Missing Not At Random，非随机缺失）

> **定义**：缺失的发生与**缺失值本身**有关——也就是说，"缺失"这件事取决于"如果填了会是什么值"。

**特点**：

- 最难处理的一类，因为缺失原因本身不可观测。
- 无法用已观测数据完全"解释"缺失，任何处理都带有假设。
- 处理不当会引入严重偏差。

**本数据集示例**：`Type.of.Death`（死亡类型）缺失 42.58%。很可能存活患者（VIVO）根本不会有"死亡类型"信息——缺失本身就和生死状态相关。这是典型的 MNAR。

### 10.4 三种机制对比

| 机制   | 缺失与什么有关        | 能否直接删除 | 处理方法               |
| ---- | -------------- | ------ | ------------------ |
| MCAR | 与任何变量都无关       | 可以，无偏差 | 直接删除 / 均值填补 / 多重插补 |
| MAR  | 与已观测变量有关       | 有偏差    | 多重插补 / 基于模型填补      |
| MNAR | 与缺失值本身有关       | 严重偏差   | 需要敏感性分析，无法完全解决     |

> **重要概念**：现实中很难严格区分 MAR 和 MNAR，因为我们看不到缺失的值本身。实践中通常**假设 MAR**（这是最可操作的假设），并在讨论中说明这一假设的局限性。

---

## 十一、为什么医学数据经常出现缺失？

结合本数据集，医学数据高缺失的常见原因：

1. **数据来源多样**：不同医院、不同医生记录习惯不同，字段覆盖不一致。
2. **检查项目因病情而异**：并非所有患者都做所有检查。例如早期患者可能不做 `Distant.metastasis`（远处转移）评估。
3. **回顾性数据收集**：历史记录不完整，早期病例的字段比新病例少。
4. **患者隐私考量**：部分敏感字段（如职业、收入）被有意省略。
5. **数据录入错误或遗漏**：人工录入难免出错。

> **小贴士**：在分析缺失时，**不要急于"填补"或"删除"**。先问自己：这个字段为什么缺失？缺失本身是否携带着信息？有时候"是否缺失"本身就是一个有价值的特征（缺失指示变量 missing indicator）。

---

## 十二、小贴士与常见问题

### 小贴士

1. **先统计，再处理**：永远先看缺失比例和模式，再决定处理策略。盲目填补会掩盖数据的问题。
2. **`isnull()` 和 `isna()` 等价**：Pandas 中两者完全一样，`isna()` 是后来加的别名，用哪个都行。
3. **抽样时固定种子**：任何涉及随机的过程（抽样、打乱、初始化）都应 `np.random.seed()`，保证可复现。
4. **热力图配色有讲究**：
   - 0/1 离散值 → 传颜色列表 `['#2ecc71', '#e74c3c']`。
   - 相关系数 → `'RdBu_r'` + `center=0`，红正蓝负白为零。
5. **相关性矩阵只画一半**：用 `np.triu(..., k=1)` 生成掩码，避免冗余信息。
6. **缺失指示变量**：对于 MNAR/MAR 的字段，可以新建一列 `XXX_is_missing`（0/1），让模型自己学习"缺失"的信号。

### 常见问题

**Q1：`df.isnull().sum()` 和 `df.isna().sum()` 有区别吗？**

A：没有区别。两者在 Pandas 中是完全相同的操作，`isna` 是 `isnull` 的别名。习惯上用哪个都可以，社区里两种都很常见。

**Q2：为什么我用 `barh` 画的图，第一个柱子在底部？**

A：因为 Matplotlib 的 y 轴默认从下到上递增。加一行 `ax.invert_yaxis()` 反转 y 轴即可让第一个柱子显示在顶部。

**Q3：`sns.heatmap` 的 `cmap` 传列表和传字符串有什么区别？**

A：传字符串（如 `'Blues'`）是传一个 colormap 对象的名字，适合连续值，会根据数值大小在颜色渐变中映射。传列表（如 `['#2ecc71', '#e74c3c']`）适合离散值（如 0/1），列表的第 i 个颜色对应数值 i。本例缺失矩阵只有 0 和 1，所以用列表更直观。

**Q4：`np.random.seed(42)` 一定要写吗？不写会怎样？**

A：不写也能运行，但每次运行结果会不同（因为随机数不同），导致图不可复现。在教程、论文、生产代码中，**强烈建议固定种子**，方便他人复现和调试。42 只是约定俗成，用任何整数都可以。

**Q5：`missing_matrix.corr()` 计算的是特征值的相关性吗？**

A：**不是**。`missing_matrix` 是 0/1 矩阵（1=缺失，0=存在），所以 `corr()` 计算的是**缺失模式**之间的相关性——即"特征 A 缺失"和"特征 B 缺失"是否倾向于同时发生。这与"特征 A 的值"和"特征 B 的值"之间的相关性是两回事。

**Q6：为什么相关性热力图要隐藏上三角？**

A：因为相关性矩阵是对称的（`corr(A,B) == corr(B,A)`），上三角和下三角信息完全重复。隐藏一半可以让图更简洁，避免视觉冗余。主对角线全是 1（自相关），也没有信息量，所以一起隐藏。

**Q7：缺失率超过多少应该直接删除列？**

A：没有绝对标准，但常见经验法则：
- 缺失 **< 5%**：可直接删除缺失行，或简单填补。
- 缺失 **5%-50%**：建议用模型填补（KNN、多重插补），并考虑加缺失指示变量。
- 缺失 **> 50%**：通常建议删除列，或仅保留"是否缺失"作为指示变量（因为值本身信息量太少）。

> **重要提醒**：删除列之前，先想清楚这个字段对业务是否关键。如果某个字段缺失 80% 但对预测至关重要（如 `TNM` 分期），可以考虑：① 保留并加缺失指示变量；② 把"缺失"作为单独的类别；③ 用领域知识辅助填补。

---

## 十三、本模块小结

本模块我们完成了以下工作：

1. **统计缺失**：用 `df.isnull().sum()` 计算每列缺失数，转换为百分比，整理成排序表格。
2. **分档归类**：用 `def` + `if-elif-else` 定义分档函数，用 `.apply()` 应用到 Series，把缺失程度分为"无/轻度/中度/高度/严重"五档。
3. **可视化**：
   - 图 2a：水平条形图展示 Top 30 缺失列，用 `ax.text()` 添加数值标签。
   - 图 2b：缺失模式热力图，抽样 5000 行，用红绿色块直观显示缺失分布。
   - 图 2c：缺失相关性热力图，用上三角掩码去除冗余，用 RdBu_r 配色显示正负相关。
4. **机制讨论**：区分 MCAR/MAR/MNAR 三种缺失机制，结合本数据集给出示例。
5. **关键发现**：24/39 列存在缺失，10 列缺失 ≥ 50%，`TNM` (77.15%)、`Extension` (60.58%) 等核心临床字段缺失严重。

> **下一步**：在了解了缺失情况后，模块 3 将进入**分布分析**——用直方图、KDE、Q-Q 图研究数值特征的分布形态，为后续的特征工程和建模选择提供依据。
