# 模块 2：数值特征分析 — T 检验与 Mann-Whitney U 检验

> **核心思想**：当我们想回答"年龄在存活组和死亡组之间有没有差异？"这个问题时，不能只看两组均值谁大谁小——因为均值差异可能是抽样波动造成的。统计检验（hypothesis testing）给我们提供了一个**量化判断标准**：在原假设（两组无差异）成立的前提下，观察到当前差异（或更极端差异）的概率有多大？这个概率就是 **p 值**。本模块将带你完整走通数值特征的单因素分析流程，并教你如何选择正确的检验方法、如何计算效应量、如何用 Bonferroni 校正应对多重比较问题。

---

## 学习目标

完成本模块后，你将能够：

1. **理解假设检验的基本框架**：原假设、备择假设、p 值、显著性水平 α 的含义。
2. **掌握数据预处理细节**：`dropna()` 的成对删除机制、布尔索引分组、最小样本量保护。
3. **理解正态性判断如何决定检验方法的选择**：正态 → T 检验，非正态 → Mann-Whitney U 检验。
4. **详解 T 检验分支**：Levene 方差齐性检验、`equal_var` 参数、独立样本 t 检验、Cohen's d 效应量及其解读。
5. **详解 Mann-Whitney U 检验分支**：秩和检验原理、`alternative` 参数的三种取值、Rank-biserial r 效应量及其解读。
6. **理解 Bonferroni 校正**：为什么多重比较需要校正，α/m 公式的含义。
7. **读懂结果字典的数据结构**：每一行 `numerical_results.append({...})` 存了哪些字段。
8. **批判性解读实际结果**：效应量比 p 值更重要；统计显著 ≠ 实际重要。

---

## 一、背景回顾

我们使用的是**巴西癌症登记数据**（Brazilian Cancer Registry）。在模块 0 中已经过滤掉 `Status.Vital`（生存状态）缺失的样本，剩余 **209,758 条**记录。

目标变量 `target` 的定义：

```python
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

- `target = 1` → VIVO（存活）
- `target = 0` → MORTO（死亡）

本模块要分析的 4 个数值特征：

| 特征名 | 含义 | 备注 |
|---|---|---|
| `Age` | 患者诊断时的年龄（岁） | 真正的连续型数值 |
| `Code.Profession` | 职业分类代码 | 缺失较多（本质是类别变量） |
| `Code.of.Morphology` | 肿瘤形态学代码（ICD-O-3） | 完整（本质是类别变量） |
| `year` | 诊断年份 | 离散型时间变量 |

> ⚠️ **重要提醒**：本模块将这 4 个特征统一当作"数值特征"做组间差异检验。但请注意，`Code.Profession` 和 `Code.of.Morphology` 本质上是**类别编码**，它们的"均值"在数学上有意义、在医学解释上需要谨慎。我们会在结果解读部分详细讨论这一点。

---

## 二、整体代码框架

数值特征分析的整体逻辑是：遍历每个数值特征 → 取出该特征和目标变量并删除缺失 → 按目标分组 → 判断正态性 → 选择 T 检验或 Mann-Whitney U 检验 → 计算效应量 → 记录结果。

```python
numerical_results = []

for col in numeric_features:
    data = df_model[[col, 'target']].dropna()
    if len(data) < 30:
        continue

    group_vivo = data.loc[data['target'] == 1, col]
    group_morto = data.loc[data['target'] == 0, col]

    if len(group_vivo) < 5 or len(group_morto) < 5:
        continue

    is_normal, norm_note = check_normality_practical(data[col])

    if is_normal:
        # ... T 检验分支 ...
    else:
        # ... Mann-Whitney U 检验分支 ...

    numerical_results.append({...})
```

下面我们逐行拆解。

---

## 三、数据预处理：取数、删除缺失、分组

### 3.1 `df_model[[col, 'target']].dropna()` —— 成对删除缺失值

```python
data = df_model[[col, 'target']].dropna()
```

**逐层拆解：**

1. `df_model[[col, 'target']]`：从 `df_model` 中**同时取出两列**——当前要分析的特征 `col` 和目标变量 `target`。注意外层是**双中括号 `[[...]]`**，这意味着返回的是一个 **DataFrame**（二维表），而不是 Series（一维序列）。
   - 单中括号 `df_model[col]` → 返回 Series
   - 双中括号 `df_model[[col, 'target']]` → 返回 DataFrame

2. `.dropna()`：删除**任何含有缺失值（NaN）的行**。这是关键点——它是**成对删除（pairwise deletion / listwise deletion）**，而不是只删一列的缺失。

> 💡 **关键概念：成对删除 vs 单列删除**
>
> 假设 `Age` 列在第 3 行缺失，`target` 列在第 7 行缺失：
>
> - `df_model['Age'].dropna()`：只删除 `Age` 缺失的行（第 3 行），保留第 7 行。
> - `df_model[['Age', 'target']].dropna()`：**只要 `Age` 或 `target` 任一缺失，就删除该行**。所以第 3 行和第 7 行都会被删除。
>
> **为什么必须成对删除？** 因为统计检验要求每组中的每个观测值都**同时**有特征值和标签值。如果某行 `Age` 有值但 `target` 缺失，这行无法被分到 VIVO 或 MORTO 组，对检验没有贡献，必须删除。反之亦然。

**实际效果**：以 `Age` 为例，原始 209,758 条记录中有 315 条 `Age` 缺失，所以 `data` 剩余 209,443 条。这就是为什么 `Age` 的样本量是 209,443，而 `Code.of.Morphology` 和 `year`（无缺失）的样本量是 209,758。

### 3.2 `if len(data) < 30: continue` —— 最小总样本量保护

```python
if len(data) < 30:
    continue
```

- `len(data)`：返回 `data` 的行数（即有效样本量）。
- `< 30`：如果有效样本少于 30 条，跳过该特征。
- `continue`：Python 循环控制语句，跳过本次循环剩余的代码，直接进入下一次循环。

> 💡 **为什么是 30？**
>
> 30 是统计学中一个常用的经验阈值（rule of thumb），来源于**中心极限定理（Central Limit Theorem, CLT）**：当样本量 n ≥ 30 时，样本均值的抽样分布近似正态，无论总体分布是什么形状。这个阈值也出现在 `check_normality_practical` 函数中（`if len(data_clean) < 30: return False, "样本不足"`）。样本量太小，正态性检验和差异检验都不可靠。

### 3.3 `data.loc[data['target'] == 1, col]` —— 布尔索引分组

```python
group_vivo = data.loc[data['target'] == 1, col]
group_morto = data.loc[data['target'] == 0, col]
```

这是 pandas 中最核心的操作之一：**布尔索引（boolean indexing）**。我们逐层拆解：

1. `data['target'] == 1`：对 `target` 列的每个值做比较，返回一个**布尔 Series**（True/False 序列）。
   - 第 0 行 target=1 → True
   - 第 1 行 target=0 → False
   - 第 2 行 target=1 → True
   - ...

2. `data.loc[布尔Series, col]`：`.loc` 是 pandas 的标签索引器，语法是 `data.loc[行选择器, 列选择器]`。
   - 第一个参数（行选择器）：传入布尔 Series 时，pandas 会**只保留 True 对应的行**。
   - 第二个参数（列选择器）：`col` 是当前特征名，只取这一列。
   - 结果：一个 Series，包含所有 `target == 1`（即 VIVO 组）的 `col` 特征值。

**直观示意**：

```
原始 data:
   Age  target
0   45       1   ← VIVO
1   67       0   ← MORTO
2   52       1   ← VIVO
3   71       0   ← MORTO

data['target'] == 1 → [True, False, True, False]

data.loc[data['target'] == 1, 'Age'] →
0    45
2    52
Name: Age, dtype: int64
```

> 💡 **小贴士**：`.loc` vs `.iloc`
> - `.loc`：基于**标签（label）**索引，行和列都用名字。
> - `.iloc`：基于**位置（integer position）**索引，行和列都用整数下标。
> 布尔索引通常用 `.loc`，因为布尔 Series 的索引是行标签。

### 3.4 `if len(group_vivo) < 5 or len(group_morto) < 5: continue` —— 每组最小样本量

```python
if len(group_vivo) < 5 or len(group_morto) < 5:
    continue
```

- `len(group_vivo)`：VIVO 组的样本量。
- `len(group_morto)`：MORTO 组的样本量。
- `or`：逻辑或，只要**任一组**少于 5 个样本，就跳过。

> 💡 **为什么每组至少需要 5 个样本？**
>
> 这是统计检验的**最小样本要求**：
>
> 1. **T 检验**：需要计算每组的均值和标准差。当 n < 5 时，标准差估计极不稳定，t 分布的自由度太小（df = n1 + n2 - 2），检验几乎没有统计效力（power）。
> 2. **Mann-Whitney U 检验**：基于秩（rank）排序。当每组少于 5 个样本时，可能的秩排列组合太少，无法产生有意义的 p 值。事实上，当 n1 = n2 = 4 时，最小的可能双侧 p 值约为 0.057，**永远无法达到 0.05 的显著性水平**——这意味着无论数据多极端，检验都无法拒绝原假设。
>
> 所以 5 是一个"地板值"，低于这个值检验在数学上就失去了意义。在实际研究中，建议每组至少 20-30 个样本才能得到可靠的结论。

---

## 四、正态性判断 —— 决定走哪个分支

```python
is_normal, norm_note = check_normality_practical(data[col])
```

`check_normality_practical` 函数（在模块 1 中已详细讲解）返回两个值：

- `is_normal`：布尔值，True 表示近似正态，False 表示非正态。
- `norm_note`：字符串，描述正态性判断的依据（如"非正态 (偏度=-0.683)"）。

判断逻辑简述：

- 偏度绝对值 < 0.5 → 近似正态
- 偏度绝对值在 [0.5, 1.0) → 用 D'Agostino-Pearson 检验辅助判断
- 偏度绝对值 ≥ 1.0 → 非正态

> 📌 **分支选择的核心原则**：
>
> ```python
> if is_normal:
>     # T 检验分支
> else:
>     # Mann-Whitney U 检验分支
> ```
>
> 这就是经典的"参数检验 vs 非参数检验"选择：
> - **正态分布** → 用参数检验（T 检验），统计效力更高。
> - **非正态分布** → 用非参数检验（Mann-Whitney U），不要求数据正态，更稳健。

在本数据集中，4 个数值特征的偏度都很大（绝对值 ≥ 0.5），所以**全部走了 Mann-Whitney U 检验分支**。但 T 检验分支同样重要，我们必须理解它，因为：
1. 在其他数据集（如正态分布的实验室指标）中会用到。
2. 理解 T 检验有助于理解 Mann-Whitney U 检验的设计动机。

下面我们分别详解两个分支。

---

## 五、T 检验分支（正态分布情况）

> ⚠️ **说明**：本数据集的 4 个特征都未走到此分支（偏度都太大），但 T 检验是统计学最基础的检验之一，必须掌握。本节作为完整教学保留。

```python
if is_normal:
    # 方差齐性检验 (Levene)
    levene_stat, levene_p = stats.levene(group_vivo, group_morto)
    equal_var = levene_p > 0.05

    t_stat, p_value = stats.ttest_ind(group_vivo, group_morto, equal_var=equal_var)
    test_used = "T-test (独立样本t检验)"

    # Cohen's d 效应量
    n1, n2 = len(group_vivo), len(group_morto)
    s1, s2 = group_vivo.std(), group_morto.std()
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    effect_size = abs((group_vivo.mean() - group_morto.mean()) / pooled_std) if pooled_std > 0 else 0
    effect_type = "Cohen's d"
```

### 5.1 Levene 方差齐性检验

```python
levene_stat, levene_p = stats.levene(group_vivo, group_morto)
```

**原理详解：**

独立样本 T 检验有一个重要前提：**两组的方差应该相等（方差齐性）**。如果方差严重不等，标准 T 检验的 p 值会失真。Levene 检验就是用来检验"两组方差是否相等"的。

- **原假设 H₀**：σ₁² = σ₂²（两组方差相等）
- **备择假设 H₁**：σ₁² ≠ σ₂²（两组方差不等）
- **检验统计量**：基于每组观测值与该组均值的**绝对偏差**（absolute deviation）。如果两组的绝对偏差平均水平相近，说明方差相等；如果差异大，说明方差不等。
- **返回值**：
  - `levene_stat`：F 统计量，越大越倾向于拒绝 H₀。
  - `levene_p`：p 值。

> 💡 **为什么用 Levene 而不是 Bartlett？**
> Bartlett 检验对正态性非常敏感——如果数据稍微偏离正态，Bartlett 容易误判方差不等。Levene 检验更稳健（robust），对非正态数据更宽容。`scipy.stats.levene` 默认使用基于中位数的变体（`center='median'`），这是最稳健的版本。

### 5.2 `equal_var = levene_p > 0.05` —— 方差齐性判断逻辑

```python
equal_var = levene_p > 0.05
```

这是一个**布尔赋值**：

- 如果 `levene_p > 0.05`：**不拒绝**原假设（方差相等），`equal_var = True`。
- 如果 `levene_p ≤ 0.05`：**拒绝**原假设（方差不等），`equal_var = False`。

> 💡 **关键概念：p > 0.05 ≠ "证明"方差相等**
>
> 严格来说，p > 0.05 只是"没有足够证据拒绝方差相等的假设"，**并不等于"证明了方差相等"**。这是假设检验的逻辑不对称性：我们只能"拒绝"或"不拒绝"原假设，不能"接受"原假设。但在实际操作中，p > 0.05 时我们通常假设方差齐性成立，使用标准 T 检验。

### 5.3 `stats.ttest_ind` —— 独立样本 T 检验

```python
t_stat, p_value = stats.ttest_ind(group_vivo, group_morto, equal_var=equal_var)
```

**原理详解：**

独立样本 T 检验用于比较**两个独立群体**的均值是否有显著差异。

- **原假设 H₀**：μ₁ = μ₂（两组均值相等）
- **备择假设 H₁**：μ₁ ≠ μ₂（两组均值不等）
- **检验统计量**（标准 T 检验，`equal_var=True`）：

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{s_p^2 \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}
$$

其中 $s_p^2$ 是合并方差，$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}$。

- **返回值**：
  - `t_stat`：t 统计量，可正可负（符号取决于 μ₁ - μ₂ 的符号）。
  - `p_value`：双侧 p 值。

**`equal_var` 参数的关键作用：**

| `equal_var` | 检验类型 | 适用场景 | 自由度 |
|---|---|---|---|
| `True` | **Student's t 检验**（标准 T 检验） | 两组方差相等 | df = n1 + n2 - 2 |
| `False` | **Welch's t 检验** | 两组方差不等（更稳健） | Welch-Satterthwaite 近似自由度 |

> 💡 **小贴士**：Welch t 检验被许多统计学家推荐为**默认选择**，因为：
> 1. 即使方差实际相等，Welch 检验的统计效力几乎与标准 T 检验相同。
> 2. 当方差不等时，标准 T 检验的 p 值会严重失真，而 Welch 检验依然稳健。
> 3. 不需要预先做 Levene 检验，简化流程。
>
> 但本教程为了教学完整性，保留了"先 Levene 再选择"的传统流程。

### 5.4 Cohen's d 效应量

```python
n1, n2 = len(group_vivo), len(group_morto)
s1, s2 = group_vivo.std(), group_morto.std()
pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
effect_size = abs((group_vivo.mean() - group_morto.mean()) / pooled_std) if pooled_std > 0 else 0
effect_type = "Cohen's d"
```

**Cohen's d 公式：**

$$
d = \frac{|\bar{X}_1 - \bar{X}_2|}{s_p}
$$

其中 $s_p$ 是**合并标准差（pooled standard deviation）**：

$$
s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}
$$

**逐项解释：**

- `n1, n2`：两组的样本量。
- `s1, s2`：两组的样本标准差（pandas 默认 ddof=1，即无偏估计）。
- `s1**2`：第一组的方差。
- `(n1 - 1) * s1**2`：第一组的"平方和"（sum of squares）。
- `(n1 - 1) * s1**2 + (n2 - 1) * s2**2`：两组平方和之和。
- `/ (n1 + n2 - 2)`：除以总自由度（两组自由度之和），得到合并方差。
- `np.sqrt(...)`：开方得到合并标准差。
- `abs((group_vivo.mean() - group_morto.mean()) / pooled_std)`：均值差的绝对值除以合并标准差，得到 Cohen's d。
- `if pooled_std > 0 else 0`：保护性判断，避免除以零（当所有值相同时标准差为 0）。

**Cohen's d 的直观含义：**

d 表示两组均值差异**相当于多少个标准差**。例如 d = 0.5 意味着两组均值相差 0.5 个标准差。

**Cohen's d 解读标准（Cohen, 1988）：**

| Cohen's d | 解读 | 直观含义 |
|---|---|---|
| ≈ 0.2 | **小效应（small）** | 均值差约为 0.2 个标准差，差异不明显 |
| ≈ 0.5 | **中等效应（medium）** | 均值差约为 0.5 个标准差，差异可见 |
| ≈ 0.8 | **大效应（large）** | 均值差约为 0.8 个标准差，差异明显 |

> 💡 **为什么用 `abs()` 取绝对值？**
> Cohen's d 的符号表示哪组均值更高（正号表示第一组更高），但效应量的**大小**只关心差异程度，不关心方向。取绝对值便于跨特征比较效应量大小。

---

## 六、Mann-Whitney U 检验分支（非正态分布情况）

> ✅ **本数据集实际使用的分支**：4 个数值特征全部走了此分支。

```python
else:
    u_stat, p_value = mannwhitneyu(group_vivo, group_morto, alternative='two-sided')
    test_used = "Mann-Whitney U"

    # Rank-biserial correlation (效应量)
    n1, n2 = len(group_vivo), len(group_morto)
    effect_size = abs(1 - (2 * u_stat) / (n1 * n2))
    effect_type = "Rank-biserial r"
```

### 6.1 Mann-Whitney U 检验原理

**Mann-Whitney U 检验**（也叫 Wilcoxon秩和检验）是一种**非参数检验**，用于比较两组独立样本的分布是否相同。它**不要求数据正态分布**，是 T 检验的非参数替代方案。

**核心思想：基于秩（rank）的比较**

1. 把两组数据**合并**在一起。
2. 按**数值大小排序**，给每个观测值分配一个**秩（rank）**——最小的秩为 1，次小的秩为 2，依此类推。
3. 分别计算两组的**秩和（rank sum）**：R₁（VIVO 组的秩和）和 R₂（MORTO 组的秩和）。
4. 如果两组分布相同，那么两组的秩和应该接近（按样本量比例分配）；如果一组秩和明显偏大，说明该组的值整体偏大。

**U 统计量公式：**

$$
U_1 = R_1 - \frac{n_1(n_1 + 1)}{2}
$$

$$
U_2 = R_2 - \frac{n_2(n_2 + 1)}{2}
$$

$$
U = \min(U_1, U_2)
$$

其中：
- $R_1, R_2$：两组的秩和
- $n_1, n_2$：两组的样本量
- $\frac{n_1(n_1+1)}{2}$：如果第一组的所有观测值都排在最前面（即第一组所有值都最小），第一组的秩和就是这个最小值。U₁ 就是"实际秩和"减去"最小可能秩和"，表示第一组"超出最小值"的程度。

**假设：**

- **原假设 H₀**：两组来自相同分布（具体来说，两组的随机变量 X 和 Y 满足 P(X > Y) = P(Y > X)，即两组没有系统性差异）。
- **备择假设 H₁**：两组分布不同（一组系统性偏大或偏小）。

> 💡 **Mann-Whitney U vs T 检验的核心区别**
>
> | 特性 | T 检验 | Mann-Whitney U |
> |---|---|---|
> | 比较对象 | **均值** | **秩和（分布）** |
> | 分布假设 | 要求正态 | 无分布假设 |
> | 数据类型 | 连续型 | 连续型或有序型 |
> | 统计效力 | 正态时更高 | 非正态时更稳健 |
> | 对离群值 | 敏感 | 稳健（离群值只影响秩，不影响幅度） |
>
> **选择依据**：数据正态 → T 检验；数据非正态或有序 → Mann-Whitney U。

### 6.2 `alternative` 参数详解

```python
u_stat, p_value = mannwhitneyu(group_vivo, group_morto, alternative='two-sided')
```

`scipy.stats.mannwhitneyu` 的 `alternative` 参数控制**备择假设的方向**：

| `alternative` | 备择假设 | 含义 |
|---|---|---|
| `'two-sided'` | P(X > Y) ≠ 0.5 | 双侧检验：两组分布**不同**（不指定方向） |
| `'less'` | P(X > Y) < 0.5 | 单侧检验：第一组（X）**系统性偏小** |
| `'greater'` | P(X > Y) > 0.5 | 单侧检验：第一组（X）**系统性偏大** |

其中 X 是第一个参数（`group_vivo`），Y 是第二个参数（`group_morto`）。

**本代码使用 `'two-sided'` 的原因：**

我们只想知道"两组有没有差异"，不预设哪组更大。这是探索性分析中最常用、最保守的选择。

> 💡 **什么时候用单侧检验？**
> 单侧检验统计效力更高（更容易达到显著），但**必须有先验理由**支持方向假设。例如：
> - 已有大量文献证明年轻癌症患者生存率更高 → 可以用 `alternative='greater'`（VIVO 组年龄更小，但注意这里 X=VIVO，如果 VIVO 年龄小则用 `'less'`）。
> - 没有先验理由 → 必须用 `'two-sided'`。
>
> **警告**：为了"更容易显著"而选择单侧检验是**学术不端**。方向假设必须在看到数据之前就确定。

### 6.3 Rank-biserial r 效应量

```python
n1, n2 = len(group_vivo), len(group_morto)
effect_size = abs(1 - (2 * u_stat) / (n1 * n2))
effect_type = "Rank-biserial r"
```

**Rank-biserial correlation 公式：**

$$
r = \left| 1 - \frac{2U}{n_1 \cdot n_2} \right|
$$

**公式含义详解：**

1. **$n_1 \cdot n_2$**：这是**最大可能的 U 值**。想象两组完全分离——VIVO 组的每个值都大于 MORTO 组的每个值，那么每一对 (vivo_i, morto_j) 都满足 vivo_i > morto_j，这种"胜出"的对数就是 $n_1 \times n_2$。所以 $n_1 n_2$ 是 U 的理论上限。

2. **$\frac{U}{n_1 n_2}$**：实际 U 值占最大 U 值的比例，范围 [0, 1]。
   - 当 U = 0：两组完全分离，一组所有值都小于另一组。
   - 当 U = n₁n₂/2：两组分布完全相同（U 取中间值）。
   - 当 U = n₁n₂：两组完全分离，方向相反。

3. **$1 - \frac{2U}{n_1 n_2}$**：把 U 标准化到 [-1, 1] 区间。
   - 当 U = 0 → r = 1（完全分离，方向一）
   - 当 U = n₁n₂/2 → r = 0（无差异）
   - 当 U = n₁n₂ → r = -1（完全分离，方向二）

4. **`abs(...)`**：取绝对值，只关心效应大小，不关心方向。

**直观理解**：Rank-biserial r 表示"从一组随机抽一个值，从另一组随机抽一个值，前者大于后者的概率偏离 0.5 的程度"。r = 0.5 意味着这个概率约为 0.75（或 0.25），表示较强的组间差异。

**Rank-biserial r 解读标准：**

| Rank-biserial r | 解读 | 对应概率 P(X > Y) |
|---|---|---|
| ≈ 0.1 | **小效应（small）** | ≈ 0.55 或 0.45 |
| ≈ 0.3 | **中等效应（medium）** | ≈ 0.65 或 0.35 |
| ≈ 0.5 | **大效应（large）** | ≈ 0.75 或 0.25 |

> 💡 **为什么 Mann-Whitney U 用 Rank-biserial r 而不是 Cohen's d？**
> Cohen's d 依赖均值和标准差，而非正态数据的均值和标准差可能不稳定（受离群值影响大）。Rank-biserial r 基于秩，与 Mann-Whitney U 检验本身基于秩的逻辑一致，更稳健、更匹配。

---

## 七、Bonferroni 校正 —— 应对多重比较问题

```python
'Significant_Bonf': 'Yes' if p_value < 0.05 / total_n_features else 'No'
```

### 7.1 什么是多重比较问题？

当我们对**多个特征**同时做假设检验时，每个检验都有 5% 的概率犯**第一类错误（Type I error，假阳性）**——即原假设为真却被拒绝。如果做 m 次检验，至少犯一次假阳性错误的概率为：

$$
P(\text{至少一次假阳性}) = 1 - (1 - 0.05)^m
$$

| 检验次数 m | 至少一次假阳性的概率 |
|---|---|
| 1 | 5% |
| 10 | 40% |
| 20 | 64% |
| 50 | 92% |

在本案例中，`total_n_features = 4（数值）+ 18（分类）= 22`，所以如果不校正，至少一次假阳性的概率高达 **1 - 0.95²² ≈ 68%**！这意味着即使所有特征都与生存无关，我们也几乎肯定会"发现"几个"显著"特征。

### 7.2 Bonferroni 校正公式

```python
0.05 / total_n_features
```

**Bonferroni 校正**是最简单、最保守的多重比较校正方法：

$$
\alpha_{\text{校正}} = \frac{\alpha}{m}
$$

其中：
- α = 0.05（原始显著性水平）
- m = `total_n_features`（总检验次数）
- α校正 = 0.05 / 22 ≈ 0.00227

**逻辑**：把显著性阈值从 0.05 降低到 0.05/m，使得"所有检验中至少犯一次假阳性"的概率控制在 0.05 以内。

**代码解读**：

```python
'Yes' if p_value < 0.05 / total_n_features else 'No'
```

- 如果 `p_value < 0.05 / 22`（即 p < 0.00227）→ Bonferroni 校正后显著 → `'Yes'`
- 否则 → `'No'`

> 💡 **Bonferroni 校正的优缺点**
>
> **优点**：
> - 简单直观，易于理解和实现。
> - 控制的是**家族错误率（FWER）**——即"至少犯一次假阳性"的概率，非常严格。
>
> **缺点**：
> - **过于保守**：随着 m 增大，α校正 越来越小，导致很多真正有效的特征被漏判（第二类错误，假阴性）。
> - 假设所有检验独立，实际有相关时更保守。
>
> **替代方法**：
> - **Benjamini-Hochberg (BH) 校正**：控制**错误发现率（FDR）**，比 Bonferroni 更宽松，常用于高通量数据分析（如基因表达）。
> - **Holm-Bonferroni 校正**：Bonferroni 的逐步改进版，稍宽松但仍控制 FWER。

---

## 八、结果字典的数据结构

```python
numerical_results.append({
    'Feature': col,
    'N': len(data),
    'Test': test_used,
    'Normality': norm_note,
    'Mean_VIVO': mean_v,
    'Mean_MORTO': mean_m,
    'Median_VIVO': median_v,
    'Median_MORTO': median_m,
    'Statistic': t_stat if is_normal else u_stat,
    'P_Value': p_value,
    'Effect_Size': effect_size,
    'Effect_Type': effect_type,
    'Significant_0.05': 'Yes' if p_value < 0.05 else 'No',
    'Significant_Bonf': 'Yes' if p_value < 0.05 / total_n_features else 'No'
})
```

**逐字段解释：**

| 字段 | 含义 | 数据类型 | 示例值 |
|---|---|---|---|
| `Feature` | 特征名 | str | `'Age'` |
| `N` | 有效样本量（删除缺失后） | int | `209443` |
| `Test` | 使用的检验方法 | str | `'Mann-Whitney U'` |
| `Normality` | 正态性判断结果描述 | str | `'非正态 (偏度=-0.683)'` |
| `Mean_VIVO` | VIVO 组均值 | float | `60.39` |
| `Mean_MORTO` | MORTO 组均值 | float | `66.14` |
| `Median_VIVO` | VIVO 组中位数 | float | `62.0` |
| `Median_MORTO` | MORTO 组中位数 | float | `68.0` |
| `Statistic` | 检验统计量（t 或 U） | float | `4195931684` |
| `P_Value` | p 值 | float | `0.0` |
| `Effect_Size` | 效应量数值 | float | `0.2084` |
| `Effect_Type` | 效应量类型 | str | `'Rank-biserial r'` |
| `Significant_0.05` | α=0.05 是否显著 | str | `'Yes'` / `'No'` |
| `Significant_Bonf` | Bonferroni 校正后是否显著 | str | `'Yes'` / `'No'` |

**代码细节：**

1. `numerical_results` 是一个**列表（list）**，初始为空 `[]`。
2. `.append({...})`：每次循环把一个**字典（dict）**追加到列表末尾。
3. `'Statistic': t_stat if is_normal else u_stat`：三元表达式。如果走了 T 检验分支，存 `t_stat`；否则存 `u_stat`。注意：如果走了 Mann-Whitney U 分支，`t_stat` 变量根本未定义，但 `else` 分支不会访问它，所以不会报错。
4. `'Yes' / 'No'` 用字符串而不是布尔值，是为了后续输出报告时更直观。

> 💡 **为什么用字典列表而不是直接用 DataFrame？**
> 字典列表是一种灵活的"中间数据结构"——可以在循环中逐步累积结果，循环结束后用 `pd.DataFrame(numerical_results)` 一次性转换成 DataFrame，便于后续分析和保存。这种模式在数据分析中非常常见。

---

## 九、实际运行结果

下表是 4 个数值特征的完整统计结果（基于 209,758 条过滤后记录）：

| 特征 | 样本量 N | 检验方法 | 正态性 | VIVO 均值 | VIVO 中位数 | MORTO 均值 | MORTO 中位数 | 统计量 U | p 值 | 效应量类型 | 效应量 | α=0.05 显著 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Age** | 209,443 | Mann-Whitney U | 非正态 (偏度=-0.683) | 60.39 | 62.0 | 66.14 | 68.0 | 4,195,931,684 | 0.0 | Rank-biserial r | **0.2084** | Yes |
| **Code.Profession** | 193,966 | Mann-Whitney U | 非正态 (偏度=0.940) | 171.81 | 0.0 | 251.06 | 0.0 | 3,886,480,041 | 0.0 | Rank-biserial r | **0.1477** | Yes |
| **Code.of.Morphology** | 209,758 | Mann-Whitney U | 非正态 (偏度=2.634) | 82,658.11 | 80,973.0 | 82,269.73 | 80,003.0 | 7,807,489,130.5 | 0.0 | Rank-biserial r | **0.4677** | Yes |
| **year** | 209,758 | Mann-Whitney U | 非正态 (偏度=-1.004) | 2013.80 | 2015.0 | 2011.15 | 2011.0 | 7,653,596,450.5 | 0.0 | Rank-biserial r | **0.4387** | Yes |

**关键观察：**

1. **4 个特征全部走了 Mann-Whitney U 分支**：偏度绝对值都 ≥ 0.5，没有通过正态性检验。
2. **4 个特征在 α=0.05 下全部显著**：p 值都为 0.0（实际是极小的浮点数，被四舍五入显示为 0）。
3. **效应量差异明显**：从 0.15（小）到 0.47（大），特征间的实际重要性差异很大。
4. **样本量差异**：`Age` 和 `Code.Profession` 有缺失，`Code.of.Morphology` 和 `year` 完整。

> ⚠️ **关于 p = 0.0 的说明**
>
> 表格中 p 值显示为 0.0，但这**不是真正的零**。在大样本（n > 20 万）下，即使两组分布只有微小差异，Mann-Whitney U 检验也能检测到，p 值会小到浮点数精度极限（如 1e-50 以下），打印时被四舍五入为 0.0。这正是大样本下"p 值失效"的典型现象——**几乎所有差异都会显著**，此时效应量比 p 值更重要。

---

## 十、结果解读

### 10.1 Age（年龄）—— 效应量 0.21（小到中等）

- **VIVO 组**：均值 60.39 岁，中位数 62.0 岁
- **MORTO 组**：均值 66.14 岁，中位数 68.0 岁
- **差异**：存活组比死亡组**年轻约 6 岁**
- **效应量 r = 0.2084**：根据 Rank-biserial r 解读标准，介于小（0.1）和中等（0.3）之间，偏向小效应。

**医学解读**：年龄是癌症生存的公认预后因素。年轻患者通常身体机能更好、对治疗的耐受性更强、并发症更少，因此生存率更高。约 6 岁的均值差异在临床上是有意义的。

### 10.2 year（诊断年份）—— 效应量 0.44（大）

- **VIVO 组**：均值 2013.80，中位数 2015.0
- **MORTO 组**：均值 2011.15，中位数 2011.0
- **差异**：存活组的诊断年份**晚约 2.65 年**
- **效应量 r = 0.4387**：接近大效应阈值（0.5），属于中大效应。

**医学解读**：这是一个非常有趣的发现。诊断年份越晚，存活率越高，可能反映：

1. **医疗进步**：治疗技术、早期筛查、药物研发在 2008-2016 年间持续改进。
2. **随访时间偏差**：诊断年份晚的患者随访时间短，可能还没来得及"死亡"就被记录为 VIVO。这是生存分析中常见的**信息性删失（informative censoring）**问题，需要谨慎解读。
3. **数据收集质量提升**：近年来的登记数据可能更完整、更准确。

> ⚠️ **重要警示**：year 的"显著"可能部分是**伪相关**。在建模时，year 通常不作为预测特征，而是作为**协变量**或**分层变量**处理。

### 10.3 Code.of.Morphology（形态学代码）—— 效应量 0.47（大）

- **VIVO 组**：均值 82,658.11，中位数 80,973.0
- **MORTO 组**：均值 82,269.73，中位数 80,003.0
- **差异**：存活组的形态学代码**略高**
- **效应量 r = 0.4677**：大效应，接近 0.5。

**医学解读**：形态学代码（ICD-O-3 Morphology）编码了肿瘤的**组织学类型**，如腺癌、鳞癌、小细胞癌等。不同组织学类型的恶性程度、侵袭性、对治疗的反应差异巨大——这是癌症生存最强的预后因素之一。

> ⚠️ **数值陷阱警示**：Code.of.Morphology 的"均值"在数学上有意义，但在医学上**没有直接解释**——因为代码是**分类标签**而非连续量。例如，代码 8140/3（腺癌）和 8070/3（鳞癌）的数值差 70，并不代表"70 单位的生物学差异"。正确的做法是把它当作**类别变量**做卡方检验（模块 3 会讲）。本模块把它当数值处理，是为了演示 Mann-Whitney U 检验的流程，但效应量的"大"反映的是不同癌症类型生存差异的真实存在，这一点是有意义的。

### 10.4 Code.Profession（职业代码）—— 效应量 0.15（小）

- **VIVO 组**：均值 171.81，中位数 0.0
- **MORTO 组**：均值 251.06，中位数 0.0
- **差异**：存活组职业代码均值更低
- **效应量 r = 0.1477**：小效应，接近 0.1。

**医学解读**：职业通过**社会经济地位（SES）**间接影响癌症生存——高 SES 人群通常有更好的医疗资源、更早的诊断、更健康的生活方式。但本数据集中：

1. **中位数都是 0**：说明大量样本的职业代码是 0（可能是"未分类"或"无业"），均值被极端值拉高。
2. **缺失率高**：209,758 - 193,966 = 15,792 条缺失（约 7.5%），可能存在**缺失非随机（MNAR）**问题。
3. **效应量小**：即使统计显著，实际预测价值有限。

> 💡 **统计显著 vs 实际重要**
>
> Code.Profession 的 p 值为 0.0（统计显著），但效应量只有 0.15（小效应）。在大样本下，**任何微小的差异都会被检测为"统计显著"**。这就是为什么我们必须看效应量——它告诉我们差异**有多大**，而 p 值只告诉我们差异**是否可排除随机波动**。

---

## 十一、两个核心引用：选择依据与效应量

> 📌 **引用一：T 检验 vs Mann-Whitney U 的选择依据**
>
> 选择检验方法的核心是**数据分布**：
>
> 1. **先判断正态性**：用偏度 + 正态性检验（如 Shapiro-Wilk、D'Agostino-Pearson）。
> 2. **正态 → T 检验**：比较均值，统计效力高，结果易解释。
> 3. **非正态 → Mann-Whitney U**：比较秩和，稳健，不要求数据正态。
>
> **额外考虑**：
> - **样本量**：小样本（n < 30）时正态性检验效力低，建议默认用 Mann-Whitney U。
> - **离群值**：数据有极端离群值时，T 检验结果不可靠，用 Mann-Whitney U。
> - **有序数据**：如 Likert 量表（1-5 分），本质是有序而非连续，用 Mann-Whitney U。
> - **大样本**：n > 5000 时，正态性检验几乎总是拒绝正态（过于敏感），建议用偏度判断或直接用 Mann-Whitney U。

> 📌 **引用二：效应量比 p 值更重要**
>
> **p 值的局限**：
> - p 值依赖样本量——n 越大，越容易显著。本案例 n = 20 万，p 值全部为 0，无法区分特征重要性。
> - p 值只回答"有没有差异"，不回答"差异有多大"。
> - p 值容易诱导"二元思维"（显著/不显著），忽略连续的差异程度。
>
> **效应量的优势**：
> - 效应量**不受样本量影响**（Cohen's d 和 Rank-biserial r 都已标准化）。
> - 效应量直接衡量"差异有多大"，具有实际意义。
> - 效应量可以跨研究、跨特征比较。
>
> **现代统计学的建议**：报告结果时，**必须同时给出 p 值和效应量**，并优先解读效应量。美国心理学会（APA）和国际医学期刊编辑委员会（ICMJE）都要求作者报告效应量。
>
> **本案例的体现**：4 个特征 p 值都是 0.0，但效应量从 0.15 到 0.47，差异 3 倍。如果只看 p 值，会错误地认为 4 个特征同等重要；看效应量才能识别出 `Code.of.Morphology` 和 `year` 是最重要的。

---

## 十二、小贴士

### 小贴士 1：`dropna()` 的 `subset` 参数

本代码用 `df_model[[col, 'target']].dropna()` 实现成对删除。另一种更高效的写法是：

```python
data = df_model.dropna(subset=[col, 'target'])
```

`subset` 参数指定只在某些列上检查缺失，避免删除其他列的缺失行。两种写法在本例中等价，但 `subset` 写法在列数多时更高效（不需要先切片）。

### 小贴士 2：`mannwhitneyu` 的 `method` 参数

对于大样本（n > 5000），`mannwhitneyu` 默认使用**正态近似（normal approximation）**计算 p 值，而不是精确分布。这是因为精确分布在 n 大时计算量爆炸。`method='asymptotic'` 是默认值，`method='exact'` 用于小样本。本案例 n = 20 万，必然用近似法。

### 小贴士 3：pandas 的 `.std()` 默认 ddof=1

```python
s1, s2 = group_vivo.std(), group_morto.std()
```

pandas 的 `.std()` 默认 `ddof=1`（无偏估计，除以 n-1），而 NumPy 的 `np.std()` 默认 `ddof=0`（有偏估计，除以 n）。在计算 Cohen's d 时，必须用 ddof=1 的标准差，否则结果会偏小。如果用 NumPy，要写 `np.std(x, ddof=1)`。

### 小贴士 4：处理 `p_value = 0.0` 的报告

当 p 值小到浮点数下溢为 0 时，报告应写 `p < 0.001` 或 `p < 1e-300`，而不是 `p = 0`。可以用：

```python
p_str = f"p < {np.finfo(float).tiny:.0e}" if p_value == 0 else f"p = {p_value:.4e}"
```

---

## 十三、常见问题（FAQ）

### Q1：为什么我的数据偏度绝对值 < 0.5，但 `check_normality_practical` 返回非正态？

**A**：`check_normality_practical` 在偏度 [0.5, 1.0) 区间会调用 `stats.normaltest`（D'Agostino-Pearson 检验）做辅助判断。如果偏度 < 0.5，应返回正态。请检查：
1. 是否传入了正确的数据列（`data[col]` 而非 `data`）。
2. 是否有 `NaN` 未被删除（`check_normality_practical` 内部会 `dropna`，但建议外部也清理）。

### Q2：Mann-Whitney U 检验的 U 统计量为什么这么大（几十亿）？

**A**：U 的理论上限是 $n_1 \times n_2$。本案例两组各约 10 万样本，$n_1 n_2 \approx 10^{10}$（百亿级），所以 U 值在十亿级是正常的。U 值本身的大小没有直观意义，要看效应量（已标准化到 [0, 1]）。

### Q3：Bonferroni 校正后所有特征还显著吗？

**A**：本案例 4 个数值特征的 p 值都极小（远小于 0.05/22 ≈ 0.00227），所以 Bonferroni 校正后**仍然全部显著**。但对于效应量小的特征（如 Code.Profession, r=0.15），即使统计显著，实际建模时可能不值得纳入。

### Q4：为什么 `Code.Profession` 的中位数是 0？

**A**：这表明超过一半的样本职业代码为 0。0 可能代表"未分类"、"无业"或"缺失被填充为 0"。这种情况下，均值（171.81）被少数高代码值拉高，**不能代表"典型"职业**。中位数 0 才是更稳健的中心趋势度量。这也是为什么非正态数据要看中位数而非均值。

### Q5：能否对 `Code.of.Morphology` 用 T 检验？

**A**：**不建议**。虽然它被 pandas 识别为数值型，但本质是**类别编码**。即使强行做 T 检验，均值差异的医学解释也站不住脚。正确做法是把它当类别变量，用卡方检验（模块 3 内容）。本模块用 Mann-Whitney U 是因为它不假设正态，比 T 检验更稳健，但仍非最优——最优是卡方检验。

### Q6：`equal_var=False`（Welch t 检验）和 `equal_var=True`（标准 t 检验）结果差异大吗？

**A**：当两组样本量相近且方差相近时，两种方法结果几乎相同。当**样本量不等且方差不等**时，标准 t 检验可能严重失真（p 值偏小，假阳性率升高），Welch t 检验更可靠。许多统计学家建议**默认用 Welch t 检验**（`equal_var=False`），跳过 Levene 检验。

### Q7：为什么 `total_n_features` 包含分类特征？

**A**：因为模块 3 会对分类特征做卡方检验，也是假设检验。Bonferroni 校正应**覆盖所有同时进行的检验**，所以 `total_n_features = 4（数值）+ 18（分类）= 22`。如果只对数值特征校正，会低估多重比较风险。

---

## 十四、本模块小结

### 核心知识点回顾

1. **数据预处理**：
   - `df_model[[col, 'target']].dropna()` 实现**成对删除**——只要特征或目标任一缺失，就删除该行。
   - `data.loc[data['target'] == 1, col]` 用**布尔索引**按目标分组。
   - 每组至少 5 个样本是统计检验的**最小要求**（Mann-Whitney U 在 n<5 时无法达到 0.05 显著性）。

2. **检验方法选择**：
   - **正态 → T 检验**：先 Levene 检验方差齐性，`equal_var` 决定用标准 t 还是 Welch t。
   - **非正态 → Mann-Whitney U**：基于秩和，稳健，不假设分布。
   - 本数据集 4 个特征全部非正态，全部走 Mann-Whitney U 分支。

3. **效应量**：
   - T 检验用 **Cohen's d** = |μ₁ - μ₂| / pooled_std，解读：0.2小/0.5中/0.8大。
   - Mann-Whitney U 用 **Rank-biserial r** = |1 - 2U/(n₁n₂)|，解读：0.1小/0.3中/0.5大。
   - **效应量比 p 值更重要**——大样本下 p 值几乎总是显著，效应量才能区分特征重要性。

4. **多重比较校正**：
   - Bonferroni 校正：α校正 = α / m = 0.05 / 22 ≈ 0.00227。
   - 控制家族错误率（FWER），保守但简单。

### 实际结果总结

| 特征 | 效应量 r | 解读 | 实际重要性 |
|---|---|---|---|
| Code.of.Morphology | 0.47 | 大 | ⭐⭐⭐ 癌症类型是核心预后因素 |
| year | 0.44 | 大 | ⭐⭐⭐ 但可能有随访偏差，建模时谨慎 |
| Age | 0.21 | 小到中 | ⭐⭐ 经典预后因素，临床有意义 |
| Code.Profession | 0.15 | 小 | ⭐ 统计显著但实际价值有限 |

### 下一步

本模块完成了**数值特征**的单因素分析。接下来：

- **模块 3**：分类特征的卡方检验——处理 `Gender`、`Extension` 等类别变量。
- **模块 4**：汇总所有显著特征，比较效应量，讨论"统计显著 vs 预测能力强"的关系。

> 🎯 **关键启示**：统计学分析不是机械地跑检验、看 p 值，而是要**结合效应量、领域知识、数据质量**综合判断。一个 p = 0.0 但效应量 0.15 的特征，可能不如一个 p = 0.01 但效应量 0.5 的特征有价值。学会这种"批判性统计思维"，是从"会跑代码"到"会做研究"的关键一步。
