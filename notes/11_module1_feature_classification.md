# 模块 1：特征分类与正态性判断

> 本模块是案例教程 2「统计学分析」的第二步。在加载数据、创建目标变量之后，我们要对 38 个原始特征进行分类（数值型 vs 分类型），并判断每个数值特征是否服从正态分布——这决定了后续用哪种统计检验方法（t 检验还是 Mann-Whitney U 检验）。本模块对应代码文件 `src/02_statistical_analysis.py` 的第 60–132 行。
>
> 本模块最核心的知识点是：**为什么大样本下正态性检验会失效，以及如何用"偏度 + D'Agostino-Pearson 检验"的实用组合策略来判断正态性**。

---

## 学习目标

学完本模块后，你将能够：

1. **理解特征分类的原则**：知道为什么要手动指定数值型和分类型候选，而不是自动推断（回顾案例 1 的"数值陷阱"）。
2. **掌握排除列表的设计**：理解为什么 ID 型、日期型、目标变量本身必须排除在分析之外。
3. **理解列表推导式的双重过滤**：学会用 `[c for c in ... if c in ... and c not in ...]` 同时检查"列存在"和"不在排除列表"。
4. **理解 Bonferroni 校正的预备知识**：知道 `total_n_features` 为什么要在数值+分类特征上统一计算。
5. **深入理解 `check_normality_practical` 函数**：掌握偏度判断、大样本抽样、D'Agostino-Pearson 检验、三档判断逻辑、异常处理、元组返回值等每个细节。
6. **理解大样本下正态性检验失效的原因**：明白为什么 20 万样本下 Shapiro-Wilk 等检验会"过度敏感"，以及为什么用偏度作为第一道筛选。
7. **根据正态性结果选择检验方法**：正态 → t 检验，非正态 → Mann-Whitney U 检验。

---

## 一、特征分类：数值型 vs 分类型

```python
print("\n" + "=" * 70)
print("步骤 1: 特征分类")
print("=" * 70)

# 识别数值型和分类型候选
all_cols = df_model.columns.tolist()
exclude_cols = ['Patient.Code', 'target', 'Status.Vital',
                'Date.of.Birth', 'Date.of.Death', 'Date.of.Last.Contact',
                'Date.of.Diagnostic']

# 数值型候选
numeric_candidates = ['Age', 'Code.Profession', 'Code.of.Morphology', 'year']

# 分类型候选 (根据领域知识和低缺失率筛选)
categorical_candidates = [
    'Gender', 'Raca.Color', 'Diagnostic.means', 'Extension',
    'Laterality', 'State.Civil', 'Degree.of.Education',
    'Description.of.Topography', 'Topography.Code',
    'Morphology.Description', 'Description.of.Disease',
    'Illness.Code', 'Child.Illness.Description',
    'Youth.Adult.Illness.Description', 'Type.of.Death',
    'Distant.metastasis', 'Nationality', 'Naturality.State'
]
```

### 1.1 为什么要做特征分类？

统计学分析的核心任务是"找出与目标变量（存活/死亡）显著相关的特征"。但数值型和分类型特征需要用**完全不同的统计检验方法**：

| 特征类型 | 检验目标 | 使用的检验 |
|---------|---------|-----------|
| 数值型（正态） | 比较两组（存活 vs 死亡）的均值是否相等 | 独立样本 t 检验 |
| 数值型（非正态） | 比较两组的中位数是否相等 | Mann-Whitney U 检验 |
| 分类型 | 检验特征与目标变量是否独立 | 卡方独立性检验 |

所以第一步必须把特征正确分类，才能选择对应的检验方法。

### 1.2 `exclude_cols` 排除列表详解

```python
exclude_cols = ['Patient.Code', 'target', 'Status.Vital',
                'Date.of.Birth', 'Date.of.Death', 'Date.of.Last.Contact',
                'Date.of.Diagnostic']
```

这个列表定义了**必须排除在统计分析之外**的字段。每类字段的排除原因如下：

#### ID 型字段：`Patient.Code`

- **为什么排除**：`Patient.Code` 是患者的唯一标识符（类似身份证号），每个值都不同。
- **统计意义**：ID 是"高基数"变量（cardinality 极高），既不是数值型也不是有意义的分类型。把它当分类变量做卡方检验会得到无意义的结果（每个类别只有 1 个样本）。
- **机器学习意义**：ID 没有预测能力，新患者的 ID 与历史患者的 ID 不同，无法泛化。

#### 目标变量本身：`target`、`Status.Vital`

- **为什么排除**：`target` 是我们从 `Status.Vital` 派生出来的目标变量，`Status.Vital` 是它的原始版本。
- **统计意义**：用目标变量预测目标变量是"数据泄露"（data leakage），会得到 100% 的"显著"结果，毫无意义。
- **机器学习意义**：特征矩阵 X 中绝不能包含目标变量 y，否则模型作弊。

#### 日期型字段：`Date.of.Birth`、`Date.of.Death`、`Date.of.Last.Contact`、`Date.of.Diagnostic`

- **为什么排除**：日期是时间戳，虽然可以转成数值（如 Unix 时间戳），但直接做统计检验没有意义。
- **统计意义**：日期的"均值"没有实际含义（"平均出生日期是 1985 年 3 月"对预测存活没帮助）。而且 `Date.of.Death` 本身就泄露了死亡信息——有死亡日期的人一定是 MORTO。
- **正确做法**：从日期中**提取有意义的特征**，比如"诊断时的年龄"（用 `Date.of.Diagnostic - Date.of.Birth` 算）、"诊断年份"（`year`，已在 numeric_candidates 中）。

> 💡 **小贴士**：排除列表是特征工程中非常重要的"防御性编程"。它的核心思想是：**在分析开始前，就把不该进入模型的字段挡在门外**，避免后续误用。养成这个习惯能避免很多低级错误（如数据泄露）。

### 1.3 为什么手动指定而非自动推断？（回顾"数值陷阱"）

你可能会问：pandas 不是能自动推断每列的数据类型吗？为什么还要手动指定 `numeric_candidates` 和 `categorical_candidates`？

这是案例教程 1 中讲过的**"数值陷阱"**：**有些字段虽然存储为数值，但本质上是类别编码，不能当数值型处理**。

本数据集有两个典型的"数值陷阱"：

#### `Code.Profession`（职业编码）

- 存储类型：整数（如 100、200、300...）
- 实际含义：职业的**编码**，比如 100=医生、200=教师、300=工程师...
- **为什么不能当数值型**：职业编码的"大小"没有意义——"医生(100)"比"教师(200)"小 100，但这不意味着医生比教师"少"。编码只是标签，数值大小是任意的。
- **本教程的处理**：仍然放在 `numeric_candidates` 中，但用 Mann-Whitney U 检验（非参数）而非 t 检验，因为它的分布明显非正态。

#### `Code.of.Morphology`（形态学编码）

- 存储类型：整数（如 8000、8010、8050...）
- 实际含义：肿瘤形态学的**ICD-O 编码**，是一种医学分类编码。
- **为什么不能当数值型**：同样是编码，数值大小无意义。
- **本教程的处理**：同上，放在 `numeric_candidates` 中用非参数检验。

#### 自动推断的风险

如果用 `df.select_dtypes(include=['number'])` 自动选数值列，会把 `Code.Profession` 和 `Code.of.Morphology` 当成真正的数值型，然后用 t 检验比较"存活组 vs 死亡组"的"平均职业编码"——这个均值毫无实际意义。

> 💡 **小贴士**：**永远不要盲目相信 pandas 的自动类型推断**。在医学/社会科学数据中，"编码型数值"非常常见（如 ICD 编码、邮政编码、区号）。正确做法是：**结合数据字典（data dictionary）和领域知识，手动确认每个字段的真正类型**。本教程手动指定候选列表，就是为了避免这种陷阱。

### 1.4 列表推导式的双重过滤

```python
numeric_features = [c for c in numeric_candidates
                    if c in df_model.columns and c not in exclude_cols]
categorical_features = [c for c in categorical_candidates
                        if c in df_model.columns and c not in exclude_cols]
```

这两行用**列表推导式**对候选列表做双重过滤。我们拆解来看：

#### 列表推导式语法

```python
[新元素 for 元素 in 可迭代对象 if 条件]
```

- `for c in numeric_candidates`：遍历候选列表中的每个列名 `c`。
- `if c in df_model.columns and c not in exclude_cols`：只保留满足条件的列名。
- `c`（最前面的）：把满足条件的 `c` 放入新列表。

#### 双重过滤的两个条件

1. **`c in df_model.columns`**：检查列名 `c` 是否真的存在于 DataFrame 中。
   - 为什么需要？因为候选列表是手动写的，可能写错列名（拼写错误、大小写不一致），或者数据集版本变化导致某些列被删除。这个检查能避免后续引用不存在的列时报错。
2. **`c not in exclude_cols`**：检查列名 `c` 是否在排除列表中。
   - 为什么需要？虽然候选列表里不太可能出现排除列，但这是"双重保险"。比如万一 `numeric_candidates` 误写了 `'target'`，这里能把它过滤掉。

#### 举例

假设 `numeric_candidates = ['Age', 'Code.Profession', 'year', 'TypoColumn']`，而 `df_model.columns` 中没有 `TypoColumn`：

| 候选列 | 在 df_model.columns? | 在 exclude_cols? | 是否保留 |
|--------|---------------------|------------------|---------|
| Age | ✅ | ❌ | ✅ 保留 |
| Code.Profession | ✅ | ❌ | ✅ 保留 |
| year | ✅ | ❌ | ✅ 保留 |
| TypoColumn | ❌ | ❌ | ❌ 过滤（不存在） |

最终 `numeric_features = ['Age', 'Code.Profession', 'year']`。

> 💡 **小贴士**：列表推导式是 Python 中非常强大的特性，能在一行内完成"过滤 + 转换"。对于简单的过滤场景，它比 `for` 循环 + `append` 更简洁、更 Pythonic。但注意不要过度使用——如果逻辑太复杂（多层嵌套、多个条件），还是用普通 `for` 循环更易读。

### 1.5 `total_n_features` 与 Bonferroni 校正

```python
# Bonferroni 校正: 在所有特征(数值+分类)上统一校正，与教学文档一致
total_n_features = len(numeric_features) + len(categorical_features)

print(f"\n  数值特征: {numeric_features}")
print(f"  类别特征: {categorical_features}")
print(f"  总特征数(用于Bonferroni校正): {total_n_features}")
```

#### 什么是 Bonferroni 校正？

当我们对同一个数据集做**多次**统计检验时，"假阳性"（Type I error，即实际不显著但被判定为显著）的概率会累积。比如：

- 做 1 次检验，显著性水平 α=0.05，假阳性概率 = 5%。
- 做 20 次检验，**至少一次**假阳性的概率 ≈ 1 - (1-0.05)^20 ≈ 64%！

Bonferroni 校正是最简单的多重检验校正方法：**把显著性阈值除以检验次数**。

$$\alpha_{\text{校正}} = \frac{\alpha}{n} = \frac{0.05}{\text{total\_n\_features}}$$

比如 `total_n_features = 22`，则校正后的阈值是 `0.05 / 22 ≈ 0.00227`。只有 p 值小于 0.00227 的特征才算"真正显著"。

#### 为什么要在数值+分类特征上统一计算？

本教程对 4 个数值特征和 18 个分类特征**一共做 22 次检验**，所以 `total_n_features = 4 + 18 = 22`。统一计算是因为这 22 次检验都在"同一数据集、同一目标变量"上进行，属于同一个"多重检验族"（family of tests），必须一起校正。

> ⚠️ **常见问题**：Bonferroni 校正是不是太保守了？
>
> 是的，Bonferroni 校正以"保守"著称——它能有效控制假阳性，但代价是**增加假阴性**（实际显著的被判定为不显著）。当检验次数很多时（如基因研究做几万次检验），Bonferroni 会让几乎所有结果都不显著。
>
> 替代方案：
> - **Benjamini-Hochberg (BH) 校正**：控制 FDR（假发现率），比 Bonferroni 更宽松，常用于大规模多重检验。
> - **Holm 校正**：Bonferroni 的逐步改进版，比 Bonferroni 稍宽松。
>
> 本教程只有 22 次检验，Bonferroni 足够，且教学上最简单易懂。

---

## 二、正态性判断函数（本模块核心）

```python
def check_normality_practical(data, sample_limit=5000):
    """实用正态性判断: 偏度法 + 大样本近似"""
    data_clean = data.dropna()
    if len(data_clean) < 30:
        return False, "样本不足"

    skewness = data_clean.skew()

    # 如果数据太大，抽样做正式检验
    if len(data_clean) > sample_limit:
        data_test = data_clean.sample(sample_limit, random_state=42)
    else:
        data_test = data_clean

    # 偏度绝对值 < 0.5 视为近似正态
    if abs(skewness) < 0.5:
        return True, f"近似正态 (偏度={skewness:.3f})"
    elif abs(skewness) < 1.0:
        # 边缘情况: 用 D'Agostino-Pearson 检验辅助判断
        try:
            _, p_value = stats.normaltest(data_test)
            if p_value > 0.05:
                return True, f"正态 (p={p_value:.4f})"
            else:
                return False, f"非正态 (偏度={skewness:.3f}, p={p_value:.4f})"
        except Exception:
            return False, f"非正态 (偏度={skewness:.3f})"
    else:
        return False, f"非正态 (偏度={skewness:.3f})"
```

这个函数是本模块的**核心**。它实现了一个"实用"的正态性判断策略，专门针对大样本场景。我们逐行拆解。

### 2.1 函数定义与参数

```python
def check_normality_practical(data, sample_limit=5000):
```

- **参数 `data`**：传入一个 pandas Series（某列数值数据）。
- **参数 `sample_limit=5000`**：抽样上限，默认 5000。当数据量超过这个值时，做正式正态性检验前会先抽样。`sample_limit=5000` 是默认参数，调用时可以省略，也可以自定义（如 `check_normality_practical(data, sample_limit=3000)`）。

### 2.2 第一步：删除缺失值

```python
data_clean = data.dropna()
```

- 正态性检验不能处理缺失值（NaN），所以先 `dropna()` 删除。
- 注意：这里删除的只是用于检验的临时数据，**不会修改原始 DataFrame**（因为 `dropna()` 返回的是新 Series，不是原地修改）。

### 2.3 第二步：样本量下限检查

```python
if len(data_clean) < 30:
    return False, "样本不足"
```

- 如果删除缺失值后样本量不足 30，直接返回"非正态"，理由是"样本不足"。
- **为什么是 30？** 这是统计学中关于**中心极限定理（Central Limit Theorem, CLT）**的一个经验下限。中心极限定理说：当样本量足够大时（通常以 n≥30 为经验阈值），样本均值的抽样分布近似正态，无论总体分布是什么。
- 但这里我们判断的是**数据本身的分布**，不是样本均值的分布。样本量 < 30 时，正态性检验的**功效（power）太低**——即使总体是正态的，小样本也可能检验不出正态性。所以直接判"非正态"是保守但合理的做法。
- 返回值是**元组** `(False, "样本不足")`：第一个元素是布尔值（是否正态），第二个元素是说明字符串。

> 💡 **小贴士**：`return False, "样本不足"` 这种写法等价于 `return (False, "样本不足")`，Python 会自动把逗号分隔的多个值打包成元组。这是 Python 函数返回多个值的常用技巧。

### 2.4 第三步：计算偏度

```python
skewness = data_clean.skew()
```

- `skew()` 是 pandas Series 的方法，计算**偏度（skewness）**，衡量数据分布的**不对称程度**。
- 偏度的解读：
  - **偏度 = 0**：完全对称（如正态分布）。
  - **偏度 > 0**：右偏（正偏），右侧尾巴长，大部分数据集中在左侧（如收入分布）。
  - **偏度 < 0**：左偏（负偏），左侧尾巴长，大部分数据集中在右侧（如考试成绩）。
- 经验法则：
  - **|偏度| < 0.5**：近似对称，可视为正态。
  - **0.5 ≤ |偏度| < 1.0**：中等偏斜，需要进一步检验。
  - **|偏度| ≥ 1.0**：严重偏斜，明显非正态。

### 2.5 第四步：大样本抽样（重要！）

```python
# 如果数据太大，抽样做正式检验
if len(data_clean) > sample_limit:
    data_test = data_clean.sample(sample_limit, random_state=42)
else:
    data_test = data_clean
```

- 如果数据量超过 `sample_limit`（默认 5000），就**随机抽样** 5000 个样本做正式检验；否则用全部数据。
- `data_clean.sample(sample_limit, random_state=42)`：
  - `sample_limit`：抽样数量。
  - `random_state=42`：随机种子，保证每次运行抽样结果一致（可复现）。42 是程序员圈的"梗"（来自《银河系漫游指南》中"生命、宇宙及一切的终极答案是 42"）。

#### 为什么大样本要抽样？（本模块核心知识点）

这是本模块最重要的概念之一。**在大样本下，标准正态性检验（如 Shapiro-Wilk、D'Agostino-Pearson）会"过度敏感"**——即使数据只是极其轻微地偏离正态，检验也会返回极小的 p 值（如 p < 0.0001），判定为"非正态"。

原因在于：**这些检验的零假设是"数据来自正态分布"，随着样本量增大，检验对任何微小偏离的检测能力都增强**。当样本量达到 20 万时，即使数据实际上"足够正态"（用于 t 检验没问题），检验也会因为微小的偏离而拒绝正态性假设。

> 💡 **核心知识点（本模块最重要）**：
>
> **为什么大样本下正态性检验会失效？**
>
> 正态性检验（如 Shapiro-Wilk、D'Agostino-Pearson、Kolmogorov-Smirnov）的零假设 H₀ 是"数据来自正态分布"。这些检验有一个特性：**样本量越大，对偏离正态的敏感度越高**。
>
> - n=30 时：只有明显偏离正态才会被拒绝。
> - n=1000 时：轻微偏离就会被拒绝。
> - n=200000 时：**几乎任何真实数据都会被拒绝**（因为真实数据永远不可能"完美"正态）。
>
> 这导致一个悖论：**大样本下，正态性检验几乎总是告诉你"非正态"，即使数据的实际分布对 t 检验来说完全没问题**。
>
> **解决方案**：本函数采用"偏度优先 + 抽样检验"的实用策略：
> 1. 先看偏度（与样本量无关的分布形状指标）。
> 2. 只在偏度处于"边缘地带"（0.5~1.0）时，才抽样 5000 个样本做正式检验，避免大样本过度敏感。
> 3. 偏度严重（≥1.0）时直接判非正态，不依赖检验。

### 2.6 第五步：三档判断逻辑

```python
# 偏度绝对值 < 0.5 视为近似正态
if abs(skewness) < 0.5:
    return True, f"近似正态 (偏度={skewness:.3f})"
elif abs(skewness) < 1.0:
    # 边缘情况: 用 D'Agostino-Pearson 检验辅助判断
    try:
        _, p_value = stats.normaltest(data_test)
        if p_value > 0.05:
            return True, f"正态 (p={p_value:.4f})"
        else:
            return False, f"非正态 (偏度={skewness:.3f}, p={p_value:.4f})"
    except Exception:
        return False, f"非正态 (偏度={skewness:.3f})"
else:
    return False, f"非正态 (偏度={skewness:.3f})"
```

这是函数的核心判断逻辑，分三档：

#### 第一档：`|偏度| < 0.5` → 直接判正态

```python
if abs(skewness) < 0.5:
    return True, f"近似正态 (偏度={skewness:.3f})"
```

- 偏度绝对值小于 0.5，说明分布**近似对称**，可以直接视为正态。
- **不做正式检验**，因为偏度已经足够小，没必要再检验。
- 返回 `True`（正态）和说明字符串。
- `f"近似正态 (偏度={skewness:.3f})"`：`:.3f` 表示保留 3 位小数，如"近似正态 (偏度=-0.234)"。

#### 第二档：`0.5 ≤ |偏度| < 1.0` → 用 D'Agostino-Pearson 检验辅助

```python
elif abs(skewness) < 1.0:
    # 边缘情况: 用 D'Agostino-Pearson 检验辅助判断
    try:
        _, p_value = stats.normaltest(data_test)
        if p_value > 0.05:
            return True, f"正态 (p={p_value:.4f})"
        else:
            return False, f"非正态 (偏度={skewness:.3f}, p={p_value:.4f})"
    except Exception:
        return False, f"非正态 (偏度={skewness:.3f})"
```

- 偏度在 0.5~1.0 之间，属于"中等偏斜"的**边缘地带**，单看偏度难以判断，需要正式检验辅助。
- **`stats.normaltest(data_test)`**：调用 D'Agostino-Pearson 正态性检验。
- **`_, p_value = ...`**：函数返回两个值（统计量, p 值），我们只关心 p 值，用 `_` 丢弃统计量（Python 约定：`_` 表示"我不需要这个值"）。
- **判断逻辑**：
  - `p_value > 0.05`：不能拒绝正态性假设，判为正态。
  - `p_value ≤ 0.05`：拒绝正态性假设，判为非正态。
- 注意这里用的是 `data_test`（抽样后的数据，最多 5000 个样本），而不是 `data_clean`（全部 20 万样本），正是为了避免大样本过度敏感。

#### 第三档：`|偏度| ≥ 1.0` → 直接判非正态

```python
else:
    return False, f"非正态 (偏度={skewness:.3f})"
```

- 偏度绝对值 ≥ 1.0，说明分布**严重偏斜**，明显非正态，无需再做检验。
- 直接返回 `False`（非正态）。

### 2.7 `stats.normaltest()` 详解

```python
_, p_value = stats.normaltest(data_test)
```

- **`stats.normaltest()`** 是 SciPy 提供的 **D'Agostino-Pearson 正态性检验**（也叫 K² 检验）。
- 它结合了**偏度**和**峰度**（kurtosis）两个指标，构造一个综合统计量 K²，检验"数据是否来自正态分布"。
- **零假设 H₀**：数据来自正态分布。
- **返回值**：元组 `(K2_statistic, p_value)`。
  - `K2_statistic`：检验统计量，越大越偏离正态。
  - `p_value`：p 值，小于显著性水平（如 0.05）则拒绝 H₀，判为非正态。
- **与其他正态性检验的对比**：

| 检验方法 | 函数 | 适用场景 | 大样本表现 |
|---------|------|---------|-----------|
| Shapiro-Wilk | `stats.shapiro()` | 小样本（n<50），功效最高 | n>5000 不推荐 |
| D'Agostino-Pearson | `stats.normaltest()` | 中等样本，综合偏度和峰度 | 适合中等样本，大样本仍过度敏感 |
| Kolmogorov-Smirnov | `stats.kstest()` | 大样本，但功效较低 | 适合大样本，但对连续分布需要指定参数 |
| Anderson-Darling | `stats.anderson()` | 对尾部偏离敏感 | 适合各种样本量 |

本教程选 `normaltest` 是因为它综合考虑偏度和峰度，且对中等样本（抽样后的 5000）表现良好。

### 2.8 `try-except` 异常处理

```python
try:
    _, p_value = stats.normaltest(data_test)
    ...
except Exception:
    return False, f"非正态 (偏度={skewness:.3f})"
```

- **为什么需要异常处理？** `stats.normaltest()` 在某些边缘情况下可能抛出异常，例如：
  - 数据中有 `inf`（无穷大）值。
  - 数据方差为 0（所有值相同，如某列全是 5）。
  - 数据类型异常（如 object 类型混入）。
- **异常时的处理**：捕获 `Exception`（所有异常），返回 `False`（非正态），并附上偏度信息。
- **设计哲学**：正态性判断是"辅助决策"环节，不应该因为一个特征的检验异常而中断整个分析流程。捕获异常后默认判为非正态（保守），让流程继续。

> 💡 **小贴士**：`except Exception` 会捕获所有非系统退出类异常（不捕获 `KeyboardInterrupt`、`SystemExit`）。在生产代码中，更推荐捕获具体异常（如 `except (ValueError, RuntimeError)`），避免意外吞掉重要错误。但在数据分析脚本中，`except Exception` 配合日志记录是常见做法，保证流程不中断。

### 2.9 函数返回值：元组 `(bool, str)`

函数返回一个**元组**，包含两个元素：

| 元素 | 类型 | 含义 |
|------|------|------|
| 第一个 | `bool` | 是否正态（True=正态，False=非正态） |
| 第二一个 | `str` | 说明字符串（包含偏度、p 值等诊断信息） |

调用时可以这样接收：

```python
is_normal, reason = check_normality_practical(df_model['Age'])
print(f"是否正态: {is_normal}, 原因: {reason}")
```

**为什么返回元组而不是只返回布尔值？**

- 布尔值只能告诉你"是/否"，但教学和调试时我们需要知道"为什么"。
- 说明字符串包含偏度、p 值等诊断信息，便于理解判断依据、撰写分析报告。
- 这是数据分析代码的好习惯：**返回结论 + 依据**，而不仅仅是结论。

---

## 三、本模块运行结果

运行特征分类和正态性判断后，控制台输出大致如下：

```
======================================================================
步骤 1: 特征分类
======================================================================

  数值特征: ['Age', 'Code.Profession', 'Code.of.Morphology', 'year']
  类别特征: ['Gender', 'Raca.Color', 'Diagnostic.means', 'Extension', 'Laterality',
            'State.Civil', 'Degree.of.Education', 'Description.of.Topography',
            'Topography.Code', 'Morphology.Description', 'Description.of.Disease',
            'Illness.Code', 'Child.Illness.Description',
            'Youth.Adult.Illness.Description', 'Type.of.Death', 'Distant.metastasis',
            'Nationality', 'Naturality.State']
  总特征数(用于Bonferroni校正): 22
```

### 3.1 数值特征的正态性检验结果

对 4 个数值特征逐一调用 `check_normality_practical`，结果如下：

| 特征 | 偏度 | 判定 | 说明 | 后续检验 |
|------|------|------|------|---------|
| Age | -0.683 | 非正态 | 偏度绝对值在 0.5~1.0 之间，D'Agostino-Pearson 检验 p<0.05 | Mann-Whitney U |
| Code.Profession | 0.940 | 非正态 | 偏度接近 1.0，边缘地带但检验拒绝正态 | Mann-Whitney U |
| Code.of.Morphology | 2.634 | 非正态 | 偏度 ≥ 1.0，严重右偏，直接判非正态 | Mann-Whitney U |
| year | -1.004 | 非正态 | 偏度绝对值 ≥ 1.0，严重左偏，直接判非正态 | Mann-Whitney U |

### 3.2 结果解读

**4 个数值特征全部判定为非正态**，因此后续全部使用 **Mann-Whitney U 检验**（而非 t 检验）。

具体分析：

1. **Age（年龄）**：偏度 -0.683，中等左偏。年龄分布左偏可能是因为数据中老年患者较多（癌症高发于老年），少数年轻患者拉低了左侧尾巴。虽然偏度不算大，但 D'Agostino-Pearson 检验仍拒绝正态性（大样本下检验敏感）。

2. **Code.Profession（职业编码）**：偏度 0.940，接近严重右偏。注意这是**编码型数值**，本质是类别变量，偏度大是正常的——编码值的分布反映的是各类职业的人数分布，少数高频职业 + 多数低频职业导致右偏。

3. **Code.of.Morphology（形态学编码）**：偏度 2.634，严重右偏。同样是编码型数值，少数形态学编码（如 8000/3 表示"恶性上皮肿瘤"）出现频率极高，其他编码频率低，导致严重右偏。

4. **year（诊断年份）**：偏度 -1.004，严重左偏。这说明数据集中**近年诊断的病例多于早年**——可能因为癌症登记系统逐年完善、覆盖率提高，近年记录更完整。这种"逐年增长"的趋势导致年份分布左偏（大部分数据集中在近年）。

### 3.3 为什么全部非正态？这正常吗？

在大样本（20 万+）的真实医学数据中，**全部数值特征非正态是非常常见的**，原因：

1. **大样本效应**：如前所述，大样本下正态性检验过度敏感，轻微偏离就判非正态。
2. **真实数据的特性**：真实世界的数据很少完美正态，尤其是编码型变量（如 Code.Profession、Code.of.Morphology）本质是类别，分布自然非正态。
3. **保守策略**：本函数的偏度阈值（0.5/1.0）相对保守，宁可判非正态用非参数检验，也不冒险用 t 检验。

> 💡 **小贴士**：全部非正态并不意味着分析失败。Mann-Whitney U 检验是非参数检验，**不要求正态性假设**，且在大样本下功效接近 t 检验。所以改用 Mann-Whitney U 是完全可靠的替代方案。事实上，在医学研究中，非参数检验比 t 检验更常用，因为医学数据很少正态。

---

## 四、小贴士与常见问题

### 💡 小贴士汇总

1. **手动指定特征类型**：不要盲目相信 pandas 的 `select_dtypes`，编码型数值（如 ICD 编码、职业编码）必须手动识别并正确分类。
2. **排除列表是防御性编程**：在分析开始前就把 ID、日期、目标变量本身挡在门外，避免数据泄露和无意义分析。
3. **大样本下别用 Shapiro-Wilk**：Shapiro-Wilk 在 n>5000 时不推荐，改用 D'Agostino-Pearson 或 Anderson-Darling，且最好先抽样。
4. **偏度是第一道筛选**：偏度与样本量无关，是稳定的分布形状指标。先用偏度筛选，再决定是否做正式检验，能避免大样本过度敏感。
5. **返回结论 + 依据**：函数返回 `(bool, str)` 元组，既给结论又给依据，便于调试和撰写报告。
6. **异常处理保证流程不中断**：正态性判断是辅助环节，不应因单个特征异常而中断整个分析。

### ⚠️ 常见问题

**Q1：为什么不用 Shapiro-Wilk 检验？它不是最权威的正态性检验吗？**

A：Shapiro-Wilk 在小样本（n<50）时功效最高，确实是"最权威"的。但它有两个问题：(1) SciPy 的 `shapiro()` 在 n>5000 时会给出警告，不推荐使用；(2) 大样本下它同样过度敏感。本数据集有 20 万样本，即使抽样到 5000，Shapiro-Wilk 仍然过于敏感。D'Agostino-Pearson 综合偏度和峰度，对中等样本表现更稳定。

**Q2：偏度阈值 0.5 和 1.0 是怎么定的？有理论依据吗？**

A：这是经验法则，没有严格理论依据。常见参考：
- |偏度| < 0.5：近似对称（Tabachnick & Fidell 的经典教材）。
- 0.5 ≤ |偏度| < 1.0：中等偏斜，需要结合其他指标判断。
- |偏度| ≥ 1.0：严重偏斜，明显非正态。
不同教材阈值略有差异（有的用 0.5/1.0，有的用 0.5/2.0），本教程采用较保守的版本。

**Q3：为什么 `random_state=42`？换成别的数字会怎样？**

A：`random_state` 是随机种子，保证抽样结果可复现。换成任何数字（如 0、1、123）都可以，结果会略有不同（因为抽到的样本不同），但整体判断（正态/非正态）通常一致。42 是程序员的"梗"，社区约定俗成。**生产代码中必须固定 random_state**，否则每次运行结果不同，无法复现。

**Q4：如果某个数值特征被判为正态，后续就用 t 检验吗？**

A：是的。本教程的逻辑是：
- 正态 → `stats.ttest_ind()`（独立样本 t 检验）
- 非正态 → `mannwhitneyu()`（Mann-Whitney U 检验）
- 分类变量 → `chi2_contingency()`（卡方检验）
本数据集 4 个数值特征全部非正态，所以全部用 Mann-Whitney U。

**Q5：Bonferroni 校正后，会不会所有特征都不显著了？**

A：有可能，但本数据集样本量大（20 万），即使校正后阈值很严（0.05/22 ≈ 0.00227），大部分真正有影响的特征（如 Age、Extension）的 p 值会远小于这个阈值（如 p < 0.0001），仍然显著。大样本的好处是"统计功效高"，能检测出微小的效应，但也意味着"统计显著 ≠ 实际重要"——这是本教程后续要讨论的核心问题。

**Q6：`Code.Profession` 和 `Code.of.Morphology` 既然是编码型数值，为什么不放到分类型？**

A：好问题！理论上它们应该当分类型处理。本教程放在数值型是为了教学演示：(1) 展示正态性判断对编码型数据的结果（必然非正态）；(2) 展示 Mann-Whitney U 检验对这类数据的处理。在实际项目中，更合理的做法是把它们转成分类型，用卡方检验。读者可以尝试这种替代方案并对比结果。

---

## 五、本模块小结

### 核心知识点回顾

1. **特征分类的原则**：
   - 手动指定 `numeric_candidates` 和 `categorical_candidates`，避免"数值陷阱"（编码型数值被误当数值型）。
   - 典型陷阱：`Code.Profession`（职业编码）、`Code.of.Morphology`（形态学编码）虽是数值，本质是类别。

2. **排除列表 `exclude_cols`**：
   - ID 型（`Patient.Code`）：高基数，无统计意义。
   - 目标变量（`target`、`Status.Vital`）：避免数据泄露。
   - 日期型（`Date.of.Birth` 等）：均值无意义，且 `Date.of.Death` 泄露死亡信息。

3. **列表推导式双重过滤**：`[c for c in candidates if c in df.columns and c not in exclude]`，同时检查"列存在"和"不在排除列表"。

4. **Bonferroni 校正预备**：`total_n_features = 数值特征数 + 分类特征数`，用于后续多重检验校正，校正阈值 = 0.05 / total_n_features。

5. **`check_normality_practical` 函数（本模块核心）**：
   - **样本量下限**：n < 30 直接判非正态（中心极限定理经验阈值）。
   - **偏度计算**：`data_clean.skew()`，与样本量无关的分布形状指标。
   - **大样本抽样**：n > 5000 时抽样 5000 个样本做检验，避免大样本过度敏感。
   - **三档判断**：
     - |偏度| < 0.5 → 直接判正态。
     - 0.5 ≤ |偏度| < 1.0 → D'Agostino-Pearson 检验辅助（`stats.normaltest`）。
     - |偏度| ≥ 1.0 → 直接判非正态。
   - **异常处理**：`try-except` 捕获检验异常，默认判非正态，保证流程不中断。
   - **返回元组**：`(bool, str)`，结论 + 依据。

6. **大样本下正态性检验失效**：样本量越大，检验越敏感，几乎任何真实数据都会被判非正态。解决方案是偏度优先 + 抽样检验。

### 运行结果

- 数值特征：4 个（Age、Code.Profession、Code.of.Morphology、year）
- 分类特征：18 个
- 总特征数：22（用于 Bonferroni 校正）
- **4 个数值特征全部非正态** → 后续全部使用 Mann-Whitney U 检验

### 下一模块预告

本模块完成了特征分类和正态性判断，确定了"4 个数值特征全部用 Mann-Whitney U 检验"。下一模块（**模块 2：数值特征单因素分析**）将：
- 对每个数值特征执行 Mann-Whitney U 检验，比较存活组 vs 死亡组的分布差异
- 计算效应量（effect size），衡量差异的实际大小
- 应用 Bonferroni 校正，控制多重检验的假阳性
- 讨论"统计显著 ≠ 实际重要"这一统计学与机器学习的核心张力

---
