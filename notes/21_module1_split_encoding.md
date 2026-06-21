# 模块 1：数据划分与编码

> 本模块承接模块 0（数据加载与特征选择），是案例教程 3「数据预处理与缺失值插补」的第二步。在采样得到 80,000 条数据后，我们需要把它划分成训练集和测试集，并对分类特征做编码，为后续的插补和建模做准备。
>
> 本模块最核心的知识点有两个：**一是分层抽样** **`stratify=y`**——为什么对不平衡数据至关重要；**二是处理测试集中"未知类别"的策略**——如何用训练集众数兜底，避免编码器报错。

***

## 学习目标

学完本模块后，你将能够：

1. **掌握特征矩阵 X 和标签向量 y 的提取**：理解 `df[feature_names].copy()` 和 `df['target'].values` 的细节。
2. **详解** **`train_test_split`** **的每个参数**：特别是 `stratify=y` 分层抽样的原理和重要性。
3. **理解缺失掩码的作用**：为什么要在编码前保存 `X_train.isnull()`，后续如何用于分析插补效果。
4. **深入理解 LabelEncoder 循环**：包括为什么先 `dropna` 再 `fit`、为什么转字符串、如何找众数。
5. **掌握** **`transform_with_unknown`** **函数的设计**：理解处理"未知类别"的三段式逻辑（缺失→NaN、已知→编码、未知→众数）。
6. **理解为什么要在插补前做编码**：sklearn 插补器要求数值输入，分类变量必须先编码。
7. **理解为什么用 LabelEncoder 而非 OneHotEncoder**：权衡维度、距离计算和序数关系。
8. **理解** **`astype(float)`** **的必要性**：sklearn 插补器要求浮点输入。

***

## 一、提取特征矩阵 X 和标签向量 y

```python
X = df_sample[feature_names].copy()
y = df_sample['target'].values
```

### 1.1 `X = df_sample[feature_names].copy()` — 提取特征矩阵

- `feature_names` 是模块 0 中定义的列表：`['Age', 'year', 'Gender', 'Diagnostic.means', 'Raca.Color']`。
- `df_sample[feature_names]` 从 `df_sample` 中选取这 5 列，返回一个 80,000 行 × 5 列的 DataFrame。
- `.copy()` 创建独立副本，后续对 `X` 的修改（如编码、插补）不会影响 `df_sample`。

**为什么用** **`feature_names`** **列表而不用** **`df_sample.drop('target', axis=1)`？**

因为 `df_sample` 可能包含很多列（原始数据有 38 个字段），我们只想保留精选的 5 个特征。用列名列表精确选取，比 `drop` 更清晰、更安全。

### 1.2 `y = df_sample['target'].values` — 提取标签向量

- `df_sample['target']` 返回一个 Series（一维数据结构）。
- `.values` 把 Series 转成 **numpy 数组**（`ndarray`）。

**为什么用** **`.values`** **转成 numpy 数组？**

1. **sklearn 的接口要求**：`train_test_split`、`LogisticRegression.fit` 等函数对 `y` 参数通常接受 numpy 数组或类数组对象。虽然它们也能处理 pandas Series，但转成 numpy 数组更"干净"，避免一些边缘问题（如 index 不连续导致的警告）。
2. **性能**：numpy 数组的底层是连续内存，比 pandas Series（带 index 和 name 等元数据）更轻量，运算更快。
3. **语义清晰**：`y` 是一维标签向量，numpy 数组比 Series 更贴合这个语义——它没有列名、没有 index，就是纯粹的数值序列。

**X 为什么不转成 numpy 数组？**

因为 `X` 后续要做列级操作（如对特定列做编码、计算每列的缺失数），DataFrame 的列名让这些操作更方便。sklearn 也完全支持 DataFrame 输入。

> 💡 **小贴士**：在 sklearn 工作流中，常见的约定是：
>
> - `X`（特征矩阵）：保留 DataFrame 格式，方便按列名操作。
> - `y`（标签向量）：转成 numpy 数组，方便数值计算。
>
> 这种约定在 sklearn 官方文档和教程中随处可见。

***

## 二、划分训练集和测试集（本模块重点之一）

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
```

### 2.1 `train_test_split` 函数详解

`train_test_split` 是 sklearn 最常用的函数之一，用于把数据集划分成训练集和测试集。它返回 4 个对象：`X_train, X_test, y_train, y_test`。

### 2.2 参数 1：`test_size=0.3` — 30% 测试集

- `test_size=0.3` 表示**测试集占 30%**，训练集占 70%。
- 80,000 条样本 → 训练集 56,000 条，测试集 24,000 条。
- `test_size` 也可以是整数（如 `test_size=24000`），表示测试集的绝对样本数。

**为什么是 70/30 而不是 80/20 或 50/50？**

| 划分比例      | 训练集        | 测试集        | 适用场景                  |
| --------- | ---------- | ---------- | --------------------- |
| 80/20     | 64,000     | 16,000     | 大数据集（百万级），20% 测试集已足够  |
| **70/30** | **56,000** | **24,000** | **中等数据集（万级），平衡训练和测试** |
| 50/50     | 40,000     | 40,000     | 小数据集，需要更多测试样本保证评估可靠   |

本教程 8 万条样本属于"中等规模"，70/30 是常见选择——训练集足够大让模型学好，测试集也足够大让评估稳定。

### 2.3 参数 2：`random_state=RANDOM_STATE` — 可复现性

- `random_state=42` 固定划分的随机种子，保证每次运行得到**完全相同的训练集和测试集**。
- 这对可复现性至关重要——如果每次划分不同，模型性能的波动就无法判断是来自划分还是来自方法本身。

### 2.4 参数 3：`stratify=y` — 分层抽样（重点）

**这是本模块最重要的参数。**

`stratify=y` 表示**分层抽样**——划分时保持训练集和测试集的**标签分布一致**。

#### 为什么需要分层抽样？

本数据集的标签分布是：VIVO 40.95%、MORTO 59.05%。如果用**普通随机划分**（不设 `stratify`），有可能出现：

- 训练集：VIVO 38%、MORTO 62%（偏离原始分布）
- 测试集：VIVO 46%、MORTO 54%（偏离原始分布）

这种偏差会导致：

1. **训练集分布偏移**：模型学到的决策边界基于偏移的分布，泛化到测试集时性能下降。
2. **测试集评估失真**：如果测试集的少数类（VIVO）比例异常高或低，AUC、Recall 等指标的方差会变大，评估不可靠。
3. **极端情况**：在小数据集 + 严重不平衡的场景下，随机划分甚至可能让测试集**完全没有少数类样本**，导致无法计算 Recall。

#### 分层抽样如何解决？

`stratify=y` 会在划分时**按标签分层**：先按 y 的取值把样本分组（VIVO 一组、MORTO 一组），然后在每组内部分别按 70/30 划分，最后合并。

```
原始数据 (80,000):
  VIVO:  32,760 (40.95%)  →  训练 22,932 (70%) + 测试 9,828 (30%)
  MORTO: 47,240 (59.05%)  →  训练 33,068 (70%) + 测试 14,172 (30%)

训练集 (56,000):
  VIVO:  22,932 (40.95%)  ← 比例保持
  MORTO: 33,068 (59.05%)  ← 比例保持

测试集 (24,000):
  VIVO:  9,828 (40.95%)   ← 比例保持
  MORTO: 14,172 (59.05%)  ← 比例保持
```

这样，训练集和测试集的标签分布**都与原始数据一致**（40.95% vs 59.05%），避免了分布偏移。

> 💡 **重点概念：分层抽样** **`stratify=y`**
>
> **什么时候必须用** **`stratify=y`？**
>
> 1. **不平衡数据集**：当某类样本占比 < 20% 时，强烈建议分层抽样，避免少数类在划分中被"稀释"。
> 2. **小数据集**：样本量 < 1000 时，随机划分的波动更大，分层抽样能稳定分布。
> 3. **需要可复现的对比实验**：分层抽样保证训练集和测试集分布一致，对比不同方法时更公平。
>
> **什么时候可以不用？**
>
> - 大数据集 + 类别平衡（如 50/50 的二分类，百万级样本）：随机划分已经足够稳定。
>
> **经验法则**：**只要数据有不平衡，就用** **`stratify=y`**。这是几乎无副作用的"保险措施"。

### 2.5 划分结果

```python
print(f"    训练集: {len(X_train):,} (VIVO: {y_train.sum():,})")
print(f"    测试集: {len(X_test):,} (VIVO: {y_test.sum():,})")
```

输出：

```
    训练集: 56,000 (VIVO: 22,932)
    测试集: 24,000 (VIVO: 9,828)
```

- 训练集 56,000 条，其中 VIVO 22,932 条（22,932/56,000 = 40.95%）
- 测试集 24,000 条，其中 VIVO 9,828 条（9,828/24,000 = 40.95%）
- 两者 VIVO 比例完全一致，验证了分层抽样的效果。

***

## 三、保存缺失掩码

```python
train_missing_mask = X_train.isnull()
test_missing_mask = X_test.isnull()
```

### 3.1 这两行代码做了什么

- `X_train.isnull()` 返回一个与 `X_train` 形状相同的**布尔 DataFrame**，True 表示该位置是缺失值，False 表示非缺失。
- 把它存到 `train_missing_mask` 变量中，后续可以用来分析"插补前后哪些位置被改动了"。

### 3.2 为什么要在编码前保存缺失掩码？

因为后续的编码和插补操作会**改变** **`X_train`** **的值**：

1. **编码阶段**：分类特征的缺失值会被 `transform_with_unknown` 函数保留为 NaN（不编码），但非缺失值会被编码成整数。
2. **插补阶段**：所有缺失值会被插补方法（Mean/KNN/MICE）填充成数值。

如果在插补后再想知道"原来哪些位置是缺失的"，就**无法从插补后的数据中恢复**了——因为缺失值已经被填上数字。所以必须在编码和插补**之前**，用 `isnull()` 把缺失位置"拍照"存下来。

### 3.3 缺失掩码的后续用途

在后续模块中，缺失掩码会用于：

- **统计插补前后的分布变化**：比如对比 `Raca.Color` 缺失样本被插补前后的类别分布。
- **评估插补质量**：如果知道真实值（通过 Complete Case），可以对比插补值和真实值的偏差。
- **可视化**：绘制"哪些位置被插补了"的热力图。

> 💡 **小贴士**：在数据预处理流程中，**任何会改变数据的操作之前**，都应该考虑"我以后还需要原始信息吗？"如果需要，就先存下来。这是一种"数据快照"的好习惯。

***

## 四、LabelEncoder 编码（本模块重点之二）

这是本模块**最复杂**的部分。我们先看完整代码，再逐行解释：

```python
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    non_null_train = X_train[col].dropna()
    le.fit(non_null_train.astype(str))

    most_common = non_null_train.value_counts().index[0]

    def transform_with_unknown(x):
        if pd.isna(x):
            return np.nan
        x_str = str(x)
        if x_str in le.classes_:
            return le.transform([x_str])[0]
        else:
            return le.transform([most_common])[0]

    X_train[col] = X_train[col].apply(transform_with_unknown)
    X_test[col] = X_test[col].apply(transform_with_unknown)
    label_encoders[col] = le
```

### 4.1 为什么需要编码？

sklearn 的插补器（`SimpleImputer`、`KNNImputer`、`IterativeImputer`）和模型（`LogisticRegression`）都**只接受数值输入**。但我们的分类特征（`Gender`、`Diagnostic.means`、`Raca.Color`）是字符串，无法直接喂给这些工具。所以必须先把字符串编码成数字。

### 4.2 `label_encoders = {}` — 存储编码器

创建一个空字典，用于存储每个分类特征的 `LabelEncoder` 实例。后续如果需要把编码后的整数**还原**回原始字符串（如解释模型结果时），可以用 `le.inverse_transform()`。把编码器存起来，是为了保留这个"逆映射"能力。

### 4.3 `for col in categorical_features:` — 遍历分类特征

`categorical_features = ['Gender', 'Diagnostic.means', 'Raca.Color']`，对每个分类特征执行一次编码流程。

### 4.4 `le = LabelEncoder()` — 创建编码器实例

`LabelEncoder` 是 sklearn 的标签编码器，能把一组类别标签映射成 0 \~ N-1 的整数。

**示例**：

```
原始类别: ['M', 'F', 'M', 'F']
fit 后:   classes_ = ['F', 'M']   (按字母排序)
transform: ['M', 'F', 'M', 'F'] → [1, 0, 1, 0]
```

### 4.5 `non_null_train = X_train[col].dropna()` — 先删除缺失值

**LabelEncoder 不能处理 NaN**——如果直接 `le.fit` 包含 NaN 的数据，会报错。所以必须先用 `dropna()` 删除缺失值，只用非缺失值拟合编码器。

**为什么不能让 NaN 也成为一个"类别"？**

因为缺失值不是一种真实的类别——它表示"未知"，不应该被编码成一个固定的整数（那会让模型误以为"缺失"是一种有意义的取值）。我们的策略是：**缺失值保持 NaN，后续用插补方法填充**。

### 4.6 `le.fit(non_null_train.astype(str))` — 转字符串后拟合

**为什么用** **`.astype(str)`** **转成字符串？**

1. **防止数值被当作连续值**：有些分类特征可能是数值形式（如 `Diagnostic.means` 可能是 1-7 的整数），如果不转字符串，`LabelEncoder` 会把它们当数值处理，可能引入"序数关系"的误解。转成字符串后，它们被当作离散类别。
2. **统一类型**：如果一列里混有字符串和数值（如 `'M'` 和 `1`），`LabelEncoder` 会报类型错误。转成字符串后类型统一。
3. **安全兜底**：即使原始数据是纯字符串，`.astype(str)` 也不会有副作用，是一种"防御性编程"。

**`fit`** **做了什么？**

`fit` 会扫描所有非缺失值，提取唯一类别，按字母排序存到 `le.classes_` 属性中。例如：

```
non_null_train = ['M', 'F', 'M', 'F', 'F', 'M', ...]
le.fit(...) 后:
  le.classes_ = array(['F', 'M'], dtype=object)
  # 'F' → 0, 'M' → 1
```

### 4.7 `most_common = non_null_train.value_counts().index[0]` — 找训练集众数

- `non_null_train.value_counts()`：统计每个类别的频数，返回一个按频数降序排列的 Series。
- `.index[0]`：取第一个（频数最高的）类别，即**众数**（mode）。

**示例**：

```
non_null_train = ['M', 'F', 'M', 'F', 'F', 'F', 'M']
value_counts():
  F    4
  M    3
.index[0] = 'F'   # 众数是 'F'
```

**为什么需要众数？**

为了处理测试集中的"未知类别"——如果测试集出现了训练集没见过的类别，就用众数替代（详见 4.8）。

### 4.8 `transform_with_unknown` 函数 — 处理未知类别（核心）

这是本模块**最关键**的函数，用于把单个值转换成编码整数，同时处理三种情况：

```python
def transform_with_unknown(x):
    if pd.isna(x):
        return np.nan
    x_str = str(x)
    if x_str in le.classes_:
        return le.transform([x_str])[0]
    else:
        return le.transform([most_common])[0]
```

#### 三段式逻辑详解

**第 1 段：`if pd.isna(x): return np.nan`** **— 缺失值保持缺失**

- `pd.isna(x)` 判断 `x` 是否是 NaN。
- 如果是缺失值，返回 `np.nan`，**不编码**。
- **为什么？** 因为缺失值要留给后续的插补方法处理。如果在这里把 NaN 编码成某个整数（如 0），插补方法就无法识别"这个位置原本是缺失的"，等于跳过了插补步骤。

**第 2 段：`if x_str in le.classes_: return le.transform([x_str])[0]`** **— 已知类别正常编码**

- `x_str = str(x)`：先把值转成字符串（与 `fit` 时的类型一致）。
- `le.classes_`：训练集中见过的所有类别（`fit` 时学到的）。
- `if x_str in le.classes_`：检查当前值是否是训练集见过的类别。
- `le.transform([x_str])[0]`：把该类别转成整数。`[0]` 是因为 `transform` 返回的是数组，取第一个元素。

**第 3 段：`else: return le.transform([most_common])[0]`** **— 未知类别用众数替代**

- 如果当前值**不在** `le.classes_` 中（即测试集出现了训练集没见过的类别），用众数（`most_common`）的编码值替代。
- **为什么用众数？** 众数是训练集中最常见的类别，用它替代未知类别是"最保守"的选择——至少不会引入训练集没见过的编码值，避免 `transform` 报错。

#### 为什么要处理"未知类别"？

这是机器学习中一个**常见陷阱**。考虑这个场景：

- 训练集 `Gender` 有 `['F', 'M']` 两类。
- 测试集 `Gender` 出现了一个 `'Unknown'`（训练集没见过）。
- 如果直接用 `le.transform(['Unknown'])`，会报错：`ValueError: y contains previously unseen labels`。

`transform_with_unknown` 函数就是为了**优雅地处理这种情况**——不让脚本因为测试集的未知类别而崩溃，而是用众数兜底。

> 💡 **重点概念：处理未知类别的策略**
>
> 在机器学习中，**测试集可能出现训练集没见过的类别**，这叫"unknown category"问题。常见的处理策略有：
>
> | 策略                                    | 做法                    | 优缺点              |
> | ------------------------------------- | --------------------- | ---------------- |
> | **众数替代**（本教程）                         | 用训练集众数替代未知类别          | 简单，但可能引入偏差       |
> | **特殊编码**                              | 给未知类别一个固定编码（如 -1 或 N） | 保留"未知"信息，但模型需能处理 |
> | **OneHot + handle\_unknown='ignore'** | 未知类别全 0 向量            | 更优雅，但维度增加        |
> | **删除样本**                              | 直接删掉含未知类别的行           | 简单，但损失数据         |
>
> 本教程选择"众数替代"，因为它简单且不增加维度，适合教学。生产环境中，`OneHotEncoder(handle_unknown='ignore')` 通常是更好的选择。

### 4.9 `X_train[col].apply(transform_with_unknown)` — 逐元素应用函数

- `.apply(func)` 是 pandas Series 的方法，对**每个元素**应用 `func` 函数。
- `transform_with_unknown` 接收单个值 `x`，返回编码后的值（整数或 NaN）。
- 结果赋值回 `X_train[col]`，完成该列的编码。

**为什么用** **`apply`** **而不是** **`le.transform`？**

因为 `le.transform` 无法处理 NaN（会报错），也无法处理未知类别（会报错）。`transform_with_unknown` 函数封装了这两种特殊情况的处理逻辑，所以必须用 `apply` 逐元素应用。

### 4.10 `X_test[col].apply(transform_with_unknown)` — 测试集用同样的函数

**关键点**：测试集用的是**训练集的** **`le`** **和** **`most_common`**，不是重新拟合。

这是机器学习的铁律：**测试集不能参与训练**。如果用测试集数据重新拟合编码器，就等于让测试集"泄露"到了编码过程中，属于数据泄露（data leakage）。

### 4.11 `label_encoders[col] = le` — 保存编码器

把每个特征的 `LabelEncoder` 实例存到字典里，键是特征名。后续如果需要逆变换（把编码整数还原成原始字符串），可以这样用：

```python
le = label_encoders['Gender']
original_labels = le.inverse_transform([0, 1, 1, 0])  # ['F', 'M', 'M', 'F']
```

***

## 五、转换为浮点数

```python
X_train = X_train.astype(float)
X_test = X_test.astype(float)
```

### 5.1 这行代码做了什么

`astype(float)` 把整个 DataFrame 的所有列转成 `float64`（浮点数）类型。

### 5.2 为什么需要转浮点数？

1. **sklearn 插补器要求浮点输入**：`SimpleImputer`、`KNNImputer`、`IterativeImputer` 内部用的是 numpy 浮点运算，如果输入是整数类型，可能报类型错误或行为异常。
2. **支持 NaN**：在 pandas/numpy 中，**整数类型不支持 NaN**——如果把 NaN 存到整数列，pandas 会自动升级成浮点类型。显式 `astype(float)` 避免类型混乱。
3. **统一类型**：编码后，数值特征（`Age`、`year`）可能是 `float64`，分类特征（编码后）可能是 `int64`。`astype(float)` 把所有列统一成 `float64`，方便后续操作。

### 5.3 整数类型不支持 NaN 的陷阱

这是一个容易踩坑的点：

```python
import pandas as pd
import numpy as np

s = pd.Series([1, 2, np.nan], dtype='int64')  # 报错！
# ValueError: Cannot convert non-finite values (NA or inf) to integer

s = pd.Series([1, 2, np.nan])  # 自动变成 float64
# 0    1.0
# 1    2.0
# 2    NaN
# dtype: float64
```

所以，**只要数据中有 NaN，就必须用浮点类型**。本教程的分类特征编码后仍有 NaN（缺失值未编码），所以必须 `astype(float)`。

> ⚠️ **常见问题**：pandas 1.0+ 引入了 `Int64`（注意大写 I）可空整数类型，支持 NaN。但它与 numpy 原生 `int64` 不兼容，sklearn 可能不识别。为了兼容性，建议统一用 `float64`。

***

## 六、为什么要在插补前做编码？

这是本模块的一个重要设计决策，需要深入理解。

### 6.1 sklearn 插补器的要求

sklearn 的三种插补器都**只接受数值输入**：

| 插补器                | 输入要求 | 为什么          |
| ------------------ | ---- | ------------ |
| `SimpleImputer`    | 数值   | 计算均值/中位数需要数值 |
| `KNNImputer`       | 数值   | 计算样本间距离需要数值  |
| `IterativeImputer` | 数值   | 回归模型预测需要数值   |

### 6.2 分类变量必须先编码

如果直接把字符串分类特征喂给插补器，会报错：

```python
# 假设 X_train['Gender'] = ['M', 'F', NaN, 'M']
imputer = KNNImputer()
imputer.fit_transform(X_train)  # 报错！无法计算 'M' 和 'F' 的距离
```

所以必须先把字符串编码成数字，插补器才能工作。

### 6.3 但缺失值要保持为 NaN

编码时有一个关键约束：**缺失值不能被编码成某个数字**，必须保持为 NaN。原因：

1. **保留"缺失"信息**：如果缺失值被编码成 0，插补器无法区分"原本是 0"和"原本是缺失"。
2. **让插补器识别缺失**：插补器通过 `np.isnan` 识别缺失位置，如果缺失值被编码成数字，插补器就找不到要插补的位置了。

这就是 `transform_with_unknown` 函数中 `if pd.isna(x): return np.nan` 这一行的作用——**缺失值"穿透"编码过程，保持 NaN**。

### 6.4 编码→插补的完整流程

```
原始分类特征 (字符串 + NaN)
    ↓ LabelEncoder 编码
编码后特征 (整数 + NaN)    ← 缺失值保持 NaN
    ↓ Imputer 插补
插补后特征 (浮点数, 无缺失) ← 缺失值被填充
```

> 💡 **小贴士**：这种"先编码再插补"的流程是处理分类特征缺失值的标准做法。另一种思路是"先插补再编码"——但那需要用支持字符串的插补方法（如 `SimpleImputer(strategy='most_frequent')`），且无法用 KNN/MICE 等基于距离/回归的方法。本教程为了统一对比三种插补方法，选择"先编码再插补"。

***

## 七、为什么用 LabelEncoder 而非 OneHotEncoder？

这是初学者常问的问题。两种编码器各有优缺点。

### 7.1 LabelEncoder 的工作方式

`LabelEncoder` 把 N 个类别映射成 0 \~ N-1 的整数。

**示例**：`Raca.Color` 有 5 个类别

```
原始: ['Branca', 'Preta', 'Amarela', 'Parda', 'Indigena']
编码: [0,         1,        2,         3,        4]
```

### 7.2 OneHotEncoder 的工作方式

`OneHotEncoder` 把每个类别变成一个独立的二进制列，N 个类别变成 N 列（或 N-1 列）。

**示例**：`Raca.Color` 有 5 个类别

```
原始:     ['Branca']
OneHot:   [[1, 0, 0, 0, 0]]   # 5 列
```

### 7.3 两种编码器的对比

| 维度       | LabelEncoder        | OneHotEncoder     |
| -------- | ------------------- | ----------------- |
| **维度**   | 不增加（5 特征仍是 5 列）     | 增加（5 特征可能变 10+ 列） |
| **序数关系** | 引入虚假序数（类别 2 > 类别 1） | 无序数关系             |
| **距离计算** | 适合 KNN（维度低）         | 高维下距离计算受"维度灾难"影响  |
| **线性模型** | 不适合（引入虚假序数）         | 适合（每个类别独立系数）      |
| **树模型**  | 适合（树能处理任意编码）        | 也可以，但维度增加         |

### 7.4 本教程为什么选 LabelEncoder？

1. **控制维度**：本教程只有 5 个特征，如果用 OneHot，`Diagnostic.means`（7 类）+ `Raca.Color`（5 类）会让维度从 5 涨到 15+，增加复杂度。
2. **适合 KNN 插补**：KNN 基于距离计算，高维数据下距离计算会受"维度灾难"影响，LabelEncoder 保持低维更合适。
3. **简化对比**：本教程的核心是对比插补方法，不希望编码方式成为干扰变量。LabelEncoder 简单直接，让焦点集中在插补上。
4. **教学简洁**：LabelEncoder 的逻辑更直观，适合教学。

### 7.5 LabelEncoder 的缺点

LabelEncoder 引入了**虚假的序数关系**——模型可能误以为"类别 2 > 类别 1 > 类别 0"。对于线性模型（如逻辑回归），这会导致学习到错误的系数。

**本教程如何缓解这个问题？**

- 后续会用 `StandardScaler` 标准化，一定程度上缓解序数影响。
- 逻辑回归对 LabelEncoder 的容忍度尚可（尤其是类别数不多时）。
- 本教程关注的是"插补方法对比"而非"编码方式对比"，所以这个缺点可以接受。

> 💡 **小贴士**：在实际项目中，如果用线性模型（逻辑回归、SVM、神经网络），**优先用 OneHotEncoder**；如果用树模型（随机森林、XGBoost），LabelEncoder 也可以。本教程为简化对比选择 LabelEncoder，后续教程可以改用 OneHot 做对比实验。

***

## 八、本模块运行结果

运行本模块代码后，控制台输出如下：

```
[1] 数据集划分与编码...
    训练集: 56,000 (VIVO: 22,932)
    测试集: 24,000 (VIVO: 9,828)
```

### 关键数字解读

| 指标          | 数值     | 说明                       |
| ----------- | ------ | ------------------------ |
| 训练集样本数      | 56,000 | 80,000 × 70%             |
| 测试集样本数      | 24,000 | 80,000 × 30%             |
| 训练集 VIVO    | 22,932 | 56,000 × 40.95% = 22,932 |
| 测试集 VIVO    | 9,828  | 24,000 × 40.95% = 9,828  |
| 训练集 VIVO 比例 | 40.95% | 与原始数据一致（分层抽样）            |
| 测试集 VIVO 比例 | 40.95% | 与原始数据一致（分层抽样）            |

### 验证分层抽样的效果

- 原始数据 VIVO 比例：40.95%
- 训练集 VIVO 比例：22,932 / 56,000 = 40.95%
- 测试集 VIVO 比例：9,828 / 24,000 = 40.95%

三者完全一致，证明 `stratify=y` 成功保持了标签分布。

### 编码后的数据样貌

编码后，`X_train` 的前几行可能如下（示意）：

| Age  | year   | Gender | Diagnostic.means | Raca.Color |
| ---- | ------ | ------ | ---------------- | ---------- |
| 65.0 | 2018.0 | 1.0    | 3.0              | 0.0        |
| 52.0 | 2020.0 | 0.0    | 1.0              | NaN        |
| NaN  | 2019.0 | 1.0    | 2.0              | 4.0        |
| 70.0 | 2017.0 | 0.0    | NaN              | 2.0        |

- 数值特征（`Age`、`year`）保持原值。
- 分类特征（`Gender`、`Diagnostic.means`、`Raca.Color`）被编码成整数。
- 缺失值保持为 NaN（如第 3 行的 `Age`、第 2 行的 `Raca.Color`）。

***

## 九、小贴士与常见问题

### 💡 小贴士汇总

1. **`stratify=y`** **是不平衡数据的保险**：只要数据有不平衡，就用 `stratify=y`，几乎无副作用。
2. **测试集不能参与训练**：编码器用训练集 `fit`，测试集只能 `transform`，不能重新 `fit`，否则就是数据泄露。
3. **缺失值要"穿透"编码**：编码时缺失值保持 NaN，不能被编码成数字，否则插补器无法识别。
4. **整数类型不支持 NaN**：有缺失值的列必须用浮点类型，`astype(float)` 是标准做法。
5. **处理未知类别要有预案**：测试集可能出现训练集没见过的类别，提前用众数兜底，避免 `transform` 报错。
6. **保存缺失掩码**：在改变数据前用 `isnull()` 拍照存档，后续分析插补效果时需要。

### ⚠️ 常见问题

**Q1：为什么** **`LabelEncoder`** **不能直接处理 NaN？**

A：因为 `LabelEncoder` 的设计假设输入是离散类别标签，NaN 不是类别。如果 `fit` 时传入 NaN，会报错。所以必须先 `dropna` 再 `fit`，编码时用自定义函数让 NaN 保持 NaN。

**Q2：为什么测试集要用训练集的编码器，不能重新拟合？**

A：因为测试集不能参与训练。如果用测试集数据重新 `fit` 编码器，编码映射可能不同（如训练集 `'F'→0`，测试集重新 fit 可能 `'F'→1`），导致模型学到的是错误的映射。更严重的是，这属于**数据泄露**——测试集信息"泄露"到了预处理阶段，会让评估结果过于乐观。

**Q3：`stratify=y`** **和** **`class_weight='balanced'`** **有什么区别？**

A：两者都处理不平衡，但作用阶段不同：

- `stratify=y`（划分时）：保证训练集和测试集的标签分布一致，是**数据层面**的处理。
- `class_weight='balanced'`（建模时）：让模型对少数类赋予更高权重，是**模型层面**的处理。
  两者可以同时用，互不冲突。

**Q4：为什么用众数替代未知类别，而不是中位数或均值？**

A：因为分类变量没有均值/中位数的概念（类别不是数值）。众数是分类变量最合理的"代表值"——它是训练集中最常见的类别，用它替代未知类别是最保守的选择。

**Q5：`astype(float)`** **会损失精度吗？**

A：对于本教程的数据不会。`Age` 和 `year` 本来就是整数，但值域远在 `float64` 的精确表示范围内（`float64` 能精确表示 -2^53 到 2^53 的所有整数）。编码后的分类特征是 0\~N-1 的小整数，更不会有精度问题。

**Q6：为什么不用** **`OneHotEncoder`？**

A：本教程为了简化对比，选择 LabelEncoder。OneHot 的优点是不引入序数关系，但会增加维度，且对 KNN 距离计算不友好。实际项目中，用线性模型时优先 OneHot，用树模型时 LabelEncoder 也可以。

**Q7：`transform_with_unknown`** **函数为什么定义在循环内部？**

A：因为它用到了循环变量 `le` 和 `most_common`。如果定义在循环外，需要把这些作为参数传入，代码更啰嗦。定义在循环内部，函数可以直接"捕获"这些变量（Python 的闭包特性）。注意：这种写法在循环中定义函数有一个陷阱——闭包捕获的是变量的引用，不是值。但本教程中每次循环都重新定义函数，所以没有问题。

***

## 十、本模块小结

### 核心知识点回顾

1. **特征矩阵和标签向量提取**：
   - `X = df_sample[feature_names].copy()`：提取特征，保留 DataFrame 格式。
   - `y = df_sample['target'].values`：提取标签，转成 numpy 数组。
2. **`train_test_split`** **参数详解**：
   - `test_size=0.3`：30% 测试集，70% 训练集。
   - `random_state=RANDOM_STATE`：固定随机种子，保证可复现。
   - `stratify=y`：分层抽样，保持训练集和测试集的标签分布一致——**不平衡数据必备**。
3. **缺失掩码**：`X_train.isnull()` 在编码前保存缺失位置，用于后续分析插补效果。
4. **LabelEncoder 编码流程**：
   - `LabelEncoder()` 创建编码器。
   - `dropna()` 先删除缺失值（LabelEncoder 不能处理 NaN）。
   - `astype(str)` 转字符串（防止数值被当作连续值）。
   - `fit()` 在训练集上学习类别→整数的映射。
   - `value_counts().index[0]` 找众数，用于兜底未知类别。
5. **`transform_with_unknown`** **函数**（处理未知类别）：
   - 缺失值 → 返回 NaN（保持缺失，留给插补器）。
   - 已知类别 → 返回编码整数。
   - 未知类别 → 返回众数编码（兜底，避免报错）。
6. **`astype(float)`** **转浮点**：sklearn 插补器要求浮点输入，且整数类型不支持 NaN。
7. **设计决策**：
   - **为什么先编码再插补**：sklearn 插补器只接受数值输入，分类变量必须先编码。
   - **为什么用 LabelEncoder**：控制维度、适合 KNN、简化对比（缺点是引入序数关系，但本教程可接受）。

### 数据概况

- 训练集：56,000 条（VIVO 22,932，40.95%）
- 测试集：24,000 条（VIVO 9,828，40.95%）
- 分层抽样保证训练集和测试集标签分布一致。

### 下一模块预告

本模块完成了数据划分和编码，得到了数值化、含 NaN 的训练集和测试集。下一模块（**模块 2：四种插补策略比较**）将：

- 实现 Complete Case（删除缺失行）作为基线
- 用 `SimpleImputer` 实现均值插补
- 用 `KNNImputer` 实现 KNN 近邻插补
- 用 `IterativeImputer` 实现 MICE 多重插补
- 对每种插补方法训练逻辑回归，比较 AUC、Recall、Brier Score
- 绘制 ROC 曲线和校准曲线，可视化对比

***

