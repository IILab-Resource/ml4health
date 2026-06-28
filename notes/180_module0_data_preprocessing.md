# 模块 0：数据预处理与 DML 输入准备

> 本模块是案例教程 18「异质性处理效应 (HTE) — 双重机器学习 (DML) 方法」的**起点模块**。在用 DROrthoForest 和 CausalForestDML 估计癌症转移对生存的因果效应之前，我们必须先把数据加载进内存、构造结果变量 Y、从 Extension 列构造处理变量 T、明确区分异质性特征 X 与控制变量 W、对类别特征做编码、对 Age 做标准化、子采样到 3000 条，并准备好 DML 模型所需的 NumPy 输入。
>
> 本模块最核心的知识点有三个：**一是因果推断四要素 Y/T/X/W 的定义**——为什么 T 来自 Extension 列（LOCALIZADO=0, METÁSTASE=1），为什么 Age 是异质性特征 X 而其他变量是控制变量 W；**二是 LabelEncoder 与 StandardScaler 的配合使用**——前者把字符串类别转成整数，后者把 Age 缩放成均值 0、标准差 1；**三是子采样到 3000 条的原因**——DML 模型（特别是 DROrthoForest）计算量大，需要在精度与速度之间权衡。

---

## 学习目标

学完本模块后，你将能够：

1. **理解因果推断四要素 Y/T/X/W 的含义**：知道结果变量 Y、处理变量 T、异质性特征 X、控制变量 W 各自的角色，以及它们在 DML 框架中如何配合。
2. **掌握处理变量 T 的构造逻辑**：明白为什么 T 来自 Extension 列，LOCALIZADO=0 和 METÁSTASE=1 的临床含义，以及 `dropna(subset=['T'])` 的必要性。
3. **区分异质性特征 X 与控制变量 W**：知道为什么 Age 是 X（效应可能随年龄变化），而 Gender、Raca.Color、year、Laterality、Diagnostic.means 是 W（需要控制的混淆变量）。
4. **掌握 LabelEncoder 的使用**：理解如何把字符串类别（如 'M'/'F'）转成整数（0/1），以及 `fit_transform` 的工作原理。
5. **掌握 StandardScaler 的使用**：理解为什么对 Age 标准化，公式 `z = (x - μ) / σ` 的含义，以及标准化对 DML 模型的影响。
6. **理解子采样到 3000 条的原因**：知道 DML 计算量大（交叉拟合 + 多棵树），3000 条是精度与速度的折中。
7. **理解 X_test = np.linspace(age_min, age_max, 50) 的作用**：知道为什么要在 Age 范围内取 50 个等距点用于后续 CATE 曲线绘制。

---
 

## 一、导入必要的库

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import time

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn import metrics

from econml.orf import DROrthoForest
from econml.dml import CausalForestDML
from econml.sklearn_extensions.linear_model import WeightedLasso

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
```

### 1.1 基础库
 

### 1.2 sklearn 模块详解
 

### 1.3 econml 模块（本教程核心）

#### `from econml.orf import DROrthoForest`

**DROrthoForest**（Double Robust Orthogonal Random Forest）是 econml 提供的**双重稳健正交随机森林**。它是本教程的**主角模型之一**。

> 💡 **重点概念：DROrthoForest 的双重稳健性**
>
> "双重稳健"（Double Robust）的含义是：**只要结果模型 E[Y|X, W] 或倾向模型 P(T=1|X, W) 中至少一个正确设定，因果效应估计就是一致的（渐近无偏）**。
>
> 这在实践中非常重要，因为我们很少能确保模型完全正确。双重稳健性给了我们"两次机会"——即使一个模型设定错误，只要另一个正确，估计仍然可靠。
>
> "正交"（Orthogonal）的含义是：把"效应估计"与"讨厌参数估计"分离，让效应估计对模型误差不敏感。

#### `from econml.dml import CausalForestDML`

**CausalForestDML**（Causal Forest DML）是 econml 提供的**因果森林 DML**。它是本教程的**另一个主角模型**。

与 DROrthoForest 相比，CausalForestDML 更轻量、训练更快、内置交叉拟合、支持更多推断功能（置信区间、p 值）。

#### `from econml.sklearn_extensions.linear_model import WeightedLasso`

**WeightedLasso** 是 econml 提供的**加权 Lasso 回归**。它扩展了 sklearn 的 Lasso，支持样本权重。在 DROrthoForest 中用作 `model_Y_final`——DML 最后阶段的回归模型，叶子节点内根据样本权重做加权 Lasso 估计。
 
---

## 三、路径配置与目录创建
 
---
 

## 四、数据加载与目标变量构造

```python
print("\n[0] 加载数据...")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})

print(f"    原始数据集形状: {df.shape}")
print(f"    目标分布: 存活={df['target'].sum()}, 死亡={(df['target']==0).sum()}")
```

### 4.1 `pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')`

读取 CSV 文件，返回 `DataFrame`。三个参数的含义：

- **`DATA_PATH`**：文件路径，由 `os.path.join()` 拼接得到。
- **`low_memory=False`**：关闭"低内存模式"。pandas 默认分块读取 CSV 来节省内存，但这会导致类型推断不一致。设为 `False` 让 pandas 一次性读完，确保类型正确。
- **`encoding='latin-1'`**：用 Latin-1 编码读取。本数据集包含一些非 ASCII 字符（如葡萄牙语的重音符号 `Á`、`É`），用默认的 UTF-8 会报错。Latin-1 是一种"宽容"的编码，能处理任意字节而不报错。

### 4.2 `df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})`

构造结果变量 Y。`Status.Vital` 列包含两个字符串值：

- `VIVO`（葡萄牙语"存活"）→ 1（正类）
- `MORTO`（葡萄牙语"死亡"）→ 0（负类）

`map()` 方法把字典映射应用到 Series 的每个元素。这是二分类任务的标准做法。

> 💡 **重点概念：为什么用 `map` 而不是 `==`？**
>
> `map` 对字典中没有定义的值（包括 NaN）返回 NaN，保持缺失状态。而 `(df['Status.Vital'] == 'VIVO').astype(int)` 会把 NaN 误判为 0（因为 IEEE 754 规定任何与 NaN 的比较都返回 False）。所以 `map` 更安全，能让后续的 `dropna` 正确识别并删除缺失样本。
 

---

## 五、定义处理变量 T（本模块核心）

这是本模块**最重要**的部分。我们先看代码：

```python
# --- 定义处理变量 T: 从 Extension 创建 ---
# T=1: METÁSTASE (转移), T=0: LOCALIZADO (局限)
# 这样 T 就代表了"癌症是否已发生转移"
df['T'] = np.nan
mask_local = df['Extension'] == 'LOCALIZADO'
mask_meta  = df['Extension'] == 'METÁSTASE'
df.loc[mask_local, 'T'] = 0
df.loc[mask_meta,  'T'] = 1

# 删除 T 缺失的样本
df = df.dropna(subset=['T'])
print(f"    有扩展信息(局部/转移)的样本: {len(df)}")
```

### 5.1 为什么 T 来自 Extension 列？

在因果推断中，**处理变量 T** 是我们想研究其因果效应的"干预"。本教程的因果问题是："癌症转移对生存的影响有多大？"

所以 T 应该表示"是否发生转移"。数据集中的 `Extension` 列正好记录了肿瘤的扩展程度：

| Extension 原值 | 含义 | T 值 |
|---------------|------|------|
| `LOCALIZADO` | 局限期（肿瘤未扩散） | 0（对照组） |
| `METÁSTASE` | 转移期（癌症已扩散） | 1（处理组） |
| 其他值（如 `REGIONAL`） | 区域性扩展 | NaN（删除） |

> 💡 **重点概念：为什么只保留 LOCALIZADO 和 METÁSTASE 两类？**
>
> `Extension` 列可能还有其他类别（如 `REGIONAL` 表示区域性扩散）。本教程只关心"局部 vs 转移"这个二值对比，所以把其他类别设为 NaN 并删除。这样 T 就是一个干净的二值变量，适合用 DML 估计因果效应。
>
> 如果保留三类，就需要用多值处理效应方法（如 `discrete_treatment` 配合多个 T 值），复杂度增加。本教程聚焦二值处理，简化分析。

### 5.2 逐行解释 T 的构造

#### `df['T'] = np.nan`

先创建一个全 NaN 的 `T` 列。这是"安全初始化"——任何没有被显式赋值的行，T 都是 NaN，后续会被 `dropna` 删除。

#### `mask_local = df['Extension'] == 'LOCALIZADO'`

创建布尔掩码（boolean mask）：`Extension` 列等于 `'LOCALIZADO'` 的行为 True，否则为 False。这是一个布尔 Series，长度等于 df 的行数。

#### `mask_meta = df['Extension'] == 'METÁSTASE'`

同理，创建"转移"的布尔掩码。注意 `METÁSTASE` 包含葡萄牙语重音符号 `Á`，所以读取数据时必须用 `encoding='latin-1'`，否则这个字符串会变成乱码，匹配失败。

#### `df.loc[mask_local, 'T'] = 0`

用 `.loc` 按布尔掩码赋值：把 `mask_local` 为 True 的行的 `T` 列设为 0。这些是"局部期"患者，作为对照组。

#### `df.loc[mask_meta, 'T'] = 1`

把"转移期"患者的 `T` 列设为 1，作为处理组。

#### `df = df.dropna(subset=['T'])`

删除 `T` 列为 NaN 的行——即 `Extension` 既不是 `LOCALIZADO` 也不是 `METÁSTASE` 的样本。`subset=['T']` 表示只检查这一列。

> ⚠️ **注意：`METÁSTASE` 的重音符号**
>
> 如果你看到代码运行后 `T` 列全是 NaN，最可能的原因是 `METÁSTASE` 的重音符号 `Á` 没有正确读取。确保 `pd.read_csv` 用了 `encoding='latin-1'`。如果用 UTF-8 读取，`Á` 可能变成乱码，导致 `== 'METÁSTASE'` 永远为 False。

---

## 六、定义异质性特征 X 和控制变量 W（本模块核心）

```python
# --- 定义异质性特征 X 和控制变量 W ---
# X: 我们认为效应可能随这些特征变化 (异质性来源)
# W: 需要控制其混淆效应的变量
feature_cols_X = ['Age']  # 主要异质性特征
feature_cols_W = [
    'Gender', 'Raca.Color', 'year', 'Laterality', 'Diagnostic.means'
]

# 选择可用列
available_X = [c for c in feature_cols_X if c in df.columns]
available_W = [c for c in feature_cols_W if c in df.columns]
all_features = available_X + available_W
print(f"    异质性特征 X: {available_X}")
print(f"    控制变量 W:   {available_W}")
```

### 6.1 因果推断四要素 Y/T/X/W

这是本模块**最重要的概念**。DML 框架需要四个变量：

| 变量 | 名称 | 本教程取值 | 角色 |
|------|------|-----------|------|
| **Y** | 结果变量 (Outcome) | `target`（存活=1, 死亡=0） | 我们想预测的变量 |
| **T** | 处理变量 (Treatment) | `T`（转移=1, 局部=0） | 我们想研究其因果效应的"干预" |
| **X** | 异质性特征 (Heterogeneity Features) | `Age` | 效应可能随 X 变化，CATE = f(X) |
| **W** | 控制变量 (Controls/Confounders) | Gender, Raca.Color, year, Laterality, Diagnostic.means | 需要控制的混淆变量 |

> 💡 **重点概念：X 和 W 的本质区别**
>
> **X（异质性特征）**：我们认为处理效应 T→Y 可能随 X 变化。例如，转移对生存的影响可能随年龄变化——老年患者转移后预后更差，年轻患者相对较好。DML 估计的就是 CATE(x) = E[Y(1) - Y(0) | X=x]，即条件平均处理效应作为 X 的函数。
>
> **W（控制变量）**：这些变量同时影响 T 和 Y，是混淆因素。如果不控制 W，估计的效应会有偏。例如，年龄越大越容易转移（T=1），年龄越大也越容易死亡（Y=0），如果不控制年龄，会高估转移的负效应。但 W 不影响效应的异质性——我们不需要估计 CATE 作为 W 的函数。
>
> **本教程的设计**：
> - X = `['Age']`：假设转移效应随年龄变化，CATE = f(Age)。
> - W = `['Gender', 'Raca.Color', 'year', 'Laterality', 'Diagnostic.means']`：这些是混淆变量，需要控制，但不假设效应随它们变化。

### 6.2 为什么 X 只有 Age？

本教程只把 `Age` 作为异质性特征 X，原因有三：

1. **临床合理性**：年龄是癌症预后最重要的因子之一，转移效应随年龄变化有明确的生物学基础——老年患者免疫系统较弱，转移后预后更差。
2. **可视化方便**：X 是一维的（只有 Age），CATE 曲线可以画成二维图（X 轴=Age, Y 轴=CATE），直观易读。如果 X 是多维的，CATE 曲线难以可视化。
3. **教学聚焦**：本教程聚焦于 DML 方法的原理，而不是高维异质性特征。用一维 X 让代码更简洁，学生能聚焦于 DML 本身。

> 💡 **小贴士：如何扩展 X 到多维？**
>
> 如果你想研究多个异质性特征，可以这样改：
> ```python
> feature_cols_X = ['Age', 'year', 'Gender']  # 多维 X
> ```
> 但要注意：
> 1. CATE 曲线无法直接可视化（需要固定其他 X 值）。
> 2. 高维 X 需要更多样本才能可靠估计。
> 3. DROrthoForest 对高维 X 的支持有限，CausalForestDML 更适合。

### 6.3 为什么 W 是这 5 个变量？

| W 变量 | 临床含义 | 为什么是混淆变量 |
|--------|---------|----------------|
| `Gender` | 性别 | 某些癌症有性别差异，性别同时影响转移风险和生存 |
| `Raca.Color` | 人种/肤色 | 反映遗传背景和社会经济地位，影响就诊时机和生存 |
| `year` | 诊断年份 | 医疗技术进步，近年患者生存更好，转移诊断率也可能变化 |
| `Laterality` | 侧别（左/右/双侧） | 肿瘤位置影响转移路径和治疗方案 |
| `Diagnostic.means` | 诊断方式 | 反映确诊确定性，间接影响分期和生存 |

这些变量都同时影响 T（是否转移）和 Y（生存），是典型的混淆因素，必须控制。

### 6.4 `available_X` 和 `available_W` 的容错设计

```python
available_X = [c for c in feature_cols_X if c in df.columns]
available_W = [c for c in feature_cols_W if c in df.columns]
```

这是**列表推导式**（list comprehension），过滤掉 df 中不存在的列。这是一种防御性编程——如果数据集缺少某个列（如 `Laterality` 不存在），代码不会报错，而是跳过该列。

- `c for c in feature_cols_X`：遍历 `feature_cols_X` 中的每个列名。
- `if c in df.columns`：只保留 df 中实际存在的列。

### 6.5 `all_features = available_X + available_W`

列表拼接，把 X 和 W 合并成 `all_features`。后续编码和标准化会对 `all_features` 中的所有列操作。

---

 
 

 

## 七、子采样

```python
# --- 子采样 (计算效率) ---
np.random.seed(RANDOM_STATE)
if len(df_model) > N_SAMPLES:
    idx = np.random.choice(len(df_model), N_SAMPLES, replace=False)
    df_model = df_model.iloc[idx].copy()
    print(f"    子采样至 {N_SAMPLES} 条记录")
```

### 7.1 `np.random.seed(RANDOM_STATE)`

设置 NumPy 的全局随机种子为 42。这保证后续的 `np.random.choice()` 调用产生可复现的结果。

### 7.2 子采样逻辑

```python
if len(df_model) > N_SAMPLES:
    idx = np.random.choice(len(df_model), N_SAMPLES, replace=False)
    df_model = df_model.iloc[idx].copy()
```

**关键代码**：从全量数据中**无放回**抽取 3000 条作为实验子集。

- **`if len(df_model) > N_SAMPLES`**：先判断数据量是否超过 3000。如果数据量本身就小于 3000，就不采样，避免报错。
- **`np.random.choice(len(df_model), N_SAMPLES, replace=False)`**：从 `[0, len(df_model))` 范围内抽取 3000 个**不重复**的索引。
  - 第一个参数 `len(df_model)`：抽样的上界（抽取范围是 0 到 len(df_model)-1）。
  - 第二个参数 `N_SAMPLES`：抽取的数量（3000）。
  - **`replace=False`**：**无放回抽样**——每个样本最多被抽中一次。
- **`df_model.iloc[idx].copy()`**：用索引数组 `idx` 选取对应行，`.copy()` 创建副本，避免后续修改影响原 DataFrame。

> 💡 **小贴士：为什么子采样在编码和标准化之后？**
>
> 本教程的顺序是：先在全量数据上编码和标准化，再子采样。这样设计的原因是：
> 1. **编码一致性**：LabelEncoder 在全量数据上 fit，确保所有类别都被学习到。如果在子采样后 fit，可能漏掉稀有类别。
> 2. **标准化一致性**：StandardScaler 在全量数据上 fit，用全量数据的均值和标准差。这样子采样的分布更接近全量数据。
>
> 严格来说，这有轻微的"数据泄露"（子采样看到了全量数据的统计量），但对教学教程影响很小。在生产环境中，应该先划分训练/测试集，再在训练集上 fit 编码器和标准化器。

---

## 八、准备 DML 输入数据

```python
# ============================================================================
# 准备 DML 输入数据
# ============================================================================
Y = df_model['target'].values                       # 结果: 存活
T = df_model['T'].values                            # 处理: 是否转移
X = df_model[available_X].values                    # 异质性特征
W = df_model[available_W].values                    # 控制变量

# 测试点: 在 Age 的范围内取 50 个点
age_min, age_max = X[:, 0].min(), X[:, 0].max()
X_test = np.linspace(age_min, age_max, 50).reshape(-1, 1)

print(f"\n    最终数据形状: Y={Y.shape}, T={T.shape}, X={X.shape}, W={W.shape}")
print(f"    Age 范围: [{age_min:.2f}, {age_max:.2f}]")
```

### 8.1 转换为 NumPy 数组

```python
Y = df_model['target'].values
T = df_model['T'].values
X = df_model[available_X].values
W = df_model[available_W].values
```

把 DataFrame 的列转成 NumPy 数组，因为 econml 的 DML 模型要求数组输入。

- `df_model['target'].values`：结果变量，形状 `(3000,)`。
- `df_model['T'].values`：处理变量，形状 `(3000,)`。
- `df_model[available_X].values`：异质性特征，形状 `(3000, 1)`（只有 Age）。
- `df_model[available_W].values`：控制变量，形状 `(3000, 5)`。

> 💡 **小贴士：`.values` vs `.to_numpy()`**
>
> `.values` 和 `.to_numpy()` 都能把 Series/DataFrame 转成 NumPy 数组。`.to_numpy()` 是 pandas 推荐的新方法（更明确），`.values` 是老方法。两者功能基本相同，本教程用 `.values` 保持简洁。

### 8.2 生成测试点 X_test

```python
age_min, age_max = X[:, 0].min(), X[:, 0].max()
X_test = np.linspace(age_min, age_max, 50).reshape(-1, 1)
```

这是本模块**最后一个关键步骤**。我们生成 50 个等距的 Age 测试点，用于后续绘制 CATE 曲线。

#### `X[:, 0].min()` 和 `X[:, 0].max()`

- `X[:, 0]`：取 X 的第 0 列（即 Age），形状从 `(3000, 1)` 变成 `(3000,)`。
- `.min()` / `.max()`：Age 的最小值和最大值（标准化后的值，大约在 [-2, 3] 范围）。

#### `np.linspace(age_min, age_max, 50)`

`np.linspace` 生成等距数列：

- 第一个参数 `age_min`：起始值。
- 第二个参数 `age_max`：终止值。
- 第三个参数 `50`：生成 50 个点。

返回一个形状为 `(50,)` 的一维数组，包含 50 个等距的 Age 值。

#### `.reshape(-1, 1)`

把一维数组 `(50,)` 重塑成二维数组 `(50, 1)`。这是必要的，因为 DML 模型的 `effect()` 方法要求 X 是二维数组（每行一个样本，每列一个特征）。

- `-1`：自动推断该维度的大小。这里 `50` 个元素重塑成 `(50, 1)`，`-1` 自动推断为 50。
- `1`：第二维大小为 1（只有 Age 一个特征）。

> 💡 **重点概念：为什么生成 X_test？**
>
> DML 模型训练后，我们想看 CATE 如何随 Age 变化。为此需要在 Age 的范围内取一组测试点，对每个点估计 CATE，然后画成曲线。
>
> 50 个点足够画出平滑的曲线，又不会太慢。如果只取 10 个点，曲线会很粗糙；如果取 500 个点，估计耗时增加但曲线没有明显改善。
>
> `X_test` 的形状是 `(50, 1)`，与训练集 X 的形状 `(3000, 1)` 一致（都是二维，第二维=1）。这样 `est.effect(X_test)` 才能正常工作。
---
 

## 小贴士

> 💡 **小贴士 1：DML 计算量大，子采样是必要的**
>
> DROrthoForest 训练 100 棵树，每棵树都要做 Bootstrap 采样 + 局部 DML 估计。计算成本远高于普通随机森林。3000 条样本在普通笔记本上约需 1–3 分钟。如果用 80000 条，可能需要数小时。注释 `# 因果推断计算量大，数据量不宜过大` 正是提醒这个权衡。

> 💡 **小贴士 2：X 和 W 的区分是 DML 的关键**
>
> 在 DML 框架中，X 和 W 的角色完全不同：
> - **X**：效应随 X 变化，CATE = f(X)。模型会估计 f。
> - **W**：混淆变量，需要控制，但不假设效应随 W 变化。模型用 W 预测 Y 和 T，但不估计 CATE(W)。
>
> 如果把 W 误放入 X，模型会尝试估计 CATE 作为 W 的函数，增加维度，估计变难。如果把 X 误放入 W，会丢失异质性信息，CATE 变成常数。

> 💡 **小贴士 3：`METÁSTASE` 的重音符号**
>
> `METÁSTASE` 包含葡萄牙语重音符号 `Á`。读取数据时必须用 `encoding='latin-1'`，否则 `Á` 会变成乱码，导致 `== 'METÁSTASE'` 匹配失败，T 列全是 NaN。如果你看到 T 列全是 NaN，首先检查编码。

> 💡 **小贴士 4：X_test 的形状必须是二维**
>
> `est.effect(X_test)` 要求 X_test 是二维数组。如果 X_test 是一维的 `(50,)`，会报错。所以要用 `.reshape(-1, 1)` 把它变成 `(50, 1)`。这是 econml 的接口要求，与 sklearn 类似。

> 💡 **小贴士 5：标准化对 DML 的影响**
>
> 标准化主要影响 Lasso 和 LogisticRegression 的训练。标准化后：
> - Lasso 的 L1 惩罚对所有系数公平，不会"偏心"大量纲特征。
> - LogisticRegression 的梯度下降收敛更快。
> - CATE 曲线的 X 轴更紧凑（[-2, 3] vs [0, 120]）。
>
> 注意：标准化不直接影响 CATE 的估计值（因为 CATE 是差分 Y(1)-Y(0)，标准化是线性变换，差分不变）。但它影响模型的训练质量，间接影响 CATE 估计的稳定性。

---

## 常见问题

> ❓ **Q1：为什么 T 来自 Extension 列，而不是直接用一个已有的二值列？**
>
> **A**：数据集中没有直接的"是否转移"二值列。`Extension` 列记录了肿瘤扩展程度，包含多个类别（LOCALIZADO、REGIONAL、METÁSTASE 等）。我们从中提取"局部 vs 转移"这个二值对比，构造 T。这样 T 就代表了"癌症是否已发生转移"这个临床有意义的干预。

> ❓ **Q2：为什么 X 只有 Age，不包含其他特征？**
>
> **A**：本教程聚焦于 DML 方法的原理，用一维 X（Age）让 CATE 曲线可以画成二维图，直观易读。如果 X 是多维的，CATE 曲线难以可视化。此外，高维 X 需要更多样本才能可靠估计。教学教程优先简洁性。在实际研究中，可以根据临床知识选择多个异质性特征。

> ❓ **Q3：`dropna()` 会损失多少样本？**
>
> **A**：`dropna()` 删除任何含 NaN 的行。`Raca.Color` 缺失率较高（约 15%），所以会损失一部分样本。本教程接受这个损失，因为 DML 不能处理 NaN 输入。如果想保留更多样本，可以在编码前用 `SimpleImputer` 插补缺失值。

> ❓ **Q4：为什么先编码再子采样，而不是先子采样再编码？**
>
> **A**：先在全量数据上编码和标准化，确保所有类别都被学习到，且统计量（均值、标准差）基于全量数据。这样子采样的分布更接近全量数据。严格来说有轻微"数据泄露"，但对教学教程影响很小。在生产环境中，应该先划分训练/测试集，再在训练集上 fit。

> ❓ **Q5：`np.linspace(age_min, age_max, 50)` 为什么取 50 个点？**
>
> **A**：50 个点足够画出平滑的 CATE 曲线，又不会太慢。如果只取 10 个点，曲线会很粗糙；如果取 500 个点，估计耗时增加但曲线没有明显改善。50 是"精度-速度"的折中。

> ❓ **Q6：为什么 `X_test` 要 `.reshape(-1, 1)`？**
>
> **A**：`np.linspace` 返回一维数组 `(50,)`，但 econml 的 `effect()` 方法要求 X 是二维数组（每行一个样本，每列一个特征）。`.reshape(-1, 1)` 把它变成 `(50, 1)`，与训练集 X 的形状 `(3000, 1)` 一致。`-1` 表示自动推断该维度的大小（这里是 50）。

> ❓ **Q7：标准化后 Age 的范围 `[-2.13, 3.05]` 对应原始年龄多少？**
>
> **A**：标准化公式是 `z = (x - μ) / σ`，反推 `x = z * σ + μ`。假设原始 Age 均值 μ=60、标准差 σ=15，则：
> - z=-2.13 对应 x = -2.13 * 15 + 60 ≈ 28 岁
> - z=3.05 对应 x = 3.05 * 15 + 60 ≈ 106 岁
>
> 所以标准化范围 `[-2.13, 3.05]` 大致对应原始年龄 28–106 岁，覆盖了主要的成年患者群体。

> ❓ **Q8：为什么 `cat_cols` 不包含 `year`？**
>
> **A**：`year` 在原始数据中是数值类型（如 2010、2015），不是字符串，所以 `pd.api.types.is_numeric_dtype` 返回 True，`year` 不在 `cat_cols` 中。它会被当作数值特征标准化。虽然 year 本质上是离散的，但用数值类型处理对 DML 影响不大。

---

## 本模块小结

本模块完成了 HTE-DML 分析的**所有数据准备工作**：

1. **理解了因果推断四要素 Y/T/X/W**：
   - Y = 存活状态（target）
   - T = 是否转移（从 Extension 列构造，LOCALIZADO=0, METÁSTASE=1）
   - X = Age（异质性特征，效应可能随年龄变化）
   - W = Gender, Raca.Color, year, Laterality, Diagnostic.means（控制变量）

2. **掌握了 T 的构造逻辑**：从 Extension 列映射，用布尔掩码赋值，删除非局部/非转移的样本。

3. **区分了 X 和 W 的角色**：X 是效应异质性的来源（CATE = f(X)），W 是需要控制的混淆变量。

4. **完成了类别编码**：LabelEncoder 把 Gender、Raca.Color、Laterality、Diagnostic.means 转成整数。

5. **完成了数值标准化**：StandardScaler 把 Age 和 year 缩放成均值 0、标准差 1。

6. **子采样到 3000 条**：DML 计算量大，3000 条是精度与速度的折中。

7. **准备了 DML 输入**：Y/T/X/W 转成 NumPy 数组，X_test 生成 50 个 Age 测试点。

**最终数据形状**：Y=(3000,), T=(3000,), X=(3000, 1), W=(3000, 5), X_test=(50, 1)。

**下一模块**将进入 DROrthoForest 训练——计算正则化参数 lambda_reg、配置模型参数、训练模型、估计 CATE 和 95% 置信区间。

---
 