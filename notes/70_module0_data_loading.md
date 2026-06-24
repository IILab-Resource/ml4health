# 模块 0：数据加载与基础预处理

> 本模块是案例教程 7「数据泄漏分析」的起点。在正式进入"泄漏版 vs 正确版"的对比实验之前，我们必须先把数据加载进内存、构造目标变量、采样、精选一组有代表性的特征，并对分类变量做标签编码。
>
> 本模块最核心的知识点：**数据泄漏的定义与危害**——这是贯穿整个案例教程的主线，理解了它才能理解后续所有实验的设计意图；

***

## 学习目标

学完本模块后，你将能够：

1. **理解数据泄漏（Data Leakage）的定义**：能够准确说出"任何使用测试集信息的操作都属于数据泄漏"，并举例说明哪些操作容易引发泄漏。
2. **建立"测试集隔离"的思维习惯**：从本模块开始就树立"测试集必须保持纯净"的原则，为后续模块的对比实验做好心理准备。

***

## 一、导入必要的库

本教程相比案例教程 4–6，**新增了** **`SelectKBest`、`f_classif`、`Pipeline`、`cross_val_score`、`GridSearchCV`**，这些是构建"特征选择泄漏"和"CV 泄漏"实验的核心工具。下面是完整的导入代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与前几个教程相同）

####

### 1.2 sklearn 模块详解（重点关注新增的特征选择与 Pipeline）

sklearn（scikit-learn）是 Python 最主流的机器学习库。本教程从 sklearn 的 6 个子模块导入了多个类和函数，下面逐一解释，并标注与前几个教程的区别。

#### `from sklearn.model_selection import train_test_split, cross_val_score` — 数据划分与交叉验证

**`model_selection`** 子模块提供了数据划分、交叉验证、超参数调优等工具。

- **`train_test_split`**：把数据集划分成训练集和测试集。在本教程中，我们会把 30,000 条样本划分成 70% 训练集（21,000 条）和 30% 测试集（9,000 条）。关键参数 `stratify=y` 会做分层抽样，保持训练集和测试集的标签分布一致。
- **`cross_val_score`**（本教程新增）：交叉验证评分函数，用于在训练集上做 K 折交叉验证。本教程的"扩展实验"会用它来演示 CV 泄漏。

> 💡 **重点概念：`cross_val_score`** **与数据泄漏的关系**
>
> `cross_val_score` 本身不会导致泄漏——它会在内部把训练集再分成 K 折，每折独立做预处理（如果你用了 Pipeline）。但如果在调用 `cross_val_score` **之前**就在全数据上做了预处理，那么 CV 的每一折都已经"看过"了其他折的信息，这就是 CV 泄漏。

####

#### `from sklearn.feature_selection import SelectKBest, f_classif` — 特征选择（本教程重点新增）

**`feature_selection`** 子模块提供了特征选择工具。这一行是本教程**最关键的新增**——`SelectKBest` 是"特征选择泄漏"实验的核心工具。

- **`SelectKBest`**：单变量特征选择器，根据某种评分函数选出得分最高的 K 个特征。
- **`f_classif`**：ANOVA F 检验评分函数，衡量每个特征与目标变量之间的线性相关性。F 值越大，特征与目标的相关性越强。

> 💡 **重点概念：特征选择是泄漏的"最危险区"**
>
> `SelectKBest(f_classif, k=4).fit_transform(X, y)` 会在全数据上计算每个特征的 F 值，然后选出 F 值最高的 4 个特征。如果你在全数据上做这个操作，那么"哪些特征被选中"就受到了测试集的影响——这就是特征选择泄漏。
>
> 在高维场景下（如基因组学有 10,000+ 特征），这种泄漏会让 AUC 虚高 0.05\~0.20，是医学 AI 论文中最常见的泄漏来源。

**`SelectKBest`** **关键参数详解**：

| 参数           | 含义     | 本教程取值                         |
| ------------ | ------ | ----------------------------- |
| `score_func` | 评分函数   | `f_classif`（ANOVA F 检验）       |
| `k`          | 选出的特征数 | 2, 3, 4, 5, 6, 7（实验 3 会遍历这些值） |

#### `from sklearn.pipeline import Pipeline` — 流水线（本教程重点新增）

**`Pipeline`** 是 sklearn 提供的流水线工具，用于把多个预处理步骤和模型串联成一个整体。这一行是本教程**最重要的防泄漏工具**。

> 💡 **重点概念：Pipeline 是天然的防泄漏工具**
>
> 当你把预处理和模型放进一个 `Pipeline`，然后调用 `cross_val_score(pipe, X_train, y_train, cv=5)` 时，sklearn 会在每一折内部独立地做 `fit_transform` 和 `transform`——也就是说，每一折的训练集独立计算 μ/σ、独立做特征选择，测试折只用 `transform`。这就天然防止了 CV 过程中的泄漏。
>
> 相反，如果你手动写循环做 CV，很容易不小心在全数据上 `fit` 一次预处理，再在每折上 `transform`——这就泄漏了。

### 1.3 本教程相比前几个教程的 import 变化总结

| 类别   | 案例教程 4                            | 案例教程 7（本教程）                             | 变化原因          |
| ---- | --------------------------------- | --------------------------------------- | ------------- |
| 特征选择 | ❌ 无                               | **`SelectKBest`** **+** **`f_classif`** | 新增特征选择泄漏实验    |
| 流水线  | ❌ 无                               | **`Pipeline`**                          | 用于演示防泄漏的正确做法  |
| 交叉验证 | ❌ 无                               | **`cross_val_score`**                   | 用于演示 CV 泄漏    |
| 标准化器 | `StandardScaler` + `RobustScaler` | 仅 `StandardScaler`                      | 本教程不聚焦标准化方法对比 |
| 采样规模 | `N_SAMPLES = 80000`               | `N_SAMPLES = 30000`                     | 减少计算开销，聚焦泄漏效应 |
| 评估指标 | AUC/Recall/Brier/PR/AP            | AUC/Recall/Brier/ROC/PR                 | 移除 AP，保留核心指标  |

> 💡 **小贴士**：sklearn 的子模块划分遵循机器学习流程的顺序：`model_selection`（划分）→ `preprocessing`（预处理）→ `impute`（插补）→ `feature_selection`（特征选择）→ `linear_model`（建模）→ `metrics`（评估）。本教程新增了 `feature_selection` 和 `pipeline` 两个环节，因为这两个环节正是泄漏的高发区。

***

## 二、路径配置与目录创建

```python
# ============================================================================
# 路径配置
# ============================================================================
BASE_DIR = ""
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

<br />

***

## 三、随机种子与采样规模

```python
RANDOM_STATE = 42
N_SAMPLES = 30000
```

###

***

## 四、加载数据与创建目标变量

```python
print("=" * 70)
print("案例教程 7: 数据泄漏分析")
print("=" * 70)

# ============================================================================
# 0. 数据加载
# ============================================================================
print("\n[0] 加载数据...")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
df = df.dropna(subset=['target'])

np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()

print(f"    样本量: {len(df):,}  (VIVO: {(df['target'] == 1).sum():,} = {(df['target'] == 1).mean()*100:.2f}%)")
```

###

###

***

## 五、特征选择与编码

### 5.1 `feature_cols` 列表的设计哲学

```python
# 选定特征
feature_cols = ['Age', 'year', 'Gender', 'Code.Profession',
                'Diagnostic.means', 'Extension', 'Raca.Color', 'State.Civil']
df_feat = df[feature_cols + ['target']].copy()
```

本教程从全量特征中精选 **8 个特征**，构成"低维安全"的实验场景。下面逐一解释每个特征：

| 特征名                | 数据类型 | 量纲范围      | 临床含义            |
| ------------------ | ---- | --------- | --------------- |
| `Age`              | 数值   | 0–120     | 患者确诊时的年龄（岁）     |
| `year`             | 数值   | 2000–2020 | 确诊年份            |
| `Gender`           | 分类   | 2 类       | 性别（男/女）         |
| `Code.Profession`  | 分类   | 0–9999    | 职业编码（取值范围很大）    |
| `Diagnostic.means` | 分类   | 多类        | 诊断手段（影像/病理/临床等） |
| `Extension`        | 分类   | 多类        | 肿瘤扩展程度          |
| `Raca.Color`       | 分类   | 多类        | 种族/肤色           |
| `State.Civil`      | 分类   | 多类        | 婚姻状态            |

> 💡 **重点概念：为什么精选 8 个特征？**
>
> 本教程的核心实验是"特征选择泄漏"。如果特征数量太少（如 3 个），特征选择的意义不大；如果特征数量太多（如 100 个），泄漏效应会非常显著，但会让"低维安全"的对比失去教学价值。
>
> 8 个特征是一个"教学友好"的规模：
>
> - 足够多，能演示 `SelectKBest(k=4)` 选出 4 个特征的过程
> - 足够少，让泄漏效应在本数据集中较小（Δ ≈ 0.000），从而引出"为什么本数据集泄漏效应小，但高维场景下会很大"的讨论
> - 涵盖数值和分类两种类型，演示 LabelEncoder 的处理

**为什么** **`Code.Profession`** **的取值范围高达 0–9999？**

`Code.Profession` 是职业编码，本教程用它来制造"量纲差异"——Age 跨度约 120、year 跨度约 20、Code.Profession 跨度约 9999。这种巨大差异会让逻辑回归"偏心"大量纲特征，从而凸显标准化的必要性。

### 5.2 LabelEncoder 循环的写法

```python
# 编码分类变量
cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color', 'State.Civil']
for col in cat_cols:
    le = LabelEncoder()
    non_null = df_feat[col].dropna().astype(str)
    le.fit(non_null)
    mc = non_null.value_counts().index[0]
    def encode(x):
        if pd.isna(x): return np.nan
        xs = str(x)
        return le.transform([xs])[0] if xs in le.classes_ else le.transform([mc])[0]
    df_feat[col] = df_feat[col].apply(encode)
```

这段代码对 5 个分类列做标签编码，处理逻辑非常精巧，下面逐行解释。

#### `cat_cols = [...]` — 待编码的分类列

```python
cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color', 'State.Civil']
```

注意 `feature_cols` 中的 8 个特征，有 5 个是分类变量（`cat_cols`），3 个是数值变量（`Age`、`year`、`Code.Profession`）。`Code.Profession` 虽然本质是分类，但已经被编码成数值，所以不需要再编码。

#### `for col in cat_cols:` — 循环处理每个分类列

对每个分类列，独立创建一个 `LabelEncoder`，独立 fit，独立编码。

#### `le = LabelEncoder()` — 创建编码器

`LabelEncoder` 会把字符串类别映射成整数 `[0, 1, 2, ..., n_classes-1]`。

#### `non_null = df_feat[col].dropna().astype(str)` — 准备非空值

- **`dropna()`**：删除缺失值，因为 `LabelEncoder` 不能处理 `NaN`。
- **`astype(str)`**：把所有值转成字符串，确保类型一致（有些列可能是混合类型，如数字和字符串混在一起）。

#### `le.fit(non_null)` — 训练编码器

`fit` 会统计 `non_null` 中所有出现的类别，按字母顺序排序，分配整数编码。例如 `Gender` 列如果有 `'F'` 和 `'M'`，则 `'F' → 0`、`'M' → 1`。

#### `mc = non_null.value_counts().index[0]` — 找出众数（最常见类别）

- **`value_counts()`**：统计每个类别的频次，按频次降序排序。
- **`.index[0]`**：取频次最高的类别（众数）。

`mc` 用于处理"未知类别"——如果测试集出现了训练集没见过的类别，就用众数代替。

> 💡 **重点概念：为什么要用众数作为回退值？**
>
> 在生产环境中，测试集可能出现训练集没见过的新类别（如新职业）。如果直接报错，模型就无法预测。用众数代替是一种简单的处理方式——众数是最常见的类别，用它的"先验概率最高"，风险最小。
>
> 更严谨的做法是用 `OneHotEncoder(handle_unknown='ignore')`，但本教程为了简化，用 LabelEncoder + 众数回退。

#### `def encode(x):` — 定义编码函数

```python
def encode(x):
    if pd.isna(x): return np.nan
    xs = str(x)
    return le.transform([xs])[0] if xs in le.classes_ else le.transform([mc])[0]
```

这个函数处理三种情况：

1. **缺失值（`pd.isna(x)`）**：返回 `np.nan`，保留缺失值标记，后续由 `SimpleImputer` 处理。
2. **已知类别（`xs in le.classes_`）**：用 `le.transform([xs])[0]` 转成整数。
3. **未知类别（不在** **`le.classes_`** **中）**：用众数 `mc` 的编码代替。

**注意**：`le.transform([xs])` 返回的是数组（如 `array([1])`），所以要用 `[0]` 取出第一个元素。

#### `df_feat[col] = df_feat[col].apply(encode)` — 应用编码

`apply(encode)` 会对该列的每个元素调用 `encode` 函数，返回编码后的新列，覆盖原列。

> ⚠️ **常见问题：闭包陷阱**
>
> 这段代码在循环中定义了 `encode` 函数，它引用了循环变量 `le` 和 `mc`。在 Python 中，闭包是"延迟绑定"的——`encode` 函数内部引用的 `le` 和 `mc` 是循环结束时的最终值，而不是定义时的值。
>
> 但在本教程中，`apply(encode)` 是在每次循环内**立即执行**的，所以 `le` 和 `mc` 就是当前迭代的值，没有闭包陷阱。如果你把 `apply(encode)` 移到循环外，就会出问题。

### 5.3 转换为 numpy 数组

```python
X_raw = df_feat[feature_cols].astype(float).values
y = df_feat['target'].values

print(f"    特征: {feature_cols}")
print(f"    特征数: {len(feature_cols)}")
```

- **`df_feat[feature_cols]`**：从 DataFrame 中取出 8 个特征列。
- **`.astype(float)`**：把所有值转成浮点数。这一步是必要的，因为：
  - 编码后的分类列可能是整数类型（int64），而 sklearn 期望 float。
  - 缺失值 `np.nan` 是浮点数，整数列无法表示 NaN。
- **`.values`**：把 DataFrame 转成 numpy 数组（`ndarray`），形状为 `(30000, 8)`。
- **`y = df_feat['target'].values`**：目标变量，形状为 `(30000,)`。

> 💡 **小贴士：为什么 sklearn 期望 float？**
>
> sklearn 的底层实现大量使用 numpy 的向量化运算和 C 扩展，这些运算对 float 类型优化最好。如果传入 int 类型，某些算法可能会报错或行为异常。所以养成"喂给 sklearn 之前先 `astype(float)`"的习惯是安全的。

### 5.4 实际运行结果

运行上述代码后，控制台会输出：

```
    特征: ['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Extension', 'Raca.Color', 'State.Civil']
    特征数: 8
```

至此，我们得到了：

- `X_raw`：形状 `(30000, 8)` 的特征矩阵，包含缺失值（NaN）
- `y`：形状 `(30000,)` 的目标向量，取值 0/1

这两个变量将作为后续所有实验的输入。

***

## 六、数据泄漏的核心定义（重点）

在进入下一模块的对比实验之前，让我们先明确"数据泄漏"的定义。

### 6.1 什么是数据泄漏？

> **数据泄漏（Data Leakage）**：任何在模型训练过程中使用了测试集信息的操作。

数据泄漏是机器学习中最常见也最危险的错误。它不是"知识不足"，而是**思维惯性**——我们容易下意识地用手中所有的数据去做"更好的"预处理。

### 6.2 数据泄漏的危害

数据泄漏最大的危害是**夸大泛化能力**。当模型在训练过程中"偷看"了测试集的信息，它在测试集上的表现就会虚高。当你把这个模型部署到真实场景（遇到完全没见过的数据）时，性能会断崖式下跌。

### 6.3 数据泄漏的常见类型

| 泄漏类型     | 泄漏了什么      | 危害程度    | 本教程是否演示    |
| -------- | ---------- | ------- | ---------- |
| 标准化泄漏    | 均值 μ、标准差 σ | 小       | ✅ 实验 1 & 2 |
| 插补泄漏     | 均值/中位数     | 小       | ✅ 扩展实验     |
| 特征选择泄漏   | 特征排名       | **中-高** | ✅ 实验 3     |
| CV 泄漏    | 最优参数       | 中       | ✅ 扩展实验     |
| 目标编码泄漏   | 目标均值       | **高**   | ❌（未演示）     |
| SMOTE 泄漏 | 类别平衡       | **高**   | ❌（未演示）     |
| 数据重复泄漏   | 患者信息       | **极高**  | ❌（未演示）     |
| 时间泄漏     | 未来信息       | **极高**  | ❌（未演示）     |

### 6.4 防止泄漏的金律

> **如果你对测试集做了任何操作之后再划分数据，你的结果就不可信了。**

正确的工作流是：

1. **先划分**：把数据分成训练集和测试集。
2. **后预处理**：所有 `fit` 操作只在训练集上做，`transform` 操作应用到测试集。
3. **锁住测试集**：在完成所有预处理、特征工程、特征选择之前，绝对不再碰测试集。

> 💡 **构建"模拟竞赛"心态**：
>
> - 训练集 = 你手头所有的数据
> - 测试集 = 现实世界中未来遇到的新患者
>
> 你不能"偷看"未来的患者来帮助你做今天的决策。

***

## 本模块小结

本模块完成了数据泄漏分析的所有准备工作：

1. **导入必要的库**：新增了 `SelectKBest`、`f_classif`、`Pipeline`、`cross_val_score`，这些是构建泄漏实验和防泄漏工具的核心。
2. **路径配置与目录创建**：与系列教程保持一致，提前创建 `img/` 和 `results/` 目录。
3. **随机种子与采样规模**：`RANDOM_STATE = 42` 保证可复现，`N_SAMPLES = 30000` 平衡计算效率与统计可靠性。
4. **数据加载与目标变量创建**：用 `map` 把 `VIVO`/`MORTO` 映射成 1/0，用 `dropna` 过滤无标签样本。
5. **特征选择与编码**：精选 8 个特征构成"低维安全"场景，用 LabelEncoder + 众数回退处理 5 个分类列。
6. **数据泄漏的核心定义**：明确了"任何使用测试集信息的操作都属于数据泄漏"，并列举了 8 种常见泄漏类型。

至此，我们得到了 `X_raw`（形状 `(30000, 8)`）和 `y`（形状 `(30000,)`），为下一模块的"标准化泄漏对比实验"做好了准备。

> **下一模块预告**：在模块 1 中，我们将用 `X_raw` 和 `y` 做第一个对比实验——"全数据标准化 vs 先划分再标准化"，观察标准化泄漏对 AUC 的影响，并绘制 ROC 曲线对比图。

