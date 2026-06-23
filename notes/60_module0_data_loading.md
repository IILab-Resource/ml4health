# 模块 0：数据加载与高维特征构造

> 本模块是案例教程 6「特征空间分析与降维技术」的起点，承接案例教程 4（特征工程）和案例教程 5（特征选择）。在探讨降维、聚类和可视化之前，我们必须先把数据加载进内存、构造一组**有冗余、有共线性、有非线性关系**的高维特征集——这正是降维技术大显身手的舞台。 
>
> 本模块最核心的知识点有三个：**一是高维特征集的构造哲学**——为什么从 11 个基础特征扩展到 22 个特征，故意制造冗余（`Age` 与 `Age_Sq`、`Age_Centered` 高度相关）以演示 PCA 的"去冗余"能力；**二是两套采样规模的设计**——`N_SAMPLES=50000` 用于常规分析与建模，`N_SAMPLES_MANIFOLD=15000` 专门用于 t-SNE/UMAP/聚类等计算密集型任务；**三是** **`SimpleImputer`** **+** **`StandardScaler`** **的流水线顺序**——为什么必须先插补再标准化，以及为什么降维对标准化尤其敏感。

***

## 学习目标

学完本模块后，你将能够：

1. **理解降维技术的动机**：明白为什么"22 个特征无法直接观察"是降维的起点，以及降维如何成为连接高维数据与人类直觉的桥梁。
2. **掌握高维特征集的构造思路**：理解 `raw_feats`（11 个基础特征）+ 11 个派生特征的设计哲学，特别是为什么**故意制造冗余**（`Age`、`Age_Sq`、`Age_Centered`、`Age_Log` 四个高度相关的特征）来演示 PCA 的去相关能力。
3. **理解两套采样规模的实验设计**：知道 `N_SAMPLES=50000` 和 `N_SAMPLES_MANIFOLD=15000` 分别用于哪些模块，以及为什么 t-SNE/UMAP 必须用更小的样本。
4. **掌握** **`LabelEncoder`** **处理缺失值和未知类别的技巧**：理解 `encode` 函数如何同时处理 `NaN` 缺失值和"测试集出现训练集没有的类别"这一常见问题。
5. **深入理解** **`SimpleImputer`** **+** **`StandardScaler`** **的顺序**：明白为什么必须先插补再标准化，以及为什么 PCA 对特征的量纲极其敏感。
6. **理解** **`train_test_split`** **的** **`stratify`** **参数**：知道为什么在不平衡数据集（VIVO 占 70%+）上必须做分层抽样。
7. **掌握派生特征构造的数学原理**：理解 `Age_Sq`（多项式）、`Age_Group`（分箱）、`Gender_x_Age`（交互）、`Age_Log`（非线性变换）四类特征构造技术的本质。
8. **理解"流形学习小样本"的必要性**：知道为什么 t-SNE 的复杂度是 O(n²)，以及 15,000 样本如何平衡可视化效果和计算时间。

***

## 一、导入必要的库

本教程相比案例教程 5（特征选择），**新增了降维和聚类相关的库**，同时**精简掉了**特征选择专用的 `VarianceThreshold`、`SelectKBest`、`RFE`、`BorutaPy` 等。这是因为本教程的核心主题是"降维"和"聚类"，不再聚焦于"特征选择方法对比"。下面是完整的导入代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, brier_score_loss,
                             adjusted_rand_score, silhouette_score,
                             confusion_matrix, roc_curve)
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与前序教程相同）

#### ` ` &#x20;

###  1.2 sklearn 模块详解（重点关注新增的降维与聚类库）

sklearn（scikit-learn）是 Python 最主流的机器学习库。本教程从 sklearn 的多个子模块导入了大量类和函数，下面按功能分组解释。

####

#### `from sklearn.decomposition import PCA` — 主成分分析（本教程核心）

**`decomposition`** 子模块提供了矩阵分解和降维工具。

- **`PCA`**（Principal Component Analysis）：主成分分析，**本教程最核心的降维方法**。它通过线性变换将原始特征投影到新的正交坐标系（主成分），每个主成分按方差大小排序，前几个主成分保留了大部分信息。

PCA 的关键参数：

| 参数             | 含义       | 本教程取值                           |
| -------------- | -------- | ------------------------------- |
| `n_components` | 保留的主成分个数 | `None`（全部保留）、`5`、`10`、`15`、`20` |
| `random_state` | 随机种子     | `42`（虽然 PCA 是确定性的，但习惯性写上）       |

#### `from sklearn.manifold import TSNE` — t-SNE 非线性降维（本教程核心）

**`manifold`** 子模块提供了流形学习（非线性降维）工具。

- **`TSNE`**（t-Distributed Stochastic Neighbor Embedding）：t-分布随机邻域嵌入，**本教程的可视化利器**。它通过保持局部邻域结构，将高维数据映射到 2D 或 3D 空间，常用于发现数据中的簇状结构。

t-SNE 的关键参数：

| 参数             | 含义           | 本教程取值                      |
| -------------- | ------------ | -------------------------- |
| `n_components` | 输出维度         | `2`（二维可视化）                 |
| `perplexity`   | 困惑度，控制"邻居"数量 | `5`、`30`（默认）、`80`          |
| `max_iter`     | 最大迭代次数       | `500`、`1000`               |
| `random_state` | 随机种子         | `42`（但 t-SNE 对种子敏感，本教程会演示） |

> ⚠️ **t-SNE 的计算复杂度**：t-SNE 的复杂度是 O(n²)，n 是样本量。15,000 样本大约需要 30–60 秒，50,000 样本可能需要 10 分钟以上。这就是为什么本教程要单独用一个 15,000 的小样本做流形学习。

#### `from sklearn.cluster import KMeans, AgglomerativeClustering` — 聚类算法（本教程核心）

**`cluster`** 子模块提供了各种聚类算法。

- **`KMeans`**：K 均值聚类，最经典的划分式聚类算法。需要预设簇数 K，通过迭代优化"簇内方差最小"。
- **`AgglomerativeClustering`**：层次聚类（自底向上），不需要预设簇数，可以生成树状图。

K-Means 的关键参数：

| 参数             | 含义          | 本教程取值       |
| -------------- | ----------- | ----------- |
| `n_clusters`   | 簇的数量        | `2`、`3`、`5` |
| `random_state` | 随机种子        | `42`        |
| `n_init`       | 用不同初始值运行的次数 | `10`（取最优结果） |

#### `from sklearn.neighbors import KNeighborsClassifier` — KNN 分类器

- **`KNeighborsClassifier`**：K 近邻分类器，用于模块 1 的"维度灾难"实验。KNN 基于距离度量，在高维空间会失效——这正是维度灾难的最佳演示工具。

####

#### `from scipy.cluster.hierarchy import dendrogram, linkage` — 层次聚类树状图

**scipy** 是科学计算库，这里的两个函数用于绘制层次聚类的树状图（dendrogram）：

- **`linkage`**：计算层次聚类的连接矩阵，`method='ward'` 表示用 Ward 方差最小化准则。
- **`dendrogram`**：根据连接矩阵绘制树状图。

### 1.3 本教程相比案例教程 5 的 import 变化总结

| 类别   | 案例教程 5                                             | 案例教程 6（本教程）                                                           | 变化原因                  |
| ---- | -------------------------------------------------- | --------------------------------------------------------------------- | --------------------- |
| 降维   | ❌ 无                                                | **`PCA`** + **`TSNE`** + `umap`                                       | 本教程核心主题               |
| 聚类   | ❌ 无                                                | **`KMeans`** + **`AgglomerativeClustering`** + `dendrogram`/`linkage` | 聚类分析模块                |
| 分类器  | ❌ 无                                                | **`KNeighborsClassifier`**                                            | 维度灾难实验                |
| 特征选择 | `VarianceThreshold`/`SelectKBest`/`RFE`/`BorutaPy` | 仅 `SelectKBest`/`RandomForest`（模块 9 对比用）                              | 不再聚焦特征选择              |
| 评估指标 | AUC/Recall/Brier                                   | + **`adjusted_rand_score`** + **`silhouette_score`**                  | 新增聚类评估指标              |
| 标准化器 | `StandardScaler`/`RobustScaler`                    | 仅 `StandardScaler`                                                    | 降维用 StandardScaler 即可 |

> 💡 **小贴士**：sklearn 的子模块划分遵循机器学习流程的顺序：`model_selection`（划分）→ `preprocessing`（预处理）→ `impute`（插补）→ `decomposition`/`manifold`（降维）→ `cluster`（聚类）→ `linear_model`/`neighbors`（建模）→ `metrics`（评估）。本教程新增了 `decomposition`、`manifold`、`cluster` 三个子模块，对应降维和聚类两大主题。

***

## 二、路径配置与目录创建

<br />

***

## 三、随机种子与两套采样规模（本教程重点）

```python
RANDOM_STATE = 42
N_SAMPLES = 50000       # 常规分析
N_SAMPLES_MANIFOLD = 15000  # t-SNE / UMAP / 聚类 (计算量大)
```

###

***

## 四、加载数据与创建目标变量

```python
print("=" * 70)
print("案例教程 6: 特征空间分析与降维技术")
print("=" * 70)

# ============================================================================
# 0. 数据加载 & 高维特征构造 (~25 维)
# ============================================================================
print("\n[0] 加载数据与高维特征构造...")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
df = df.dropna(subset=['target'])

np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()

print(f"    样本量: {len(df):,}  (VIVO: {(df['target'] == 1).sum():,} = {(df['target'] == 1).mean() * 100:.2f}%)")
```

### &#x20;

### &#x20;

***

## 五、构造 22 维高维特征集（本教程核心设计）

### 5.1 基础医学特征（11 个）

```python
# ---------- 构造 ~25 维特征集 ----------
# 基础医学特征
raw_feats = ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology',
             'Diagnostic.means', 'Extension', 'Laterality',
             'Raca.Color', 'State.Civil', 'Degree.of.Education']

df_feat = df[raw_feats + ['target']].copy()
```

本教程从原始数据中精选了 **11 个基础医学特征**，涵盖患者人口学、肿瘤学、诊断学三个维度：

| 特征                    | 含义         | 类型       | 量纲范围      |
| --------------------- | ---------- | -------- | --------- |
| `Age`                 | 患者年龄       | 数值       | 0–120     |
| `year`                | 诊断年份       | 数值       | 2000–2015 |
| `Gender`              | 性别         | 分类（2 类）  | 1–2       |
| `Code.Profession`     | 职业代码       | 分类（数值编码） | 0–9999    |
| `Code.of.Morphology`  | 肿瘤形态学代码    | 分类（数值编码） | 0–9999    |
| `Diagnostic.means`    | 诊断手段       | 分类       | 1–6       |
| `Extension`           | 肿瘤扩展程度     | 分类       | 1–9       |
| `Laterality`          | 侧别（左/右/双侧） | 分类       | 1–5       |
| `Raca.Color`          | 种族/肤色      | 分类       | 1–5       |
| `State.Civil`         | 婚姻状态       | 分类       | 1–6       |
| `Degree.of.Education` | 教育程度       | 分类       | 1–6       |

> 💡 **重点概念：为什么选这 11 个特征？**
>
> 这 11 个特征的设计意图是**制造量纲差异和共线性**，为降维演示提供素材：
>
> 1. **量纲差异巨大**：`Age`（0–120）、`year`（2000–2015）、`Code.Profession`（0–9999）三个特征的数值跨度差异巨大。如果不标准化，PCA 会"偏心"`Code.Profession`。
> 2. **分类变量数值化**：`Gender`、`Diagnostic.means` 等分类变量被编码成整数，但整数大小没有顺序意义（如 `Gender=2` 不比 `Gender=1` "大"）。PCA 会把它们当连续变量处理，这是 PCA 的局限。
> 3. **潜在共线性**：`Extension`（扩展程度）和 `Code.of.Morphology`（形态学）可能相关；`Raca.Color`（种族）和 `Degree.of.Education`（教育）可能相关——这些共线性正是 PCA 要解决的。

### 5.2 分类变量标签编码

```python
# 编码分类变量
cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Laterality',
            'Raca.Color', 'State.Civil', 'Degree.of.Education']
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

这段代码对 7 个分类变量做标签编码，处理逻辑非常精巧，逐行解释：

#### `le = LabelEncoder()`

为每个分类列创建一个独立的 `LabelEncoder` 实例。`LabelEncoder` 会把字符串类别映射成整数（如 `'M' → 0, 'F' → 1`）。

#### `non_null = df_feat[col].dropna().astype(str)`

- **`dropna()`**：先删除缺失值，避免 `LabelEncoder` 把 `NaN` 当成一个类别。
- **`astype(str)`**：转成字符串类型。有些分类列可能是混合类型（数字 + 字符串），转成字符串后编码更稳定。

#### `le.fit(non_null)`

用非缺失值拟合编码器，学习"类别 → 整数"的映射关系。

#### `mc = non_null.value_counts().index[0]`

- **`value_counts()`**：统计每个类别的频次，按频次降序排列。
- **`.index[0]`**：取频次最高的类别（众数，mode）。
- **`mc`**：存储众数，用于后续处理"未知类别"。

> 💡 **重点概念：为什么要存众数** **`mc`？**
>
> 在实际应用中，测试集可能出现训练集没有的类别（如训练集 `Gender` 只有 `'M'`/`'F'`，测试集出现 `'Unknown'`）。`LabelEncoder` 没见过 `'Unknown'`，直接 `transform` 会报错。
>
> 本教程的处理策略是：**遇到未知类别，用众数替代**。这是一种简单但实用的策略，避免了"测试集报错"的尴尬。

#### `def encode(x): ...`

这是一个嵌套函数，处理单个值的编码：

```python
def encode(x):
    if pd.isna(x): return np.nan           # 缺失值保持 NaN
    xs = str(x)                             # 转字符串
    return le.transform([xs])[0] if xs in le.classes_ else le.transform([mc])[0]
```

- **`if pd.isna(x): return np.nan`**：缺失值保持 `NaN`，后续用 `SimpleImputer` 统一处理。
- **`xs = str(x)`**：转字符串，确保和 `le.classes_`（也是字符串）匹配。
- **`le.transform([xs])[0] if xs in le.classes_ else le.transform([mc])[0]`**：
  - 如果 `xs` 在编码器见过的类别中（`le.classes_`），返回对应的整数编码。
  - 如果 `xs` 是未知类别，返回众数 `mc` 的编码——把未知类别"归入"最常见的那一类。

#### `df_feat[col] = df_feat[col].apply(encode)`

对整列应用 `encode` 函数，把字符串类别转成整数。

> ⚠️ **常见问题：为什么不用** **`pd.get_dummies`** **做独热编码？**
>
> 独热编码（One-Hot Encoding）会把一个 5 类的分类变量扩展成 5 列 0/1 变量，这会让特征维度爆炸（7 个分类变量可能扩展成 30+ 列）。
>
> 本教程的目标是演示"22 维 → 2 维"的降维效果，如果用独热编码变成 40 维，教学复杂度增加但核心概念不变。所以本教程用标签编码，保持 22 维的简洁特征集。
>
> 但要注意：**标签编码把分类变量当连续变量处理，PCA 会假设"2 比 1 大"这种顺序关系，这是不严谨的**。在实际科研中，分类变量应该用独热编码或专门的降维方法（如 MCA）。

### 5.3 构造 11 个派生特征（故意制造冗余）

```python
# 构造派生特征
df_feat['Age_Group'] = df_feat['Age'].apply(
    lambda a: np.nan if pd.isna(a) else (0 if a < 18 else 1 if a < 40 else 2 if a < 60 else 3 if a < 75 else 4))
df_feat['Age_Sq'] = df_feat['Age'] ** 2
df_feat['Year_From_2000'] = df_feat['year'] - 2000
df_feat['Year_Decade'] = df_feat['year'].apply(
    lambda y: np.nan if pd.isna(y) else (0 if y < 2005 else 1 if y < 2010 else 2))
df_feat['Is_Child'] = (df_feat['Age'] < 18).astype(float)
df_feat['Is_Elderly'] = (df_feat['Age'] >= 75).astype(float)
df_feat['Age_Centered'] = df_feat['Age'] - 60
df_feat['Gender_x_Age'] = df_feat['Gender'] * df_feat['Age']
df_feat['Profession_Log'] = np.log1p(df_feat['Code.Profession'])
df_feat['Age_Log'] = np.log1p(df_feat['Age'])
df_feat['Age_x_Year'] = df_feat['Age'] * df_feat['Year_From_2000']
```

本教程构造了 **11 个派生特征**，分四类技术。**关键设计意图：故意制造冗余和共线性**，让 PCA 有"去冗余"的素材。

#### 5.3.1 分箱特征（Binning）

```python
df_feat['Age_Group'] = df_feat['Age'].apply(
    lambda a: np.nan if pd.isna(a) else (0 if a < 18 else 1 if a < 40 else 2 if a < 60 else 3 if a < 75 else 4))
```

- **`Age_Group`**：把连续年龄分成 5 组：0=儿童（<18）、1=青年（18–39）、2=中年（40–59）、3=老年（60–74）、4=高龄（≥75）。
- **`lambda a: ...`**：用嵌套的三元表达式实现分段函数。
- **`np.nan if pd.isna(a)`**：缺失值保持 `NaN`，不参与分箱。

```python
df_feat['Year_Decade'] = df_feat['year'].apply(
    lambda y: np.nan if pd.isna(y) else (0 if y < 2005 else 1 if y < 2010 else 2))
```

- **`Year_Decade`**：把诊断年份分成 3 个年代：0=2000–2004、1=2005–2009、2=2010+。

> 💡 **冗余设计**：`Age_Group` 和 `Age` 高度相关（都是年龄的信息），`Year_Decade` 和 `year` 高度相关。这些冗余特征会被 PCA 识别并合并到同一个主成分中。

#### 5.3.2 多项式特征（Polynomial）

```python
df_feat['Age_Sq'] = df_feat['Age'] ** 2
```

- **`Age_Sq`**：年龄的平方。用于让线性模型捕捉 U 型非线性关系（如年龄与死亡率的 U 型曲线）。
- **`Age_Sq`** **和** **`Age`** **的相关性极高**（约 0.98），这是**故意制造的共线性**——PCA 会把它们合并到同一个主成分。

#### 5.3.3 中心化与对数变换

```python
df_feat['Year_From_2000'] = df_feat['year'] - 2000
df_feat['Age_Centered'] = df_feat['Age'] - 60
df_feat['Profession_Log'] = np.log1p(df_feat['Code.Profession'])
df_feat['Age_Log'] = np.log1p(df_feat['Age'])
```

- **`Year_From_2000`**：年份减去 2000，把 2000–2015 变成 0–15。这只是**平移**，不改变方差，和 `year` 完全线性相关（相关系数 = 1.0）。
- **`Age_Centered`**：年龄减去 60，把 0–120 变成 -60–60。同样是平移，和 `Age` 完全线性相关。
- **`Profession_Log`**：`np.log1p(x) = log(1 + x)`，对数变换压缩大量纲特征。和 `Code.Profession` 高度相关但非线性。
- **`Age_Log`**：年龄的对数变换，和 `Age` 高度相关。

> 💡 **重点概念：为什么故意制造完全线性相关的特征？**
>
> `Year_From_2000` 和 `year` 的相关系数是 1.0（完全线性相关），这在特征选择中是"灾难"（VIF 会无穷大），但在 PCA 中是"素材"——PCA 会把它们合并成同一个主成分，方差解释率不变，但特征数减少。
>
> 这正是本教程想演示的：**PCA 能完美处理共线性，而特征选择不能**。

#### 5.3.4 交互特征（Interaction）

```python
df_feat['Gender_x_Age'] = df_feat['Gender'] * df_feat['Age']
df_feat['Age_x_Year'] = df_feat['Age'] * df_feat['Year_From_2000']
```

- **`Gender_x_Age`**：性别 × 年龄的交互项。捕捉"性别对死亡率的影响随年龄变化"这种交互效应。
- **`Age_x_Year`**：年龄 × 年份的交互项。捕捉"年龄效应随时间变化"这种队列效应。

#### 5.3.5 布尔特征

```python
df_feat['Is_Child'] = (df_feat['Age'] < 18).astype(float)
df_feat['Is_Elderly'] = (df_feat['Age'] >= 75).astype(float)
```

- **`Is_Child`**：是否儿童（Age < 18），1 是，0 否。
- **`Is_Elderly`**：是否高龄（Age ≥ 75），1 是，0 否。
- **`.astype(float)`**：转成浮点数，因为后续要喂给 sklearn（sklearn 不接受 bool 类型）。

> ⚠️ **常见问题**：`(df_feat['Age'] < 18)` 对 `NaN` 返回 `False`，所以 `Is_Child` 对缺失年龄的样本会是 0（不是儿童）。这在统计上不严谨，但本教程为了简化，不做额外处理。在实际科研中，应该用 `.loc` 把缺失值的 `Is_Child` 设回 `NaN`。

### 5.4 完整特征列表

```python
# --- 全部特征列表 ---
all_features = raw_feats + ['Age_Group', 'Age_Sq', 'Year_From_2000',
    'Year_Decade', 'Is_Child', 'Is_Elderly', 'Age_Centered',
    'Gender_x_Age', 'Profession_Log', 'Age_Log', 'Age_x_Year']

n_dims = len(all_features)
print(f"    特征维度: {n_dims}")
print(f"    特征列表: {all_features}")
```

- **`all_features`**：11 个基础特征 + 11 个派生特征 = **22 个特征**。
- **`n_dims = 22`**：这就是本教程的"高维空间"维度。

**实际运行输出**：

```
    特征维度: 22
    特征列表: ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology', 'Diagnostic.means', 'Extension', 'Laterality', 'Raca.Color', 'State.Civil', 'Degree.of.Education', 'Age_Group', 'Age_Sq', 'Year_From_2000', 'Year_Decade', 'Is_Child', 'Is_Elderly', 'Age_Centered', 'Gender_x_Age', 'Profession_Log', 'Age_Log', 'Age_x_Year']
```

> 💡 **重点概念：22 维的"高维"到底有多高？**
>
> 在机器学习中，"高维"是相对概念：
>
> - **低维**：1–10 维（可以直接画散点图、热力图）
> - **中维**：10–100 维（需要降维才能可视化）
> - **高维**：100–1000 维（必须降维，否则距离度量失效）
> - **超高维**：1000+ 维（如基因表达数据、文本 TF-IDF）
>
> 本教程的 22 维属于"中维"——已经无法直接可视化（最多画 2D 散点图），但还没到"维度灾难"的极端。这是一个适合教学的维度：既能演示降维的必要性，又不会让计算太慢。

### 5.5 特征冗余性分析

让我们分析这 22 个特征的冗余结构（这是 PCA 要解决的问题）：

| 冗余组     | 特征                                                                                                | 相关性       | 冗余原因          |
| ------- | ------------------------------------------------------------------------------------------------- | --------- | ------------- |
| **年龄组** | `Age`, `Age_Sq`, `Age_Group`, `Age_Centered`, `Age_Log`, `Is_Child`, `Is_Elderly`, `Gender_x_Age` | 极高（0.9+）  | 全部由 `Age` 派生  |
| **年份组** | `year`, `Year_From_2000`, `Year_Decade`, `Age_x_Year`                                             | 极高（0.95+） | 全部由 `year` 派生 |
| **职业组** | `Code.Profession`, `Profession_Log`                                                               | 极高（0.99）  | 对数变换          |

**冗余特征数**：22 个特征中，约 14 个是冗余的（由 4 个基础特征派生）。

> 💡 **PCA 的价值**：如果用特征选择，我们很难决定"保留 `Age` 还是 `Age_Centered`"——它们信息一样。但 PCA 会把它们合并成同一个主成分（PC1 可能就是"年龄方向"），完美解决冗余问题。这就是 PCA 相对于特征选择的核心优势。

***

## 六、缺失值插补与标准化

```python
# 转为 float & 插补 + 标准化
X_full = df_feat[all_features].astype(float)
y = df_feat['target'].values

imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_full_arr = scaler.fit_transform(imputer.fit_transform(X_full))
X_full_df = pd.DataFrame(X_full_arr, columns=all_features)
```

### 6.1 `X_full = df_feat[all_features].astype(float)`

- 从 `df_feat` 中取出 22 个特征列，转成浮点数类型。
- **为什么转 float？** sklearn 的算法要求输入是数值型数组，分类变量经过 LabelEncoder 后是整数，派生特征中 `Is_Child` 是浮点数，混合类型转成统一的 float 才能喂给 sklearn。

### 6.2 `y = df_feat['target'].values`

- 取出目标变量，`.values` 把 pandas Series 转成 numpy 数组。

### 6.3 插补与标准化的顺序（重点）

```python
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
X_full_arr = scaler.fit_transform(imputer.fit_transform(X_full))
```

这三行代码的执行顺序是：

1. **`imputer.fit_transform(X_full)`**：用均值插补缺失值。
   - `fit`：计算每列的均值（忽略 NaN）。
   - `transform`：用均值填充 NaN。
2. **`scaler.fit_transform(...)`**：对插补后的数据做标准化。
   - `fit`：计算每列的均值和标准差。
   - `transform`：用 `(x - μ) / σ` 标准化。

> ⚠️ **重点概念：为什么必须先插补再标准化？**
>
> 如果先标准化再插补，会有两个问题：
>
> 1. **StandardScaler 不能处理 NaN**：`scaler.fit()` 遇到 NaN 会报错或返回 NaN。
> 2. **插补会破坏标准化**：如果先标准化（均值 0、标准差 1），再用均值（0）插补缺失值，插补后的数据均值不再是 0，标准化被破坏。
>
> 正确顺序是：**先插补（用原始均值），再标准化（用插补后的均值和标准差）**。这样插补和标准化都能正确完成。

### 6.4 `SimpleImputer(strategy='mean')` 详解

- **`strategy='mean'`**：用列均值填充缺失值。
- 其他可选策略：
  - `'median'`：用中位数（对异常值鲁棒）
  - `'most_frequent'`：用众数（适合分类变量）
  - `'constant', fill_value=...`：用常数填充

> 💡 **为什么用均值而不是中位数？** 本教程的核心是降维，不是插补方法对比。用均值插补最简单，且对 PCA 的影响最小（均值插补不改变列均值，PCA 的第一主成分方向不受影响）。

### 6.5 `StandardScaler` 详解

- **`fit_transform`**：先 `fit`（计算 μ 和 σ），再 `transform`（应用标准化）。
- 标准化后，每列的均值 = 0，标准差 = 1。

> 💡 **重点概念：为什么 PCA 必须标准化？**
>
> 看看本教程的特征量纲差异：
>
> - `Age`：0–120，方差约 400
> - `year`：2000–2015，方差约 20
> - `Code.Profession`：0–9999，方差约 8,000,000
> - `Age_Sq`：0–14400，方差约 8,000,000
>
> 如果不标准化，PCA 的第一主成分会**几乎完全指向** **`Code.Profession`** **和** **`Age_Sq`**（因为它们的方差最大），其他特征的信息被完全忽略。
>
> 标准化后，所有特征的方差都变成 1，PCA 能公平地比较每个特征方向的方差，找到真正"信息最丰富"的方向。

### 6.6 `X_full_df = pd.DataFrame(X_full_arr, columns=all_features)`

- 把标准化后的 numpy 数组转回 DataFrame，保留列名。
- 这个 DataFrame 后续用于查看标准化后的数据分布（虽然本教程不深入展示，但保留列名方便调试）。

***

## 七、数据划分与流形学习小样本

### 7.1 训练集/测试集划分

```python
# 划分
X_train, X_test, y_train, y_test = train_test_split(
    X_full_arr, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y)
print(f"    训练集: {len(X_train):,}  测试集: {len(X_test):,}")
```

#### `train_test_split` 参数详解

- **`X_full_arr`**：特征矩阵（50,000 × 22）。
- **`y`**：目标变量（50,000，）。
- **`test_size=0.3`**：30% 作为测试集，70% 作为训练集。
- **`random_state=RANDOM_STATE`**：固定随机种子，确保划分可复现。
- **`stratify=y`**：**分层抽样**，保持训练集和测试集的标签分布一致。

> 💡 **重点概念：为什么必须** **`stratify=y`？**
>
> 本数据集 VIVO 占 71.68%，MORTO 占 28.32%。如果不分层，可能训练集里 VIVO 占 75%、测试集里 VIVO 占 65%——分布不一致会让模型评估有偏差。
>
> 分层抽样后，训练集和测试集的 VIVO 比例都是 71.68%，评估更公平。

**实际运行输出**：

```
    训练集: 35,000  测试集: 15,000
```

### 7.2 流形学习小样本采样

```python
# 降维用的小样本 (t-SNE / UMAP / 聚类)
np.random.seed(RANDOM_STATE)
if len(X_full_arr) > N_SAMPLES_MANIFOLD:
    midx = np.random.choice(len(X_full_arr), N_SAMPLES_MANIFOLD, replace=False)
    X_manifold = X_full_arr[midx]
    y_manifold = y[midx]
else:
    X_manifold = X_full_arr.copy()
    y_manifold = y.copy()
print(f"    流形学习样本: {len(X_manifold):,}")
```

#### 这段代码的逻辑

1. **`np.random.seed(RANDOM_STATE)`**：固定随机种子，确保采样可复现。
2. **`if len(X_full_arr) > N_SAMPLES_MANIFOLD`**：如果总样本量（50,000）大于流形学习样本量（15,000），则采样。
3. **`np.random.choice(len(X_full_arr), N_SAMPLES_MANIFOLD, replace=False)`**：无放回抽取 15,000 个索引。
4. **`X_manifold = X_full_arr[midx]`**：按索引取出小样本特征。
5. **`y_manifold = y[midx]`**：按索引取出对应的标签（保持 X 和 y 对应）。

#### `X_manifold` 和 `y_manifold` 的用途

| 变量                 | 用途                   | 使用模块     |
| ------------------ | -------------------- | -------- |
| `X_manifold`       | t-SNE、UMAP、聚类的输入     | 模块 2、3、4 |
| `y_manifold`       | 可视化时给点上色（VIVO/MORTO） | 模块 2、3   |
| `X_train`/`X_test` | KNN、LogReg 的训练/测试    | 模块 1、5   |
| `y_train`/`y_test` | KNN、LogReg 的标签       | 模块 1、5   |

> 💡 **重点概念：为什么流形学习用小样本，而建模用全样本？**
>
> - **流形学习（t-SNE/UMAP）**：目的是**可视化**，不需要严格的训练/测试划分。15,000 个点画在 2D 平面上已经足够密集，再多就重叠看不清了。而且 t-SNE 的 O(n²) 复杂度让大样本不可行。
> - **建模（KNN/LogReg）**：目的是**评估性能**，需要严格的训练/测试划分和足够大的样本量来保证统计稳定性。50,000 样本的 70/30 划分（35,000 训练 + 15,000 测试）足够稳定。

**实际运行输出**：

```
    流形学习样本: 15,000
```

***

## 八、本模块数据流总结

```
原始数据 (209,758 行)
    │
    ├── map({'VIVO':1, 'MORTO':0}) → target 列
    ├── dropna(subset=['target']) → 过滤缺失标签
    │
    ▼
有标签数据 (~209,000 行)
    │
    ├── np.random.choice(50000) → 采样
    │
    ▼
采样数据 (50,000 行)
    │
    ├── 取 11 个基础特征 (raw_feats)
    ├── LabelEncoder 编码 7 个分类变量
    ├── 构造 11 个派生特征 (Age_Sq, Age_Group, ...)
    │
    ▼
22 维特征集 (50,000 × 22)
    │
    ├── SimpleImputer(mean) → 插补缺失值
    ├── StandardScaler → 标准化 (均值0, 标准差1)
    │
    ▼
标准化特征矩阵 X_full_arr (50,000 × 22)
    │
    ├── train_test_split(70/30, stratify=y)
    │   ├── X_train (35,000 × 22) → 模块 1 (KNN), 模块 5 (LogReg)
    │   └── X_test  (15,000 × 22) → 模块 1 (KNN), 模块 5 (LogReg)
    │
    └── np.random.choice(15000) → 流形学习小样本
        ├── X_manifold (15,000 × 22) → 模块 2 (PCA/t-SNE/UMAP), 模块 4 (聚类)
        └── y_manifold (15,000,)     → 模块 2 (上色), 模块 4 (ARI 评估)
```

***

## 小贴士

1. **降维教程的特征设计哲学**：本教程故意构造了 14 个冗余特征（由 4 个基础特征派生），目的是让 PCA 有"去冗余"的素材。在实际科研中，你应该避免这种冗余——除非你想演示 PCA。
2. **两套样本量的设计**：`N_SAMPLES=50000` 用于建模，`N_SAMPLES_MANIFOLD=15000` 用于流形学习。这种"双样本量"设计在大型数据集的降维分析中很常见，平衡了统计稳定性和计算效率。
3. **`stratify=y`** **是不平衡数据集的必备参数**：本数据集 VIVO 占 71.68%，如果不分层抽样，训练集和测试集的类别比例可能差异很大，导致评估有偏。
4. **先插补再标准化**：这是 sklearn 预处理的黄金顺序。如果反过来，`StandardScaler` 会因为遇到 NaN 而报错。
5. **`LabelEncoder`** **处理未知类别的技巧**：用众数替代未知类别是一种简单策略。更严谨的做法是用 `OneHotEncoder(handle_unknown='ignore')`，但会增加特征维度。
6. **t-SNE 的 O(n²) 复杂度**：这是本教程设计 15,000 小样本的根本原因。50,000 样本的 t-SNE 可能需要 5–10 分钟，而 15,000 样本只需 30 秒。

***

## 常见问题

### Q1: 为什么本教程用 50,000 样本，而案例教程 3/4 用 80,000？

**A**: 本教程的核心任务是降维和聚类，计算开销比"标准化比较"和"特征构造"更大。50,000 是一个平衡点：足够大让 PCA 和 KNN 的结论稳定，又不会让 t-SNE 等慢算法等太久。案例教程 3/4 不涉及 t-SNE，所以可以用更大的 80,000。

### Q2: 为什么不直接用独热编码处理分类变量？

**A**: 独热编码会让 7 个分类变量扩展成 30+ 列，特征维度从 22 变成 50+，教学复杂度增加。本教程用标签编码保持 22 维的简洁特征集，但要注意：**标签编码把分类变量当连续变量处理，PCA 会假设"2 比 1 大"这种顺序关系，这是不严谨的**。在实际科研中，分类变量应该用独热编码或专门的降维方法（如 MCA）。

### Q3: `Year_From_2000` 和 `year` 完全线性相关，为什么要构造？

**A**: 这是**故意制造的冗余**，目的是演示 PCA 的"去冗余"能力。PCA 会把它们合并到同一个主成分，方差解释率不变，但特征数减少。这正好展示了 PCA 相对于特征选择的核心优势——特征选择很难处理"完全相关"的特征（VIF 会无穷大），但 PCA 可以完美处理。

### Q4: 为什么 `Is_Child` 对缺失年龄的样本是 0 而不是 NaN？

**A**: `(df_feat['Age'] < 18)` 对 `NaN` 返回 `False`，所以 `Is_Child` 对缺失年龄的样本会是 0（不是儿童）。这在统计上不严谨——我们不知道缺失年龄的样本是否儿童。更严谨的做法是用 `.loc` 把缺失值的 `Is_Child` 设回 `NaN`，但本教程为了简化省略了这一步。

### Q5: `stratify=y` 为什么重要？

**A**: 本数据集 VIVO 占 71.68%，MORTO 占 28.32%。如果不分层抽样，可能训练集里 VIVO 占 75%、测试集里 VIVO 占 65%——分布不一致会让模型评估有偏差。分层抽样后，训练集和测试集的 VIVO 比例都是 71.68%，评估更公平。

### Q6: 为什么 t-SNE 用 15,000 样本而不是 50,000？

**A**: t-SNE 的复杂度是 O(n²)，50,000 样本可能需要 5–10 分钟，而 15,000 样本只需 30 秒。对于教学演示，30 秒是可接受的等待时间。而且 15,000 个点画在 2D 平面上已经足够密集，再多就重叠看不清了。

### Q7: `RANDOM_STATE=42` 能保证完全可复现吗？

**A**: 在**同一版本、同一环境**下可以。但不同版本的 sklearn 内部实现可能有细微变化（如算法优化、默认参数调整），跨版本对比时仍可能有微小差异。t-SNE 尤其敏感——本教程会演示不同随机种子下 t-SNE 的结果差异。

***

## 本模块小结

本模块完成了案例教程 6 的所有数据准备工作：

1. **加载了 50,000 条癌症患者数据**，VIVO（存活）占 71.68%，MORTO（死亡）占 28.32%。
2. **构造了 22 维高维特征集**：11 个基础医学特征 + 11 个派生特征（分箱、多项式、对数、交互、布尔）。
3. **故意制造了特征冗余**：`Age`/`Age_Sq`/`Age_Centered`/`Age_Log` 高度相关，`year`/`Year_From_2000`/`Year_Decade` 完全线性相关——为 PCA 的"去冗余"演示提供素材。
4. **完成了预处理流水线**：`LabelEncoder`（分类变量编码）→ `SimpleImputer(mean)`（缺失值插补）→ `StandardScaler`（标准化）。
5. **划分了训练集和测试集**：35,000 训练 + 15,000 测试，分层抽样保持类别比例。
6. **准备了流形学习小样本**：15,000 样本，专门用于 t-SNE/UMAP/聚类等计算密集型任务。

这些准备工作为后续模块奠定了基础：

- **模块 1**：用 `X_train`/`X_test` 做 KNN 维度灾难实验。
- **模块 2**：用 `X_full_arr` 做 PCA 方差分析。
- **模块 3**：用 `X_manifold`/`y_manifold` 做 PCA/t-SNE/UMAP 可视化。
- **模块 4**：用 `X_manifold`/`y_manifold` 做 K-Means 聚类。
- **模块 5**：用 `X_train`/`X_test` 做 PCA + LogReg 性能对比。

> 💡 **下一模块预告**：模块 1 将用 KNN 分类器演示"维度灾难"——随着特征维度从 2 增加到 22，KNN 的 AUC 如何变化？为什么高维空间会让"近邻"失去意义？我们将用实验数据回答这些问题。

