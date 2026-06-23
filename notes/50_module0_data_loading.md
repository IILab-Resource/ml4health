# 模块 0：数据加载与特征准备

> 本模块是案例教程 5「特征选择」的起点，承接案例教程 4「特征工程」。在用六层方法（相关性、VIF、Filter、LASSO、RF、Boruta）层层筛选特征之前，我们必须先把数据加载进内存、构造目标变量、采样、构建一个"故意包含冗余"的特征集（包含 `Age` 与 `Age_Sq`、`year` 与 `Year_From_2000` 等高度相关的特征对），并对分类变量做标签编码、对缺失值做均值插补、对数值做标准化。
>
> 本模块最核心的知识点有三个：**一是构建一个"教学用"的特征集**——故意把 `Age`、`Age_Sq`、`Age_Group`、`Is_Child` 同时放入，让学生看到特征选择如何识别冗余；**二是同时准备未标准化和标准化两份数据**——`X_imp` 给树模型用（树模型不需要标准化），`X_scaled` 给 LASSO 和逻辑回归用（线性模型需要标准化）；**三是分层抽样划分训练集和测试集**——所有六层特征选择都只在训练集上做，避免数据泄漏。

***

## 学习目标

学完本模块后，你将能够：

1. **理解每个 import 语句的作用**：特别是本教程新增的 `SelectKBest`、`f_classif`、`mutual_info_classif`、`LassoCV`、`RandomForestClassifier`、`permutation_importance`、`BorutaPy`，以及它们在六层特征选择流水线中各自的角色。
2. **理解** **`N_SAMPLES = 50000`** **的设计动机**：知道为什么本教程把样本量从案例教程 4 的 80,000 降到 50,000（因为 Boruta 计算量极大，需要多次训练随机森林）。
3. **掌握"故意制造冗余"的特征集设计**：明白为什么把 `Age`、`Age_Sq`、`Age_Group`、`Is_Child` 同时放入特征集——这是为了在后续模块中演示相关性分析、VIF、LASSO 如何识别并处理这些冗余特征。
4. **理解** **`base_feat`** **列表的构成**：知道 8 个基础特征的数据类型、量纲范围和临床含义。
5. **掌握 LabelEncoder 循环的写法**：理解 `encode` 函数如何同时处理"未知类别"和"NaN 缺失值"。
6. **理解** **`SimpleImputer(strategy='mean')`** **的作用**：知道为什么本教程只用最简单的均值插补（控制变量法）。
7. **重点理解"同时准备两份数据"的设计**：明白为什么需要 `X_imp`（未标准化）和 `X_scaled`（标准化）两份数据，分别给哪类模型用。
8. **掌握** **`train_test_split`** **中的** **`stratify=y`** **参数**：理解分层抽样为什么对不平衡数据集至关重要。

***

## 一、导入必要的库

本教程相比案例教程 4，**新增了大量特征选择相关的库**——这是本教程的核心。下面是完整的导入代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss
from sklearn.inspection import permutation_importance
from boruta import BorutaPy

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与前四个教程相同）

####

### 1.2 sklearn 模块详解（重点关注新增的特征选择库）

sklearn（scikit-learn）是 Python 最主流的机器学习库。本教程从 sklearn 的 5 个子模块导入了多个类和函数，下面逐一解释。

#### `from sklearn.model_selection import train_test_split` — 数据划分

**`model_selection`** 子模块提供了数据划分、交叉验证、超参数调优等工具。

- **`train_test_split`**：把数据集划分成训练集和测试集。在本教程中，我们会把 50,000 条样本划分成 70% 训练集（35,000 条）和 30% 测试集（15,000 条）。
- 关键参数 `stratify=y` 会做分层抽样，保持训练集和测试集的标签分布一致。
- **与案例教程 4 完全一致**，无变化。

#### `from sklearn.preprocessing import StandardScaler, LabelEncoder` — 标准化和标签编码

**`preprocessing`** 子模块提供了数据预处理工具。

- **`StandardScaler`**：标准化器，把数值特征缩放成**均值 0、标准差 1** 的分布。公式为 `z = (x - μ) / σ`。本教程用它准备 `X_scaled`，喂给 LASSO 和逻辑回归（线性模型需要标准化）。
- **`LabelEncoder`**：标签编码器，把分类变量的字符串类别映射成整数。本教程用它对 `Gender`、`Diagnostic.means`、`Extension`、`Raca.Color` 四个分类特征做编码。

> 💡 **与案例教程 4 的区别**：案例教程 4 还导入了 `RobustScaler`（用于对比不同标准化方法）。本教程的核心主题是"特征选择"，不再聚焦标准化对比，所以移除了 `RobustScaler`，只用 `StandardScaler`。

#### `from sklearn.impute import SimpleImputer` — 简单插补

**`impute`** 子模块提供了缺失值插补工具。

- **`SimpleImputer`**：简单插补器，用单一统计量（均值、中位数、众数、常数）填充缺失值。本教程只用 `strategy='mean'`（均值插补）。

> 💡 **控制变量法**：本教程的核心主题是"特征选择方法对比"，所以插补环节只用最简单的均值插补，不让插补方法的差异干扰特征选择的结论。这是一种"控制变量法"的实验设计。

#### `from sklearn.feature_selection import (SelectKBest, f_classif, mutual_info_classif)` — 特征选择（本教程核心新增）

**`feature_selection`** 子模块是本教程**最核心的新增**——它提供了各种特征选择工具。

- **`SelectKBest`**：单变量特征选择器，按某种评分函数选出 Top-K 个特征。本教程用 `k='all'` 表示不筛选，只计算所有特征的得分（用于排序）。
- **`f_classif`**：ANOVA F 检验的评分函数，衡量特征对不同目标组的区分能力。F 值越大，区分能力越强。这是**线性视角**的特征重要性。
- **`mutual_info_classif`**：互信息（Mutual Information）的评分函数，衡量特征与目标之间的"共享信息量"。这是**非线性视角**的特征重要性，能捕捉 F 检验漏掉的非线性关系。

> 💡 **重点概念：为什么要同时用 ANOVA 和 MI？**
>
> ANOVA F 检验只能捕捉**线性关系**——如果特征和目标的关系是非线性的（如 U 型关系），F 检验可能给出低分。而互信息能捕捉**任意关系**（包括非线性），但估计方差更大、计算更慢。
>
> 同时用两种方法，可以互补：ANOVA 给出稳定的线性排名，MI 补充非线性关系。本教程用"平均排名"融合两种方法的结果。

#### `from sklearn.linear_model import LogisticRegression, LassoCV` — 逻辑回归和 LASSO

**`linear_model`** 子模块提供了各种线性模型。

- **`LogisticRegression`**：逻辑回归，二分类任务最经典的模型。本教程用它作为"评估不同特征集效果"的统一模型——所有六层选出的特征集都喂给同一个逻辑回归，比较它们的 AUC、Recall、Brier Score。
- **`LassoCV`**：带交叉验证的 LASSO 回归。LASSO（Least Absolute Shrinkage and Selection Operator）在损失函数中添加 L1 正则化项 `α × Σ|wᵢ|`，使不重要特征的系数被压缩至 0。`LassoCV` 会自动用交叉验证选择最优的 `α`（正则化强度）。

> 💡 **重点概念：LASSO 为什么能做特征选择？**
>
> L1 正则化项 `α × Σ|wᵢ|` 会对所有非零系数施加"惩罚"。当 `α` 足够大时，不重要特征的系数会被**完全压缩至 0**，从而实现特征选择。这与 L2 正则化（Ridge）不同——L2 只会把系数缩小，但不会变成 0。
>
> `LassoCV` 用交叉验证自动选择 `α`：`α` 太小，正则化不足（过拟合）；`α` 太大，正则化过度（欠拟合）。CV 会找到"刚刚好"的 `α`。

#### `from sklearn.ensemble import RandomForestClassifier` — 随机森林

**`ensemble`** 子模块提供了集成学习模型。

- **`RandomForestClassifier`**：随机森林分类器，由多棵决策树集成。本教程用它做两件事：
  1. **第五层**：直接用 RF 的 `feature_importances_` 属性获取特征重要性，选 Top 特征。
  2. **第六层**：作为 Boruta 的"估计器"（estimator），Boruta 内部会多次训练 RF 来比较真实特征和阴影特征。

> 💡 **为什么 RF 能给出特征重要性？**
>
> RF 的每棵决策树在分裂节点时会选择"最能降低不纯度（Gini 杂质）"的特征。一个特征被用于分裂的次数越多、带来的不纯度下降越大，它的重要性就越高。RF 把所有树的重要性平均，得到最终的特征重要性。

#### `from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss` — 评估指标

**`metrics`** 子模块提供了模型评估指标。

- **`roc_auc_score`**：计算 ROC 曲线下面积（AUC），衡量模型区分正负类的能力。AUC=1 完美，AUC=0.5 等同随机。
- **`recall_score`**：计算召回率（Recall），即真正例占所有实际正例的比例。本教程关注 `Recall(VIVO)`——有多少存活患者被正确识别。
- **`brier_score_loss`**：Brier 分数，衡量预测概率的校准程度（预测概率和真实标签的均方误差）。Brier 越低，预测概率越可靠。

> 💡 **为什么用三个指标？** AUC 衡量区分能力，Recall 衡量正类识别能力，Brier 衡量概率校准。三者互补，能全面评估特征选择对模型性能的影响。

#### `from sklearn.inspection import permutation_importance` — 置换重要性（导入但未使用）

**`inspection`** 子模块提供了模型检验工具。

- **`permutation_importance`**：置换重要性，打乱某个特征的值后观察模型性能下降程度。下降越多，特征越重要。

<br />

#### `from boruta import BorutaPy` — Boruta 算法（本教程重点）

**`boruta`** 是一个独立的第三方库（不属于 sklearn），实现了 Boruta 算法。

- **`BorutaPy`**：Boruta 算法的 Python 实现。核心思想是"让真实特征与随机阴影特征竞争"——为每个特征创建一个"打乱版"的阴影特征，训练随机森林后比较真实特征和阴影特征的重要性。如果真实特征显著优于阴影特征，则"Confirmed"（确认）；如果不显著优于，则"Rejected"（拒绝）；如果无法确定，则"Tentative"（暂定）。

> 💡 **安装提示**：`boruta` 不在 sklearn 中，需要单独安装：
>
> ```bash
> pip install boruta
> ```
>
> <br />

### 1.3 本教程相比案例教程 4 的 import 变化总结

| 类别   | 案例教程 4                            | 案例教程 5（本教程）                                         | 变化原因              |
| ---- | --------------------------------- | --------------------------------------------------- | ----------------- |
| 标准化器 | `StandardScaler` + `RobustScaler` | 仅 `StandardScaler`                                  | 不再聚焦标准化对比         |
| 特征选择 | ❌ 无                               | **`SelectKBest`、`f_classif`、`mutual_info_classif`** | 本教程核心             |
| 线性模型 | `LogisticRegression`              | `LogisticRegression` + **`LassoCV`**                | 新增 LASSO 做特征选择    |
| 集成模型 | ❌ 无                               | **`RandomForestClassifier`**                        | 用于 RF 重要性和 Boruta |
| 评估指标 | AUC/Recall/Brier/PR 等             | AUC/Recall/Brier                                    | 精简，不再用 PR 曲线      |
| 第三方库 | ❌ 无                               | **`BorutaPy`**                                      | 本教程核心             |
| 检验工具 | ❌ 无                               | `permutation_importance`                            | 教学扩展用             |

> 💡 **小贴士**：本教程的 import 结构反映了"六层特征选择"的设计——每层方法对应一个库：相关性分析用 pandas 的 `corr()`，VIF 用 statsmodels，Filter 用 sklearn 的 `SelectKBest`，LASSO 用 sklearn 的 `LassoCV`，RF 用 sklearn 的 `RandomForestClassifier`，Boruta 用第三方库 `BorutaPy`。

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
N_SAMPLES = 50000   # Boruta 计算量大，适度采样
```

###

###

***

## 四、加载数据与创建目标变量

```python
# ============================================================================
# 0. 数据加载 & 特征工程
# ============================================================================
print("\n[0] 加载数据与特征准备...")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
df = df.dropna(subset=['target'])

np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()

print(f"    样本量: {len(df):,}  (VIVO: {(df['target'] == 1).sum():,} | {(df['target'] == 1).mean() * 100:.2f}%)")
```

###

### 4.4 采样逻辑

```python
np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()
```

- **`np.random.seed(RANDOM_STATE)`**：固定随机种子，确保每次运行采样的样本相同。
- **`if len(df) > N_SAMPLES:`**：只有当数据量超过 50,000 时才采样，避免数据量不足时报错。
- **`np.random.choice(len(df), N_SAMPLES, replace=False)`**：从 `[0, len(df))` 范围内**无放回**（`replace=False`）抽取 50,000 个索引。
- **`df.iloc[idx].copy()`**：按索引取子集，`.copy()` 创建副本，避免后续修改影响原 DataFrame（避免 SettingWithCopyWarning）。

### 4.5 实际运行结果

运行上述代码后，输出如下：

```
[0] 加载数据与特征准备...
    样本量: 50,000  (VIVO: 41,234 | 82.47%)
```

**解读**：

- 采样后共 50,000 条样本。
- 其中 VIVO（存活）41,234 条，占 82.47%。
- MORTO（死亡）8,766 条，占 17.53%。
- 这是一个**不平衡数据集**（正负类比例约 4.7:1），后续建模时需要用 `class_weight='balanced'` 处理。

> ⚠️ **常见问题**：为什么正类（VIVO）占 82%？这样模型不是只要全部预测"存活"就能达到 82% 准确率吗？
>
> 是的！这就是"准确率悖论"——在不平衡数据集上，准确率会误导。所以本教程用 AUC、Recall、Brier 三个指标，而不是准确率。AUC 不受类别分布影响，Recall 关注正类识别，Brier 衡量概率校准。

***

## 五、构建"故意包含冗余"的特征集

```python
# ---------- 构建丰富的特征集 ----------
base_feat = ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology',
             'Diagnostic.means', 'Extension', 'Raca.Color']
df_feat = df[base_feat + ['target']].copy()
```

### 5.1 `base_feat` 列表的设计

本教程从原始数据中精选 8 个基础特征：

| 特征名                  | 数据类型      | 量纲范围      | 临床含义     |
| -------------------- | --------- | --------- | -------- |
| `Age`                | 数值        | 0–120     | 患者诊断时的年龄 |
| `year`               | 数值        | 2000–2020 | 诊断年份     |
| `Gender`             | 分类（编码后整数） | 0–1       | 性别       |
| `Code.Profession`    | 数值（编码）    | 0–9999    | 职业代码     |
| `Code.of.Morphology` | 数值（编码）    | 0–9999    | 肿瘤形态学代码  |
| `Diagnostic.means`   | 分类（编码后整数） | 0–4       | 诊断手段     |
| `Extension`          | 分类（编码后整数） | 0–9       | 肿瘤扩散程度   |
| `Raca.Color`         | 分类（编码后整数） | 0–4       | 种族/肤色    |

> 💡 **重点概念：为什么选这 8 个特征？**
>
> 这 8 个特征涵盖了**人口学**（Age、Gender、Raca.Color）、**时间**（year）、**职业**（Code.Profession）、**肿瘤特征**（Code.of.Morphology、Extension、Diagnostic.means）四个维度，是临床预测模型的常见特征集。
>
> 注意：`Age` 和 `year` 是真正的连续数值；`Gender`、`Diagnostic.means`、`Extension`、`Raca.Color` 是分类变量（需要 LabelEncoder）；`Code.Profession` 和 `Code.of.Morphology` 虽然是数字编码，但本质上是分类变量（代码本身没有大小关系）。

### 5.2 `df_feat = df[base_feat + ['target']].copy()`

- **`base_feat + ['target']`**：把特征列表和目标列拼接成一个列表，用于从 `df` 中选取子集。
- **`.copy()`**：创建副本，避免后续修改影响原 DataFrame。

***

## 六、标签编码分类变量

```python
# 标签编码分类变量
cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color']
for col in cat_cols:
    le = LabelEncoder()
    non_null = df_feat[col].dropna().astype(str)
    le.fit(non_null)
    most_common = non_null.value_counts().index[0]
    def encode(x):
        if pd.isna(x): return np.nan
        xs = str(x)
        return le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]
    df_feat[col] = df_feat[col].apply(encode)
```

### 6.1 为什么需要标签编码？

sklearn 的模型只接受**数值型**输入，不能直接处理字符串。所以需要把分类变量（如 `Gender` 的 `'M'`/`'F'`）映射成整数（如 `0`/`1`）。

### 6.2 逐行解析

#### `cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color']`

定义需要标签编码的分类列列表。注意 `Code.Profession` 和 `Code.of.Morphology` 不在这里——它们已经是数字编码，不需要再编码。

#### `for col in cat_cols:`

遍历每个分类列，分别做标签编码。

#### `le = LabelEncoder()`

为每列创建一个新的 `LabelEncoder` 实例。注意：每列的编码器是独立的，因为不同列的类别不同。

#### `non_null = df_feat[col].dropna().astype(str)`

- **`dropna()`**：删除缺失值，因为 `LabelEncoder` 不能处理 NaN。
- **`astype(str)`**：转成字符串，确保所有值都是字符串类型（避免混合类型报错）。

#### `le.fit(non_null)`

用非缺失值拟合编码器，学习"类别 → 整数"的映射。

#### `most_common = non_null.value_counts().index[0]`

找出该列的**众数**（出现次数最多的类别），用于后续处理"未知类别"。

- **`value_counts()`**：按出现次数降序排列。
- **`.index[0]`**：取第一个（即众数）的类别名。

#### `def encode(x):` 函数

这是一个**闭包**，捕获了 `le`、`most_common` 等变量。它的逻辑：

```python
def encode(x):
    if pd.isna(x): return np.nan          # 缺失值保持 NaN
    xs = str(x)                            # 转成字符串
    return le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]
```

- **`if pd.isna(x): return np.nan`**：如果是缺失值，返回 NaN（保持缺失，后续用 `SimpleImputer` 处理）。
- **`xs = str(x)`**：转成字符串，确保与 `le.classes_` 中的类型一致。
- **`le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]`**：
  - 如果 `xs` 在已学习的类别中（`le.classes_`），返回对应的整数编码。
  - 如果 `xs` 不在（即"未知类别"，可能是测试集中出现了训练集没有的类别），返回众数的编码。

> 💡 **重点概念：为什么要处理"未知类别"？**
>
> 在实际项目中，测试集可能出现训练集没有见过的新类别（如新职业、新种族）。如果不处理，`le.transform` 会报错。本教程的策略是"用众数替代未知类别"——这是一种简单但鲁棒的处理方式。
>
> 另一种常见策略是用 `-1` 表示未知类别，但这可能让某些模型（如线性模型）误解为"有大小关系"。

#### `df_feat[col] = df_feat[col].apply(encode)`

对整列应用 `encode` 函数，把字符串类别替换成整数编码。

> ⚠️ **常见问题**：为什么不用 `pd.get_dummies` 做独热编码？
>
> 独热编码会把一个分类变量拆成多个 0/1 列，这会让特征数膨胀。本教程的核心是"特征选择"，如果用独热编码，`Gender` 会变成 `Gender_M`、`Gender_F` 两列，特征选择的解释会变复杂。所以本教程用标签编码，保持特征数不变。
>
> 但要注意：标签编码会引入"虚假的大小关系"（如 `Extension=9` 比 `Extension=1` "大"），这对线性模型（如逻辑回归）可能有问题。对树模型（如 RF）则没有影响，因为树模型只看分裂阈值，不关心大小关系。

***

## 七、构造新特征（故意制造冗余）

```python
# 构造新特征 (基于领域知识)
def age_group(a):
    if pd.isna(a): return np.nan
    a = float(a)
    return 0 if a < 18 else 1 if a < 40 else 2 if a < 60 else 3 if a < 75 else 4

df_feat['Age_Group'] = df_feat['Age'].apply(age_group)
df_feat['Age_Sq'] = df_feat['Age'] ** 2
df_feat['Year_From_2000'] = df_feat['year'] - 2000
df_feat['Is_Child'] = (df_feat['Age'] < 18).astype(float)

all_features = base_feat + ['Age_Group', 'Age_Sq', 'Year_From_2000', 'Is_Child']
n_feat = len(all_features)
print(f"    特征总数: {n_feat}")
print(f"    特征列表: {all_features}")
```

### 7.1 构造 4 个新特征

本教程构造 4 个新特征，**故意与基础特征高度相关**，用于演示特征选择如何识别冗余：

#### `Age_Group` — 年龄分组

```python
def age_group(a):
    if pd.isna(a): return np.nan
    a = float(a)
    return 0 if a < 18 else 1 if a < 40 else 2 if a < 60 else 3 if a < 75 else 4

df_feat['Age_Group'] = df_feat['Age'].apply(age_group)
```

把连续年龄分成 5 组：

- 0: <18 岁（儿童）
- 1: 18–39 岁（青年）
- 2: 40–59 岁（中年）
- 3: 60–74 岁（老年）
- 4: ≥75 岁（高龄）

**与** **`Age`** **的关系**：`Age_Group` 是 `Age` 的"离散化版本"，两者高度相关（后续会看到 r ≈ 0.95）。

#### `Age_Sq` — 年龄的平方

```python
df_feat['Age_Sq'] = df_feat['Age'] ** 2
```

把年龄平方，用于捕捉"年龄的非线性效应"。

**与** **`Age`** **的关系**：`Age_Sq` 是 `Age` 的二次变换，两者高度相关（后续会看到 r ≈ 0.98）。

#### `Year_From_2000` — 距 2000 年的年数

```python
df_feat['Year_From_2000'] = df_feat['year'] - 2000
```

把诊断年份减去 2000，得到"距 2000 年的年数"（如 2010 → 10）。

**与** **`year`** **的关系**：`Year_From_2000` 是 `year` 的线性变换（减常数），两者**完全相关**（r = 1.0000）！

#### `Is_Child` — 是否为儿童

```python
df_feat['Is_Child'] = (df_feat['Age'] < 18).astype(float)
```

二值特征：`Age < 18` 为 1（儿童），否则为 0。

**与** **`Age`** **的关系**：`Is_Child` 是 `Age` 的"阈值化版本"，相关性较弱（因为只用了部分信息）。

### 7.2 `all_features` 列表

```python
all_features = base_feat + ['Age_Group', 'Age_Sq', 'Year_From_2000', 'Is_Child']
n_feat = len(all_features)
```

把 8 个基础特征和 4 个新特征拼接，得到 12 个特征：

```python
['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology',
 'Diagnostic.means', 'Extension', 'Raca.Color',
 'Age_Group', 'Age_Sq', 'Year_From_2000', 'Is_Child']
```

### 7.3 实际运行结果

```
    特征总数: 12
    特征列表: ['Age', 'year', 'Gender', 'Code.Profession', 'Code.of.Morphology', 'Diagnostic.means', 'Extension', 'Raca.Color', 'Age_Group', 'Age_Sq', 'Year_From_2000', 'Is_Child']
```

> 💡 **重点概念：为什么故意制造冗余？**
>
> 本教程的核心是"特征选择"，所以需要**有冗余可删**。如果特征集已经很"干净"（无冗余），特征选择就没什么可做的了。
>
> 这 4 个新特征中：
>
> - `Year_From_2000` 与 `year` 完全相关（r=1.0），**必然被删除**。
> - `Age_Sq` 与 `Age` 高度相关（r=0.98），**很可能被删除**。
> - `Age_Group` 与 `Age` 高度相关（r=0.95），**很可能被删除**。
> - `Is_Child` 与 `Age` 相关性较弱，但信息量低（Boruta 会判定为 Rejected）。
>
> 这些冗余特征会在后续模块中被六层方法逐一识别和处理。

***

## 八、缺失值插补与标准化

```python
# 转为 float 并处理缺失
X_full = df_feat[all_features].astype(float)
y = df_feat['target'].values

imputer = SimpleImputer(strategy='mean')
X_imp = imputer.fit_transform(X_full)
X_imp_df = pd.DataFrame(X_imp, columns=all_features)

# 标准化 (用于 LASSO / 逻辑回归等)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features)
```

### 8.1 `X_full = df_feat[all_features].astype(float)`

- 从 `df_feat` 中选取 12 个特征列。
- **`astype(float)`**：把所有列转成浮点数。这是 sklearn 的要求——sklearn 的输入必须是数值型数组，不能有混合类型（如 int + str）。

### 8.2 `y = df_feat['target'].values`

- 提取目标变量。
- **`.values`**：把 pandas Series 转成 numpy 数组，sklearn 要求标签是数组形式。

### 8.3 `imputer = SimpleImputer(strategy='mean')`

创建均值插补器：

- **`strategy='mean'`**：用列的均值填充缺失值。其他选项：`'median'`（中位数）、`'most_frequent'`（众数）、`'constant'`（常数，需配合 `fill_value`）。

> 💡 **为什么用均值？**
>
> 均值插补是最简单的插补方法，能保持列的均值不变。本教程的核心是"特征选择"，不想让插补方法的差异干扰结论，所以用最简单的均值插补。
>
> 均值插补的缺点是会**低估方差**（因为所有缺失值都用同一个值填充，减少了变异性）。但在样本量大（50,000 条）的情况下，这种影响很小。

### 8.4 `X_imp = imputer.fit_transform(X_full)`

- **`fit_transform`**：拟合并转换。`fit` 计算每列的均值，`transform` 用均值填充缺失值。
- 返回值 `X_imp` 是 numpy 数组，没有列名。

### 8.5 `X_imp_df = pd.DataFrame(X_imp, columns=all_features)`

把 numpy 数组转回 DataFrame，并恢复列名。这个 DataFrame 会在后续 VIF 分析中用到（因为 `statsmodels` 的 `variance_inflation_factor` 需要 DataFrame 输入）。

### 8.6 标准化

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features)
```

- **`StandardScaler()`**：创建标准化器。
- **`fit_transform(X_imp)`**：拟合并转换。`fit` 计算每列的均值和标准差，`transform` 用 `z = (x - μ) / σ` 标准化。
- **`X_scaled_df`**：标准化后的 DataFrame，用于 LASSO 和逻辑回归。

> 💡 **重点概念：为什么需要两份数据（X\_imp 和 X\_scaled）？**
>
> 不同的模型对标准化的需求不同：
>
> | 模型     | 是否需要标准化 | 原因                                     |
> | ------ | ------- | -------------------------------------- |
> | 逻辑回归   | ✅ 需要    | 梯度下降优化，未标准化会导致特征间梯度差异巨大                |
> | LASSO  | ✅ 需要    | L1 正则化对所有系数施加相同惩罚，未标准化会让大量纲特征"不公平"地被惩罚 |
> | 随机森林   | ❌ 不需要   | 树模型只看分裂阈值，不关心特征尺度                      |
> | Boruta | ❌ 不需要   | 内部用 RF，不需要标准化                          |
>
> 所以本教程准备两份数据：
>
> - **`X_imp`**（未标准化）：给 RF 和 Boruta 用。
> - **`X_scaled`**（标准化）：给 LASSO 和逻辑回归用。
>
> 注意：ANOVA F 检验和互信息**理论上不需要标准化**（它们是尺度无关的），但本教程用 `X_train_s`（标准化）做 ANOVA，用 `X_train`（未标准化）做 MI，这是为了演示两种数据都可以用。

***

## 九、划分训练集和测试集

```python
# 划分
X_train, X_test, X_train_s, X_test_s, y_train, y_test = train_test_split(
    X_imp, X_scaled, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"    训练集: {len(X_train):,}  测试集: {len(X_test):,}")
```

### 9.1 `train_test_split` 参数详解

这行代码同时划分两份数据（`X_imp` 和 `X_scaled`），确保它们的索引一致：

- **`X_imp`**：未标准化数据，划分后得到 `X_train` 和 `X_test`。
- **`X_scaled`**：标准化数据，划分后得到 `X_train_s` 和 `X_test_s`。
- **`y`**：标签，划分后得到 `y_train` 和 `y_test`。
- **`test_size=0.3`**：测试集占 30%（15,000 条），训练集占 70%（35,000 条）。
- **`random_state=RANDOM_STATE`**：固定随机种子，确保划分可复现。
- **`stratify=y`**：分层抽样，保持训练集和测试集的标签分布一致。

### 9.2 `stratify=y` 的重要性

本数据集正负类比例约 4.7:1（VIVO 82% vs MORTO 18%）。如果不用分层抽样，可能测试集中 MORTO 的比例会偏离 18%（如变成 15% 或 21%），影响评估的可靠性。

分层抽样确保：

- 训练集中 VIVO 占 82.47%，MORTO 占 17.53%。
- 测试集中 VIVO 占 82.47%，MORTO 占 17.53%。

### 9.3 为什么同时划分两份数据？

`train_test_split` 可以同时划分多个数组，**保持它们的索引对应关系**。也就是说：

- `X_train[i]` 和 `X_train_s[i]` 是同一条样本（前者未标准化，后者标准化）。
- `y_train[i]` 是这条样本的标签。

这样设计的好处是：在后续验证阶段，我们可以用同一个测试集（`X_test_s`）评估不同特征集的效果，确保对比公平。

### 9.4 实际运行结果

```
    训练集: 35,000  测试集: 15,000
```

**解读**：

- 训练集 35,000 条（70%），用于特征选择和模型训练。
- 测试集 15,000 条（30%），用于最终验证。

> ⚠️ **重要：数据泄漏的防范**
>
> 本教程的所有特征选择（六层）都**只在训练集上做**，测试集完全不参与特征选择。这是为了防止**数据泄漏**（Data Leakage）——如果特征选择用了测试集的信息，会导致评估过于乐观。
>
> 具体来说：
>
> - 相关性分析：用 `X_imp`（全量），但实际应该用 `X_train`。本教程为了简化教学，用全量做相关性分析，但因为相关性是特征间的关系（不涉及标签），泄漏风险较低。
> - VIF：同上。
> - Filter（ANOVA/MI）：用 `X_train_s` 和 `y_train`，正确。
> - LASSO：用 `X_train_s` 和 `y_train`，正确。
> - RF：用 `X_train` 和 `y_train`，正确。
> - Boruta：用 `X_train` 和 `y_train`，正确。
>
> 在生产环境中，建议所有特征选择都严格在训练集上做，包括相关性分析和 VIF。

***

## 小贴士

1. **`N_SAMPLES = 50000`** **是教学效率与统计可靠性的平衡**：Boruta 计算量大，50,000 条样本能让 Boruta 在合理时间内完成，同时保证统计结论可靠。
2. **同时准备** **`X_imp`** **和** **`X_scaled`** **两份数据**：这是处理"不同模型对标准化需求不同"的标准做法。树模型用未标准化数据，线性模型用标准化数据。
3. **`stratify=y`** **对不平衡数据集至关重要**：本数据集正负类比例 4.7:1，分层抽样能保持训练集和测试集的标签分布一致。
4. **故意制造冗余特征是教学设计**：`Age_Sq`、`Age_Group`、`Year_From_2000` 这些特征与基础特征高度相关，用于演示特征选择如何识别冗余。
5. **`RANDOM_STATE = 42`** **贯穿全教程**：从采样到模型训练，所有随机操作都用同一个种子，确保结果完全可复现。
6. **标签编码 vs 独热编码**：本教程用标签编码保持特征数不变，便于特征选择的解释。但要注意标签编码会引入"虚假的大小关系"，对线性模型可能有问题。

***

## 常见问题

### Q1: 为什么 `boruta` 库安装失败？

**A**: `boruta` 不在 sklearn 中，需要单独安装。常见安装方式：

```bash
pip install boruta
# 或
pip install Boruta
```

如果安装失败，可能是网络问题，可以尝试：

```bash
pip install boruta -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q2: 为什么 `N_SAMPLES = 50000` 而不是 80,000？

**A**: 因为 Boruta 计算量极大（要训练 20 次随机森林，每次 100 棵树）。50,000 条样本能让 Boruta 在 30 秒内完成；80,000 条可能要 1–2 分钟。教学效率优先。

### Q3: 为什么 `Age_Sq` 和 `Age` 都保留？不是冗余吗？

**A**: 是的，它们高度冗余（r=0.98）。本教程**故意**保留它们，用于演示特征选择如何识别冗余。在后续模块中，你会看到相关性分析、LASSO 等方法如何处理这对冗余特征。

### Q4: `Year_From_2000 = year - 2000`，为什么相关性是 1.0000？

**A**: 因为 `Year_From_2000` 是 `year` 的**线性变换**（减常数）。线性变换不改变相关性，所以 r = 1.0000。这意味着这两个特征携带**完全相同的信息**，保留任何一个即可。

### Q5: 为什么用 `map` 而不是 `==` 创建目标变量？

**A**: `map` 会把不在字典中的值（如 NaN）映射成 `NaN`，便于后续 `dropna` 过滤。`==` 会返回 `False`（即 0），会把缺失值误判为"死亡"，导致标签错误。

### Q6: 标签编码后，`Gender=1` 比 `Gender=0` "大"吗？

**A**: 数值上是的，但**语义上没有大小关系**。`LabelEncoder` 的映射是按字母顺序（或首次出现顺序），`Gender=0` 可能是 `'F'`，`Gender=1` 可能是 `'M'`，1 不比 0 "大"。这对树模型没问题（只看分裂阈值），但对线性模型可能引入虚假的大小关系。

### Q7: 为什么 `Code.Profession` 和 `Code.of.Morphology` 不做标签编码？

**A**: 因为它们已经是数字编码（0–9999），不需要再编码。但要注意：它们本质上是分类变量（代码本身没有大小关系），用标签编码或独热编码更合适。本教程为了简化，直接当数值处理。

***

## 本模块小结

本模块完成了特征选择的**数据准备工作**，主要内容包括：

1. **导入库**：新增了 `SelectKBest`、`f_classif`、`mutual_info_classif`（Filter 方法）、`LassoCV`（LASSO）、`RandomForestClassifier`（RF 重要性）、`BorutaPy`（Boruta）等特征选择库。
2. **路径配置**：与案例教程 4 一致，输出 9 张图（`08a`–`08i`）和 2 个结果文件（`11_feature_selection_*`）。
3. **随机种子与采样**：`RANDOM_STATE = 42`，`N_SAMPLES = 50000`（比案例教程 4 少，因为 Boruta 计算量大）。
4. **数据加载与目标变量**：从 `Status.Vital` 列映射出 `target`（VIVO=1, MORTO=0），删除缺失标签，采样 50,000 条。最终正类占 82.47%，是不平衡数据集。
5. **构建特征集**：8 个基础特征 + 4 个新特征（`Age_Group`、`Age_Sq`、`Year_From_2000`、`Is_Child`），共 12 个特征。新特征**故意与基础特征冗余**，用于演示特征选择。
6. **标签编码**：对 4 个分类列（`Gender`、`Diagnostic.means`、`Extension`、`Raca.Color`）做 LabelEncoder，处理了未知类别和缺失值。
7. **缺失值插补**：用 `SimpleImputer(strategy='mean')` 做均值插补，得到 `X_imp`。
8. **标准化**：用 `StandardScaler` 标准化，得到 `X_scaled`。两份数据分别给树模型（不需要标准化）和线性模型（需要标准化）用。
9. **数据划分**：`train_test_split` 同时划分两份数据，`test_size=0.3`，`stratify=y` 分层抽样。训练集 35,000 条，测试集 15,000 条。

至此，数据准备完成。下一模块我们将进入**第一层特征选择：相关性分析**，识别并删除高度共线的特征对。
