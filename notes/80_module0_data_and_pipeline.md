# 模块 0：数据加载与基础 Pipeline 构建

> 本模块是案例教程 8「数据划分与交叉验证」的起点，承接案例教程 7（数据泄漏）。在比较 7 种交叉验证方法之前，我们必须先把数据加载进内存、构造目标变量、采样、精选一组基础特征、对分类变量做标签编码，并构建一个**可复用的 Pipeline 工厂函数**——这个 Pipeline 会在后续所有交叉验证实验中被反复调用。 
>
> 本模块最核心的知识点有三个：**一是** **`sklearn.model_selection`** **中各种交叉验证类的导入**——这是本教程相比案例 7 新增的核心内容，包括 `KFold`、`StratifiedKFold`、`RepeatedKFold`、`RepeatedStratifiedKFold`、`LeaveOneOut`、`cross_val_score`、`GridSearchCV` 等；**二是** **`Pipeline`** **工厂函数** **`create_pipeline(C=1.0)`** **的设计**——为什么要把预处理和建模封装成一个函数，而不是直接写死在主流程里；**三是** **`class_weight='balanced'`** **和** **`max_iter=5000`** **这两个逻辑回归参数的含义**——它们如何影响模型训练和收敛。

***

## 学习目标

学完本模块后，你将能够：

1. **理解本教程的整体框架**：知道我们要对比 7 种交叉验证方法，以及为什么这个对比对模型评估至关重要。
2. **掌握** **`sklearn.model_selection`** **中各种 CV 类的导入**：明白 `KFold`、`StratifiedKFold`、`RepeatedKFold`、`RepeatedStratifiedKFold`、`LeaveOneOut` 各自的用途，以及 `cross_val_score` 和 `GridSearchCV` 的角色。
3. **深入理解** **`create_pipeline(C=1.0)`** **工厂函数**：明白为什么用函数封装 Pipeline，`C` 参数如何作为超参数被传递，以及 `class_weight='balanced'`、`max_iter=5000` 的作用。
4. **理解 Pipeline 在交叉验证中的关键作用**：明白为什么必须用 Pipeline 包裹预处理和建模，否则会发生数据泄漏。

***

## 一、导入必要的库

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     RepeatedKFold, RepeatedStratifiedKFold,
                                     LeaveOneOut, cross_val_score,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss

warnings.filterwarnings('ignore')
```

### 11 基础库（与案例教程 7 相同）

### 1.2 sklearn 模块详解（重点关注 model\_selection 中的 CV 类）

这是本教程**最核心的导入**——相比案例教程 7，新增了大量交叉验证相关的类。

#### `from sklearn.model_selection import (...)` — 交叉验证核心（本教程重点）

**`model_selection`** 子模块是 sklearn 中专门处理"模型选择"的工具集，包括数据划分、交叉验证、超参数调优。本教程用括号跨行导入了 9 个工具：

**1.** **`train_test_split`** **— 单次划分**

最基础的数据划分函数，把数据集分成训练集和测试集。本教程第一部分用它做 20 次不同随机种子的划分，演示"单次划分的不稳定性"。

**2.** **`KFold`** **— K 折交叉验证**

最基础的 K 折交叉验证类。把数据集均分成 K 份，每次用 K-1 份训练、1 份验证，循环 K 次。本教程会用 `KFold(n_splits=5, shuffle=True, random_state=42)` 和 `KFold(n_splits=10, ...)`。

**3.** **`StratifiedKFold`** **— 分层 K 折交叉验证**

在 KFold 的基础上，**保持每折的类别比例与整体一致**。例如整体有 41% 的正类，那么每折也尽量保持 41% 的正类。这是医学数据的标准做法。

**4.** **`RepeatedKFold`** **— 重复 K 折交叉验证**

把 KFold 重复多次（每次用不同的随机洗牌），把所有结果取平均。例如 `RepeatedKFold(n_splits=5, n_repeats=10)` 会做 10 次 5 折 CV，总共 50 个模型。

**5.** **`RepeatedStratifiedKFold`** **— 重复分层 K 折交叉验证**

把 StratifiedKFold 重复多次。结合了"分层"和"重复"两个优势，是医学项目的推荐选择。

**6.** **`LeaveOneOut`** **— 留一法交叉验证**

K 折的极端情况：K = n（样本数）。每次留 1 个样本做验证，其余 n-1 个做训练，循环 n 次。偏差最小但计算成本最高。

**7.** **`cross_val_score`** **— 交叉验证评分函数**

最便捷的 CV 工具：传入模型、数据、CV 策略，自动完成"划分→训练→评估"的循环，返回每折的分数。本教程大量使用这个函数。

**8.** **`GridSearchCV`** **— 网格搜索交叉验证**

用于超参数调优。在参数网格上做穷举搜索，每个参数组合用 CV 评估，选出最佳参数。本教程在 Nested CV 部分用它做内层调参。

> 💡 **重点概念：CV 类 vs CV 函数**
>
> 注意区分两类工具：
>
> - **CV 类**（`KFold`、`StratifiedKFold`、`LeaveOneOut` 等）：定义"如何划分数据"。你需要先实例化它们（如 `cv = KFold(n_splits=5)`），然后把它们传给 `cross_val_score` 或在循环中调用 `cv.split(X, y)`。
> - **CV 函数**（`cross_val_score`、`GridSearchCV`）：定义"如何使用划分来评估模型"。它们接受一个 CV 类（或整数）作为 `cv` 参数。
>
> 这种"策略对象 + 执行函数"的设计是 sklearn 的核心模式，叫做**策略模式**。

###

<br />

***

## 二、数据加载与预处理

```python
# ============================================================================
# 0. 数据加载 & 预处理
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

***

## 三、特征准备与标签编码

```python
# 特征准备 (同案例 7)
feature_cols = ['Age', 'year', 'Gender', 'Code.Profession',
                'Diagnostic.means', 'Extension', 'Raca.Color']
df_feat = df[feature_cols + ['target']].copy()

cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color']
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

X = df_feat[feature_cols].astype(float).values
y = df_feat['target'].values
```

### 3.1 特征选择

```python
feature_cols = ['Age', 'year', 'Gender', 'Code.Profession',
                'Diagnostic.means', 'Extension', 'Raca.Color']
df_feat = df[feature_cols + ['target']].copy()
```

本教程选用 7 个特征，与案例 7 完全一致：

| 特征                 | 类型 | 临床含义              |
| ------------------ | -- | ----------------- |
| `Age`              | 数值 | 患者年龄              |
| `year`             | 数值 | 诊断年份              |
| `Gender`           | 分类 | 性别                |
| `Code.Profession`  | 分类 | 职业代码（取值范围 0–9999） |
| `Diagnostic.means` | 分类 | 诊断方式              |
| `Extension`        | 分类 | 肿瘤扩展程度            |
| `Raca.Color`       | 分类 | 种族/肤色             |

<br />

### 3.2 分类变量标签编码

```python
cat_cols = ['Gender', 'Diagnostic.means', 'Extension', 'Raca.Color']
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

<br />

***

## 四、构建基础 Pipeline（本模块核心）

```python
# ---------- 建立基础 Pipeline ----------
def create_pipeline(C=1.0):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=C, class_weight='balanced',
                                   max_iter=5000, random_state=RANDOM_STATE))
    ])

print(f"    特征数: {len(feature_cols)}")
```

这是本模块**最核心**的代码——一个**Pipeline 工厂函数**。后续所有交叉验证实验都会调用这个函数来创建模型。

### 4.1 为什么用函数封装 Pipeline？

你可能会问：为什么不直接写 `pipe = Pipeline([...])`，而要包成一个函数？

原因有三个：

1. **参数化超参数**：函数接受 `C` 参数，可以创建不同 `C` 值的 Pipeline。这在 Nested CV 的内层调参中很重要——`GridSearchCV` 会用不同的 `C` 值调用这个函数。
2. **避免状态污染**：sklearn 的 Pipeline 对象是有状态的（fit 过之后会记住训练数据）。如果直接复用同一个 Pipeline 对象，第二次 fit 会受第一次的影响。用函数每次返回一个**全新的** Pipeline，保证每次 fit 都是从头开始。
3. **代码复用**：本教程要跑 7 种 CV 方法，每种都要用同一个 Pipeline。用函数封装后，每次调用 `create_pipeline()` 就能得到一个标准化的模型，避免重复代码。

### 4.2 Pipeline 的三个步骤

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(C=C, class_weight='balanced',
                               max_iter=5000, random_state=RANDOM_STATE))
])
```

`Pipeline` 接受一个列表，列表中每个元素是一个 `(名称, 步骤)` 元组。本 Pipeline 有三个步骤：

**步骤 1：缺失值插补**

```python
('imputer', SimpleImputer(strategy='median'))
```

- 名称：`'imputer'`
- 步骤：`SimpleImputer(strategy='median')`，用中位数填充缺失值。
- 为什么用中位数而不是均值？因为中位数对异常值更鲁棒（参考案例 4 的标准化讨论）。

**步骤 2：标准化**

```python
('scaler', StandardScaler())
```

- 名称：`'scaler'`
- 步骤：`StandardScaler()`，把特征缩放成均值 0、标准差 1。
- 为什么需要标准化？因为逻辑回归使用梯度下降优化，未标准化的特征（如 `Code.Profession` 范围 0–9999 和 `Age` 范围 0–120）会导致梯度更新步长差异巨大，影响收敛。

**步骤 3：逻辑回归**

```python
('lr', LogisticRegression(C=C, class_weight='balanced',
                           max_iter=5000, random_state=RANDOM_STATE))
```

- 名称：`'lr'`
- 步骤：`LogisticRegression(...)`，逻辑回归模型。

### 4.3 LogisticRegression 的参数详解

这是本模块需要重点理解的参数：

#### `C=C`

**正则化强度的倒数**。`C` 是逻辑回归最重要的超参数：

- `C` 越大，正则化越弱，模型越复杂（可能过拟合）。
- `C` 越小，正则化越强，模型越简单（可能欠拟合）。
- 默认 `C=1.0`。
- 在 Nested CV 部分，我们会搜索 `C ∈ {0.01, 0.1, 1, 10}` 这 4 个值。

> 💡 **为什么** **`C`** **是"倒数"？**
>
> sklearn 的 `C` 对应正则化项系数的倒数：`C = 1/λ`。这是为了和 LIBSVM/LIBLINEAR 的约定保持一致。所以 `C` 大 → `λ` 小 → 正则化弱。

#### `class_weight='balanced'`

**类别权重**。这个参数告诉模型"自动根据类别频率调整权重"：

- 不设（默认 `None`）：所有类别权重相同。
- `'balanced'`：自动调整权重，使少数类获得更大权重。权重公式为 `n_samples / (n_classes * np.bincount(y))`。

本数据集 VIVO 占 41.15%、MORTO 占 58.85%，虽然不算严重不平衡，但用 `balanced` 可以让模型更关注少数类（VIVO）。

> 💡 **`class_weight='balanced'`** **的权重计算**：
>
> 以本数据集为例（n=20000，VIVO=8230，MORTO=11770）：
>
> - VIVO 权重 = 20000 / (2 × 8230) ≈ 1.215
> - MORTO 权重 = 20000 / (2 × 11770) ≈ 0.850
>
> 也就是说，模型会把每个 VIVO 样本"当作" 1.215 个样本，每个 MORTO 样本"当作" 0.850 个样本。这样少数类（VIVO）在损失函数中的占比就提高了。

#### `max_iter=5000`

**最大迭代次数**。逻辑回归用梯度下降优化，`max_iter` 限制最大迭代轮数：

- 默认 `max_iter=100`。
- 本教程设为 `5000`，因为数据量较大（20000 样本）且特征未标准化前量纲差异大，可能需要更多迭代才能收敛。
- 如果迭代未收敛，sklearn 会警告 `ConvergenceWarning`。设为 5000 可以避免这个警告。

#### `random_state=RANDOM_STATE`

**随机种子**。逻辑回归在某些求解器（如 `sag`、`saga`、`liblinear`）中涉及随机性（如数据洗牌）。设置 `random_state=42` 保证结果可复现。

### 4.4 为什么必须用 Pipeline？（数据泄漏的关键）

> 💡 **重点概念：Pipeline 是防止数据泄漏的关键**
>
> 在交叉验证中，每一折的训练集和验证集是分开的。**预处理（插补、标准化）必须在每折的训练集上独立 fit，然后 transform 验证集**。如果先在全数据上 fit 预处理，再传给 CV，就会发生**数据泄漏**——验证集的信息（如均值、中位数）泄漏到了训练过程。
>
> `Pipeline` 会自动处理这个问题：当你把 Pipeline 传给 `cross_val_score` 时，Pipeline 会在每折的训练集上 `fit_transform`，在验证集上只 `transform`，确保没有泄漏。
>
> ```python
> # ✅ 正确：Pipeline 在每折中独立 fit
> scores = cross_val_score(create_pipeline(), X, y, cv=5)
>
> # ❌ 错误：预处理泄漏
> X_scaled = StandardScaler().fit_transform(X)  # 用了全数据!
> scores = cross_val_score(LogisticRegression(), X_scaled, y, cv=5)
> ```
>
> 这正是案例教程 7（数据泄漏）的核心教训。本教程用 Pipeline 来确保所有 CV 实验都不会泄漏。

### 4.5 打印特征数

```python
print(f"    特征数: {len(feature_cols)}")
```

输出：        `    特征数: 7`

简单打印特征数量，确认数据准备完成。

***

## 小贴士

1. **Pipeline 命名规范**：Pipeline 中每个步骤的名称（如 `'imputer'`、`'scaler'`、`'lr'`）是任意的，但要简短有意义。在 `GridSearchCV` 中调参时，参数名格式是 `步骤名__参数名`（如 `'lr__C'`），所以步骤名要方便拼写。
2. **`C`** **参数的搜索范围**：本教程搜索 `C ∈ {0.01, 0.1, 1, 10}`，这是对数均匀分布的 4 个值。在实践中，超参数搜索通常用对数尺度（因为参数的影响是指数级的），而不是线性尺度。
3. **`class_weight='balanced'`** **不是万能的**：它只是简单地把权重按频率反比调整。对于严重不平衡的数据（如 1:100），可能需要更精细的方法，如 SMOTE 过采样（参考案例教程的后续内容）。
4. **`max_iter=5000`** **的取舍**：设得太小可能不收敛，设得太大可能浪费时间。5000 是一个比较安全的值，对于 20000 样本的数据集足够。如果你看到 `ConvergenceWarning`，可以适当增大。
5. **随机种子的位置**：`np.random.seed(RANDOM_STATE)` 在采样前设置，保证采样可复现。但 sklearn 内部的随机性（如 `LogisticRegression` 的 `random_state`）需要单独传参，不受 `np.random.seed` 影响。

***

## 常见问题

### Q: Pipeline 的步骤顺序能调换吗？

**A**: 不能随意调换。本教程的顺序是 `imputer → scaler → lr`，这个顺序有道理：

- 插补必须在标准化之前：因为 `StandardScaler` 不能处理 NaN，会报错。
- 标准化必须在建模之前：因为逻辑回归对量纲敏感。
- 如果顺序错了（如 `scaler → imputer`），`StandardScaler` 会在 NaN 上报错。

### &#x20;

***

