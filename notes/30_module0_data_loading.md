# 模块 0：数据加载与基础预处理

> 本模块是案例教程 4「特征工程」的起点，承接案例教程 1（EDA）、案例教程 2（统计学分析）和案例教程 3（数据预处理与缺失值插补）。在比较不同标准化方法和构造新特征之前，我们必须先把数据加载进内存、构造目标变量、采样、精选一组有代表性的基础特征，并对分类变量做标签编码。本模块对应代码文件 `src/04_feature_engineering.py` 的第 20–91 行。
>
> 本模块最核心的知识点有三个：**一是 `RobustScaler` 的引入**——这是本教程新增的标准化器，用中位数和四分位距（IQR）替代均值和标准差；**二是 `base_features` 列表的设计**——为什么从 5 个特征扩展到 6 个，新增的 `Code.Profession` 如何制造"量纲差异"这一教学重点；**三是量纲差异如何成为后续标准化比较的动机**——Age 跨度约 120、year 跨度约 15、Code.Profession 跨度约 9999，这种巨大差异会让逻辑回归"偏心"大量纲特征。

---

## 学习目标

学完本模块后，你将能够：

1. **理解每个 import 语句的作用**：特别是本教程新增的 `RobustScaler`，以及相比案例教程 3 精简掉了哪些库（`KNNImputer`、`IterativeImputer`、`calibration_curve` 等）及其原因。
2. **掌握 `RobustScaler` 与 `StandardScaler` 的本质区别**：明白为什么"用中位数和 IQR"比"用均值和标准差"更抗异常值，以及为什么本教程要把两者放在一起对比。
3. **理解随机种子和采样规模的设计**：知道 `RANDOM_STATE = 42` 和 `N_SAMPLES = 80000` 如何与案例教程 3 保持一致，便于跨案例比较。
4. **回顾 `map` vs `==` 的本质区别**：明白为什么本教程继续用 `map` 创建目标变量，缺失值如何被正确保留并过滤。
5. **深入理解 `base_features` 列表的设计哲学**：明白为什么从 5 个特征扩展到 6 个，新增 `Code.Profession` 的教学意图，以及每个特征的数据类型、量纲范围和临床含义。
6. **重点理解"量纲差异是标准化的动机"**：能够说出 Age、year、Code.Profession 三个特征在量纲跨度上的巨大差异，以及这种差异如何影响逻辑回归的优化过程。
7. **掌握 LabelEncoder 循环的写法**：理解 `encode` 函数如何同时处理"未知类别"和"NaN 缺失值"，以及为什么要把编码器存进 `label_encoders` 字典。
8. **理解 `astype(float)` 的必要性**：明白为什么在喂给 sklearn 之前要把整个 DataFrame 转成浮点数。

---

## 一、导入必要的库

本教程相比案例教程 3，**新增了 `RobustScaler`**，同时**精简掉了** `KNNImputer`、`IterativeImputer`、`enable_iterative_imputer`、`calibration_curve`、`gridspec` 等。这是因为本教程的核心主题是"标准化比较"和"特征构造"，不再聚焦于"插补方法对比"，所以插补环节只用最简单的均值插补即可。下面是完整的导入代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, roc_curve,
                             brier_score_loss, precision_recall_curve,
                             average_precision_score)

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与案例教程 3 相同）

#### `import pandas as pd`

**pandas** 是 Python 中最核心的数据分析库，提供 `DataFrame` 二维表格结构。在本教程中用于：`read_csv()` 读数据、`map()` 标签映射、`dropna()` 删除缺失行、`isnull()` 检测缺失值、`apply()` 应用编码函数等。

#### `import numpy as np`

**NumPy** 是 Python 数值计算的基石，提供高效的多维数组 `ndarray`。在本教程中用于：`np.random.seed()` 设置随机种子、`np.random.choice()` 采样、`np.nan` 表示缺失值等。

#### `import matplotlib.pyplot as plt`

**matplotlib** 是 Python 最基础的可视化库。本教程后续会绘制标准化分布对比图、系数对比图、ROC 曲线、PR 曲线、特征构造性能对比图等。

> 💡 **与案例教程 3 的区别**：案例教程 3 还导入了 `from matplotlib import gridspec`，用于精细控制子图布局（因为要画 4 种插补方法的对比图）。本教程的对比维度更简单（3 种标准化、2 种特征集），用普通的 `plt.subplots` 就够了，所以不再需要 `gridspec`。

#### `import os` 和 `import warnings`

- **os**：标准库，用于路径操作（`os.path.join()`、`os.makedirs()`）。
- **warnings**：标准库，`warnings.filterwarnings('ignore')` 关闭所有警告，让输出更干净。

#### `import time`

**time** 是 Python 标准库，用于记录时间。本教程会用 `time.time()` 记录不同标准化方法下逻辑回归的**收敛速度**——这是评估标准化效果的重要维度（标准化后模型通常收敛更快）。

### 1.2 sklearn 模块详解（重点关注新增的 RobustScaler）

sklearn（scikit-learn）是 Python 最主流的机器学习库。本教程从 sklearn 的 4 个子模块导入了多个类和函数，下面逐一解释，并标注与案例教程 3 的区别。

#### `from sklearn.model_selection import train_test_split` — 数据划分

**`model_selection`** 子模块提供了数据划分、交叉验证、超参数调优等工具。

- **`train_test_split`** 是其中最常用的函数，用于把数据集划分成训练集和测试集。
- 在本教程中，我们会把 80,000 条样本划分成 70% 训练集（56,000 条）和 30% 测试集（24,000 条）。
- 关键参数 `stratify=y` 会做分层抽样，保持训练集和测试集的标签分布一致。
- **与案例教程 3 完全一致**，无变化。

#### `from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder` — 标准化和标签编码（本教程重点）

**`preprocessing`** 子模块提供了数据预处理工具。这一行是本教程**最关键的变化**——新增了 `RobustScaler`。

- **`StandardScaler`**：标准化器，把数值特征缩放成**均值 0、标准差 1** 的分布。公式为 `z = (x - μ) / σ`，其中 μ 是均值、σ 是标准差。这是案例教程 3 已经用过的标准化器。
- **`RobustScaler`**（本教程新增，重点）：**鲁棒标准化器**，用**中位数**和**四分位距（IQR）**替代均值和标准差。公式为 `z = (x - median) / IQR`，其中 IQR = Q3 − Q1（第 75 百分位减第 25 百分位）。
- **`LabelEncoder`**：标签编码器，把分类变量的字符串类别映射成整数。本教程用它对 `Gender`、`Diagnostic.means`、`Raca.Color` 三个分类特征做编码。

> 💡 **重点概念：为什么本教程要新增 `RobustScaler`？**
>
> `StandardScaler` 用均值和标准差，这两个统计量都**对异常值非常敏感**——一个极端值就能把均值拉偏、把标准差放大。而 `RobustScaler` 用中位数和 IQR，这两个都是**鲁棒统计量**（robust statistics），不受少数极端值影响。
>
> 本教程的数据中，`Code.Profession` 的取值范围高达 0–9999，可能存在极端值；`Age` 也可能有录入错误（如 200 岁）。用 `RobustScaler` 可以让标准化结果不受这些异常值的干扰。
>
> 本教程的核心实验之一就是**对比 `StandardScaler` 和 `RobustScaler` 在同一数据上的表现**，观察哪种标准化更适合本数据集。

**`StandardScaler` vs `RobustScaler` 对比表**：

| 特性 | StandardScaler | RobustScaler |
|------|----------------|--------------|
| 中心化统计量 | 均值 μ | 中位数 median |
| 尺度化统计量 | 标准差 σ | 四分位距 IQR = Q3 − Q1 |
| 公式 | `z = (x - μ) / σ` | `z = (x - median) / IQR` |
| 对异常值的鲁棒性 | ❌ 敏感（一个极端值就拉偏） | ✅ 鲁棒（中位数和 IQR 不受极端值影响） |
| 输出分布 | 均值 0、标准差 1 | 中位数 0、IQR 1 |
| 适用场景 | 数据近似正态、无显著异常值 | 数据有异常值或重尾分布 |
| 计算开销 | 低 | 略高（需计算分位数） |

#### `from sklearn.impute import SimpleImputer` — 简单插补（本教程精简）

**`impute`** 子模块提供了缺失值插补工具。

- **`SimpleImputer`**：简单插补器，用单一统计量（均值、中位数、众数、常数）填充缺失值。本教程只用 `strategy='mean'`（均值插补）。

> 💡 **与案例教程 3 的区别（重要）**：
>
> 案例教程 3 的核心主题是"缺失值插补方法对比"，所以导入了三种插补器：
> ```python
> from sklearn.impute import SimpleImputer, KNNImputer
> from sklearn.experimental import enable_iterative_imputer
> from sklearn.impute import IterativeImputer
> ```
>
> 而本教程的核心主题是"标准化比较"和"特征构造"，**插补不再是重点**。为了让实验变量更纯粹（只隔离出"标准化方法"和"特征构造"这两个变量），本教程**只用最简单的均值插补**处理缺失值，不再对比 KNN 和 MICE。
>
> 这是一种"控制变量法"的实验设计：当我们想比较 A（标准化）的影响时，就要把 B（插补）固定下来，不让 B 的差异干扰 A 的结论。

#### `from sklearn.linear_model import LogisticRegression` — 逻辑回归

**`linear_model`** 子模块提供了各种线性模型。

- **`LogisticRegression`**：逻辑回归，二分类任务最经典的模型。本教程用它作为"评估标准化方法和特征构造效果"的统一模型——所有实验都喂给同一个逻辑回归，比较它们的 AUC、Recall、Brier Score、收敛速度和系数，这样能隔离出"标准化方法"或"特征构造"这两个变量对性能的影响。
- **为什么选逻辑回归？** 因为逻辑回归对特征的量纲**非常敏感**——它使用梯度下降优化，未标准化的特征会导致不同特征的梯度更新步长差异巨大，影响收敛速度和系数可解释性。这正是本教程"标准化比较"实验的最佳载体。
- **与案例教程 3 完全一致**，无变化。

#### `from sklearn.metrics import (...)` — 评估指标函数

**`metrics`** 子模块提供了大量模型评估指标。本教程用括号跨行导入了一组函数：

- **`roc_auc_score`**：计算 ROC 曲线下面积（AUC），衡量模型区分正负类的能力。AUC=1 完美，AUC=0.5 等同随机。
- **`recall_score`**：计算召回率（Recall），即真正例占所有实际正例的比例。本教程关注 `Recall(VIVO)`——有多少存活患者被正确识别。
- **`roc_curve`**：计算 ROC 曲线的三个序列（FPR、TPR、阈值），用于绘制 ROC 曲线图。
- **`brier_score_loss`**：Brier 分数，衡量预测概率的校准程度（预测概率和真实标签的均方误差）。Brier 越低，预测概率越可靠。
- **`precision_recall_curve`**：计算 PR 曲线，用于不平衡数据集的评估。
- **`average_precision_score`**：平均精度（AP），即 PR 曲线下面积，对不平衡数据比 AUC 更敏感。

> 💡 **与案例教程 3 的区别**：案例教程 3 还导入了 `from sklearn.calibration import calibration_curve`，用于绘制校准曲线（评估插补方法对预测概率可靠性的影响）。本教程不再聚焦于概率校准，所以移除了这个导入。其余评估指标完全一致。

### 1.3 本教程相比案例教程 3 的 import 变化总结

| 类别 | 案例教程 3 | 案例教程 4（本教程） | 变化原因 |
|------|-----------|---------------------|---------|
| 标准化器 | `StandardScaler` | `StandardScaler` + **`RobustScaler`** | 新增鲁棒标准化器，用于对比实验 |
| 插补器 | `SimpleImputer` + `KNNImputer` + `IterativeImputer` | 仅 `SimpleImputer` | 插补不再是重点，只用均值插补固定变量 |
| 实验性 API | `enable_iterative_imputer` | ❌ 移除 | 不再用 IterativeImputer |
| 校准工具 | `calibration_curve` | ❌ 移除 | 不再聚焦概率校准 |
| 子图布局 | `gridspec` | ❌ 移除 | 对比维度更简单，用普通 subplots 即可 |
| 评估指标 | AUC/Recall/Brier/PR 等 | 完全一致 | 无变化 |
| 模型 | `LogisticRegression` | 完全一致 | 无变化 |

> 💡 **小贴士**：sklearn 的子模块划分遵循机器学习流程的顺序：`model_selection`（划分）→ `preprocessing`（预处理）→ `impute`（插补）→ `linear_model`（建模）→ `metrics`（评估）。本教程移除了 `calibration`（校准）这一环，因为不再聚焦概率校准。记住这个流程，就能快速定位到需要的子模块。

---

## 二、路径配置与目录创建

```python
BASE_DIR = "/home/wjj/Documents/trae_projects/ml_template"
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

这部分与前三个教程完全一致，简要说明：

- `BASE_DIR`：项目根目录，所有路径基于它拼接。
- `DATA_PATH`：数据文件路径，指向 `data/cancer_data_eng.csv`（英文版癌症数据）。
- `IMG_DIR` / `RESULTS_DIR`：图片和结果输出目录。本教程会输出 7 张图（`07a`–`07g`）和 2 个结果文件（`09_scaling_comparison`、`10_feature_engineering_comparison`）。
- `os.path.join()`：跨平台拼接路径（Windows 用 `\`，Linux/Mac 用 `/`）。
- `os.makedirs(..., exist_ok=True)`：递归创建目录，`exist_ok=True` 表示目录已存在时不报错。

> 💡 **小贴士**：脚本开头提前创建输出目录是好习惯，避免脚本运行到一半保存文件时才发现目录不存在。

---

## 三、随机种子与采样规模

```python
RANDOM_STATE = 42
N_SAMPLES = 80000
```

### 3.1 `RANDOM_STATE = 42` — 可复现性的基石

**`RANDOM_STATE`** 是一个整数种子，用于控制所有涉及随机性的操作，确保**结果可复现**。

在本教程中，`RANDOM_STATE` 会被用在以下地方：

| 使用位置 | 作用 |
|---------|------|
| `np.random.seed(RANDOM_STATE)` | 固定 numpy 的随机数生成器 |
| `np.random.choice(...)` | 固定采样时选中的样本索引 |
| `train_test_split(..., random_state=RANDOM_STATE)` | 固定训练集/测试集的划分 |
| `LogisticRegression(..., random_state=RANDOM_STATE)` | 固定模型内部随机性 |

**为什么需要固定随机种子？**

机器学习流程中有大量随机操作（采样、划分、模型初始化等）。如果不固定种子，每次运行脚本结果都会不同——这会让调试变得困难，也让"对比不同标准化方法"失去意义（你无法判断性能差异是来自标准化方法还是随机波动）。

固定种子后，**任何人、任何时间、任何机器上运行这段代码，都会得到完全相同的结果**，这就是"可复现性"。

**为什么是 42？**

42 是编程界的"梗"——来自《银河系漫游指南》（The Hitchhiker's Guide to the Galaxy），书中"生命、宇宙和一切的终极答案"就是 42。它已经成为机器学习社区最常用的随机种子（仅次于 0 和 1）。

### 3.2 `N_SAMPLES = 80000` — 与案例教程 3 保持一致（重点）

本教程从 209,758 条有标签样本中**采样 80,000 条**进行分析，而不是用全量数据。

**为什么与案例教程 3 保持完全一致的 80,000？**

> 💡 **重点概念：跨案例比较的可复现性**
>
> 本教程是案例教程 3 的延续，两者使用**完全相同的数据子集**（同样的 `RANDOM_STATE=42`、同样的 `N_SAMPLES=80000`、同样的采样逻辑）。这样设计的好处是：
>
> 1. **跨案例比较公平**：案例教程 3 的最佳 AUC（用 MICE 插补 + StandardScaler）和本教程的 AUC（用均值插补 + 不同标准化）可以直接对比，因为数据完全一样。
> 2. **隔离变量更纯粹**：如果两个教程用不同的数据子集，性能差异可能来自数据采样而非方法差异，结论就不可靠。
> 3. **复用前序结论**：案例教程 3 已经证明"均值插补 + StandardScaler"的基线性能，本教程可以在此基础上观察"换用 RobustScaler"或"构造新特征"带来的提升。

**为什么不直接用全量 20 万条？**

1. **计算开销**：本教程要对比 3 种标准化方法 + 2 种特征集（基础 vs 构造），每种组合都要训练逻辑回归。8 万条样本足够快，20 万条会让每次实验慢 2.5 倍。
2. **教学效率**：教程需要能快速跑通，让学生聚焦于"方法对比"而非"等待计算"。
3. **统计可靠性**：8 万条样本已经足够大，AUC、Brier Score 等指标的置信区间都很窄，结论不会因样本量不足而失真。

> ⚠️ **常见问题**：我设置了 `RANDOM_STATE = 42`，为什么不同版本的 sklearn 结果还是略有不同？
>
> 因为不同版本的 sklearn 内部实现可能有细微变化（如算法优化、默认参数调整）。`RANDOM_STATE` 只保证**同一版本、同一环境**下可复现。跨版本对比时，仍可能有微小差异。

---

## 四、加载数据与创建目标变量

```python
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')

df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
df = df.dropna(subset=['target'])
```

### 4.1 读取 CSV

`pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')` 与前三个教程完全一致：

- `low_memory=False`：一次性读取整个文件，避免类型推断警告。
- `encoding='latin-1'`：处理可能残留的葡萄牙语特殊字符。

### 4.2 创建目标变量——回顾 `map` vs `==`（回顾案例教程 2/3）

本教程继续用 `map` 而非 `==` 创建目标变量：

```python
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

这是从案例教程 2 继承的关键改进。两种方法的本质区别在于**对缺失值的处理**：

| Status.Vital 原值 | `(== 'VIVO').astype(int)` | `.map({'VIVO':1, 'MORTO':0})` |
|-------------------|--------------------------|-------------------------------|
| VIVO | 1 | 1 |
| MORTO | 0 | 0 |
| NaN（缺失） | **0** ⚠️ 被误判为 MORTO | **NaN** ✅ 保持缺失 |

**为什么 `==` 会把 NaN 当作 0？**

因为 IEEE 754 浮点数标准规定：**任何与 NaN 的比较都返回 False**，包括 `NaN == 'VIVO'`。所以 `(NaN == 'VIVO')` 为 `False`，`.astype(int)` 后变成 `0`，缺失值被错误地标记为"死亡"。

**为什么 `map` 更安全？**

`map()` 对字典中**没有定义的值**（包括 NaN）返回 NaN，保持缺失状态。这样后续 `dropna(subset=['target'])` 才能正确识别并删除这些样本。

> 💡 **核心知识点（回顾案例教程 2/3）**：
>
> 当原始标签列可能存在缺失值时，**必须用 `map` 而不是 `==`** 创建目标变量。`map` 让缺失值保持 NaN，`==` 会把缺失值误判为 0。这是从案例教程 1 升级到案例教程 2/3/4 的关键改进。

### 4.3 过滤缺失样本

```python
df = df.dropna(subset=['target'])
```

`dropna(subset=['target'])` 删除 `target` 列为 NaN 的行。由于前面用 `map` 创建 target，`Status.Vital` 缺失的样本在 target 列也是 NaN，会被这一步正确删除。

处理后：原始 1,778,176 条 → 保留 209,758 条有标签样本。

---

## 五、采样

```python
np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df = df.iloc[idx].copy()
```

### 5.1 `np.random.seed(RANDOM_STATE)` — 固定随机数生成器

在调用任何 numpy 随机函数之前，先设置随机种子。这样 `np.random.choice` 每次运行都会选中**相同的样本索引**，保证可复现。

### 5.2 `np.random.choice(len(df), N_SAMPLES, replace=False)` — 无放回采样

**`np.random.choice`** 是 numpy 的随机采样函数，参数含义：

- 第 1 个参数 `len(df)`：从 `0` 到 `len(df)-1` 的整数序列中采样（即所有样本的行索引）。
- 第 2 个参数 `N_SAMPLES`：采样数量，这里是 80,000。
- `replace=False`：**无放回采样**——每个样本最多被选中一次。

**`replace=False` vs `replace=True` 的区别**：

| 参数 | 含义 | 结果 |
|------|------|------|
| `replace=False` | 无放回 | 80,000 个**不重复**的样本（子集） |
| `replace=True` | 有放回 | 80,000 个样本，但可能有重复（bootstrap 采样） |

本教程用 `replace=False`，因为我们要得到一个**真实数据的子集**，不希望有重复样本。

> 💡 **小贴士**：`replace=False` 类似于"从一副牌中抽 5 张"——每张牌只能抽一次；`replace=True` 类似于"掷骰子 5 次"——每次都是独立的，可能掷出相同的点数。机器学习中，训练集划分用无放回，Bootstrap 集成方法（如随机森林的 bootstrap sampling）用有放回。

### 5.3 `df.iloc[idx].copy()` — 按位置索引选行并创建副本

- `df.iloc[idx]`：用整数位置索引选取被选中的 80,000 行。
- `.copy()`：创建**独立副本**，后续对 `df` 的修改不会影响原始数据。这避免了 pandas 的 `SettingWithCopyWarning` 警告。

> ⚠️ **常见问题**：什么时候需要加 `.copy()`？
>
> 经验法则：**当你打算后续修改一个从原 DataFrame 派生出来的子集时，就应该加 `.copy()`**。本教程后续会对 `df` 做大量修改（提取特征、编码），所以必须加 `.copy()`。

### 5.4 `if len(df) > N_SAMPLES` 的容错设计

这个 `if` 是一种**容错设计**：

- 如果原始数据量大于 `N_SAMPLES`（80,000），执行采样。
- 如果原始数据量小于等于 `N_SAMPLES`（比如换了一个小数据集），跳过采样，直接用全量数据。

这种写法让代码更健壮——即使数据集规模变化，代码也能正常运行。

---

## 六、基础特征选择（本模块核心）

这是本模块**最重要**的部分。我们先看代码：

```python
base_features = ['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Raca.Color']
df_feat = df[base_features + ['target']].copy()
```

### 6.1 为什么选这 6 个特征？（与案例教程 3 的区别）

案例教程 3 选了 5 个特征（`Age`、`year`、`Gender`、`Diagnostic.means`、`Raca.Color`），本教程**新增了 `Code.Profession`**，扩展到 6 个。这是基于以下三个原则的**精心设计**：

**原则 1：兼顾数据类型的多样性**

需要同时包含数值型和分类型特征，且分类特征要有不同的类别数：

| 特征 | 类型 | 类别数/范围 |
|------|------|------------|
| `Age` | 连续数值 | 0–120 |
| `year` | 离散数值 | 2000–2015 |
| `Gender` | 二分类 | 2 类 |
| `Code.Profession` | 分类（编码为数值） | 0–9999 |
| `Diagnostic.means` | 多分类 | 7 类 |
| `Raca.Color` | 多分类 | 5 类 |

**原则 2：兼顾缺失率的多样性**

本教程虽然不再对比插补方法，但仍需处理缺失值。所选特征的缺失率有梯度：

| 特征 | 缺失率 | 说明 |
|------|--------|------|
| `year` | 0.00% | 无缺失 |
| `Gender` | ~0% | 几乎无缺失 |
| `Age` | 0.15% | 极少缺失 |
| `Diagnostic.means` | 0.36% | 少量缺失 |
| `Code.Profession` | 较低 | 职业编码 |
| `Raca.Color` | 15.31% | 高缺失率 |

**原则 3：兼顾临床意义**

每个特征都要有明确的临床含义，不能是无关变量：

| 特征 | 临床含义 |
|------|---------|
| `Age` | 患者年龄，癌症生存的最强预测因子之一 |
| `year` | 诊断年份，反映医疗技术进步的影响 |
| `Gender` | 性别，某些癌症有性别差异 |
| `Code.Profession` | 职业编码，反映社会经济地位和职业暴露风险 |
| `Diagnostic.means` | 诊断方式（如病理确诊、临床诊断等），反映确诊的确定性 |
| `Raca.Color` | 人种/肤色，反映遗传背景和社会经济因素 |

### 6.2 为什么新增 `Code.Profession`？（本教程的关键设计）

这是本教程相比案例教程 3 最重要的特征选择变化。新增 `Code.Profession` 的**教学意图**是制造一个"量纲极端"的特征，让标准化比较的动机更鲜明：

- `Code.Profession` 是一个**分类变量**，但已经被编码成**数值**，取值范围高达 **0–9999**。
- 这个跨度远大于其他所有特征（Age 约 120、year 约 15）。
- 如果不做标准化，逻辑回归会"偏心"这个大量纲特征——它的系数会被压得很小，梯度更新主导整个优化过程。

> 💡 **重点概念：为什么 `Code.Profession` 是本教程标准化的"试金石"？**
>
> 案例教程 3 的 5 个特征中，最大量纲是 `Age`（约 120），最小是 `year`（约 15），差距约 8 倍。这种差距虽然存在，但还不够"极端"，标准化的效果差异可能不明显。
>
> 本教程新增 `Code.Profession`（跨度约 9999）后，最大与最小量纲的差距扩大到 **约 666 倍**（9999 / 15）。这种极端的量纲差异会让"未标准化 vs StandardScaler vs RobustScaler"的对比效果非常显著——逻辑回归在未标准化数据上可能根本无法收敛，或者收敛极慢。
>
> 这正是本教程"标准化比较"实验想要展示的核心现象。

### 6.3 每个特征的数据类型和量纲详解（重点）

#### `Age`（年龄）— 连续数值

- **类型**：连续数值（数值型）
- **量纲范围**：约 0–120（岁）
- **跨度**：约 120
- **缺失率**：0.15%（8 万样本中约 120 条缺失）
- **临床含义**：患者确诊时的年龄。年龄是癌症生存预测最重要的因子之一——老年患者通常预后更差。

#### `year`（诊断年份）— 离散数值

- **类型**：离散数值（数值型）
- **量纲范围**：约 2000–2015
- **跨度**：约 15
- **缺失率**：0%（无缺失）
- **临床含义**：患者确诊的年份。随着医疗技术进步，近年确诊的患者生存率通常更高。
- **注意**：虽然 year 看起来是"连续数值"，但它实际上是离散的整数年份，且跨度很小（约 15）。

#### `Gender`（性别）— 二分类

- **类型**：二分类（分类型）
- **量纲范围**：编码后为 0–1（2 个类别）
- **跨度**：约 1
- **缺失率**：接近 0%
- **临床含义**：患者性别。某些癌症（如乳腺癌）有显著性别差异。
- **取值**：通常为 `M`（男性）和 `F`（女性）两类，LabelEncoder 后变成 0 和 1。

#### `Code.Profession`（职业编码）— 分类（编码为数值，本教程新增重点）

- **类型**：分类变量（已被编码为数值）
- **量纲范围**：约 0–9999
- **跨度**：约 9999 ⚠️ **本教程中量纲最大的特征**
- **临床含义**：患者的职业编码，反映社会经济地位和职业暴露风险（某些职业接触致癌物）。
- **特殊性**：这个特征虽然取值是数值，但**数值大小没有顺序意义**——职业编码 5000 并不比编码 1000"大 5 倍"。它本质上是一个分类变量，只是被编码成了大数值。
- **教学作用**：它的极端量纲（跨度约 9999）是本教程标准化比较的"试金石"。

#### `Diagnostic.means`（诊断方式）— 多分类（7 类）

- **类型**：多分类（分类型，7 个类别）
- **量纲范围**：编码后为 0–6（7 个类别）
- **跨度**：约 6
- **缺失率**：0.36%（8 万样本中约 288 条缺失）
- **临床含义**：癌症的确诊方式，如病理确诊（最可靠）、临床诊断、影像学诊断等。诊断方式反映了确诊的确定性，间接影响生存预测。

#### `Raca.Color`（人种/肤色）— 多分类（5 类）

- **类型**：多分类（分类型，5 个类别）
- **量纲范围**：编码后为 0–4（5 个类别）
- **跨度**：约 4
- **缺失率**：15.31%（8 万样本中约 12,248 条缺失）——**本教程中缺失率最高的特征**
- **临床含义**：患者的人种/肤色分类（巴西人口普查标准，如白人、黑人、棕色人种、亚裔、原住民）。人种反映遗传背景和社会经济地位，两者都会影响癌症生存。

### 6.4 量纲差异——标准化的动机（本模块最重要的知识点）

> 💡💡💡 **重点中的重点：量纲差异是标准化的动机**
>
> 让我们把 6 个特征的量纲跨度放在一起对比：
>
> | 特征 | 量纲跨度 | 相对差距（除以最小跨度） |
> |------|---------|------------------------|
> | `Gender` | ~1 | 1× |
> | `Raca.Color` | ~4 | 4× |
> | `Diagnostic.means` | ~6 | 6× |
> | `year` | ~15 | 15× |
> | `Age` | ~120 | 120× |
> | `Code.Profession` | ~9999 | **9999×** |
>
> **最大与最小量纲的差距高达约 9999 倍！**
>
> 这种巨大的量纲差异会带来三个严重问题：
>
> 1. **梯度下降失衡**：逻辑回归用梯度下降优化，不同特征的梯度更新步长与特征量纲成正比。`Code.Profession` 的梯度会比 `Gender` 大近 1 万倍，导致优化过程被大量纲特征主导，小量纲特征"学不动"。
> 2. **收敛极慢甚至不收敛**：为了不让大量纲特征的梯度爆炸，学习率必须设得很小，但这又会让小量纲特征收敛极慢。在未标准化的数据上，逻辑回归可能需要几千次迭代才能收敛，甚至根本不收敛。
> 3. **系数不可解释**：逻辑回归的系数反映特征对预测的影响程度。但未标准化时，`Code.Profession` 的系数会被压得很小（因为特征值很大），`Gender` 的系数会很大（因为特征值小），无法直接比较特征的相对重要性。
>
> **这就是为什么本教程要做"标准化比较"实验**——我们要观察 `Raw`（不标准化）、`StandardScaler`、`RobustScaler` 三种方案下，逻辑回归的 AUC、Recall、Brier Score、收敛速度、系数大小如何变化。`Code.Profession` 的极端量纲让这个实验的效果非常显著。

### 6.5 `df_feat = df[base_features + ['target']].copy()` — 提取子集

这行代码从原始 `df` 中提取 6 个基础特征 + 目标变量，组成新的 DataFrame `df_feat`：

- `base_features + ['target']`：列表拼接，得到 `['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Raca.Color', 'target']`。
- `df[...]`：用列表索引选取多列。
- `.copy()`：创建独立副本，后续修改 `df_feat` 不影响 `df`。

> 💡 **小贴士**：`base_features + ['target']` 这种列表拼接写法很常见。注意 `base_features` 本身不含 `target`，所以要用 `+` 把 `target` 加进去。这样设计的好处是 `base_features` 列表保持"纯特征"，后续可以单独用它做特征工程，不会被 `target` 污染。

---

## 七、标签编码分类变量

```python
cat_cols = ['Gender', 'Diagnostic.means', 'Raca.Color']
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    non_null = df_feat[col].dropna().astype(str)
    le.fit(non_null)
    most_common = non_null.value_counts().index[0]
    def encode(x):
        if pd.isna(x):
            return np.nan
        xs = str(x)
        return le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]
    df_feat[col] = df_feat[col].apply(encode)
    label_encoders[col] = le
```

### 7.1 为什么只编码 3 个分类特征？

注意 `cat_cols` 只包含 `['Gender', 'Diagnostic.means', 'Raca.Color']`，**不包含 `Code.Profession`**。这是因为：

- `Gender`、`Diagnostic.means`、`Raca.Color` 在原始数据中是**字符串**（如 `'M'`、`'F'`），需要 LabelEncoder 转成整数。
- `Code.Profession` 在原始数据中**已经是数值**（0–9999），不需要再编码。
- `Age` 和 `year` 本身就是数值，也不需要编码。

### 7.2 逐行解释编码循环

#### `le = LabelEncoder()` — 创建编码器实例

每个分类特征都要一个独立的 `LabelEncoder` 实例，因为不同特征的类别映射不同。

#### `non_null = df_feat[col].dropna().astype(str)` — 准备训练数据

- `df_feat[col].dropna()`：删除缺失值，因为 LabelEncoder 不能在含 NaN 的数据上 `fit`。
- `.astype(str)`：把所有值转成字符串，确保类型一致（避免混合类型报错）。

#### `le.fit(non_null)` — 训练编码器

让 LabelEncoder 学习该列的所有唯一类别，建立"类别字符串 → 整数"的映射。例如 `Gender` 列有 `['F', 'M']`，编码后可能是 `{'F': 0, 'M': 1}`（按字母排序）。

#### `most_common = non_null.value_counts().index[0]` — 找众数

- `non_null.value_counts()`：统计每个类别的频次，按频次降序排列。
- `.index[0]`：取频次最高的类别（众数）。

**为什么要找众数？** 用于后续处理"未知类别"——如果测试集出现训练集没见过的类别，用众数代替（详见下方 `encode` 函数）。

#### `def encode(x)` — 定义编码函数（重点）

这是本段代码的核心，处理三种情况：

```python
def encode(x):
    if pd.isna(x):                                          # 情况 1: 缺失值
        return np.nan
    xs = str(x)
    return le.transform([xs])[0] if xs in le.classes_ else le.transform([most_common])[0]
    #                                                       情况 2: 已知类别  情况 3: 未知类别
```

- **情况 1：缺失值（`pd.isna(x)` 为 True）** → 返回 `np.nan`，保持缺失状态。这样后续的 `SimpleImputer` 才能正确识别并插补。
- **情况 2：已知类别（`xs in le.classes_`）** → 用 `le.transform([xs])[0]` 转成整数。
- **情况 3：未知类别（不在 `le.classes_` 中）** → 用众数 `most_common` 的编码代替。

**为什么需要处理"未知类别"？**

虽然本教程在全部数据上 `fit` 编码器（理论上不会出现未知类别），但这种写法是一种**防御性编程**的好习惯。在实际项目中，我们通常只在训练集上 `fit`，然后在测试集上 `transform`——如果测试集出现了训练集没见过的类别，就需要用众数（或其他策略）兜底，否则 `le.transform` 会报错。

#### `df_feat[col] = df_feat[col].apply(encode)` — 应用编码

`apply(encode)` 对该列的每个元素应用 `encode` 函数，返回编码后的 Series，覆盖原列。

#### `label_encoders[col] = le` — 保存编码器

把每个特征的 LabelEncoder 存进 `label_encoders` 字典，键是特征名，值是编码器实例。

**为什么要保存编码器？**

1. **可逆性**：后续如果需要把整数编码还原成原始类别（如绘图时显示标签），可以用 `le.inverse_transform()`。
2. **复用性**：如果后续有新数据需要用同样的编码规则，可以直接调用保存的编码器，保证编码一致。

> 💡 **与案例教程 3 的对比**：案例教程 3 的编码逻辑在"模块 1：数据划分与编码"中，且是在训练集上 `fit`、在测试集上 `transform`。本教程简化了流程，在**全部数据**上 `fit` 并 `transform`，因为本教程的重点不是"数据泄露"问题，而是"标准化"和"特征构造"。这种简化让代码更紧凑，聚焦于核心主题。

### 7.3 闭包陷阱（进阶知识点）

注意 `encode` 函数是在 `for` 循环内部定义的，它引用了循环变量 `le` 和 `most_common`。在 Python 中，这种"函数引用循环变量"的写法称为**闭包**（closure）。

由于每次循环都重新定义了 `encode` 函数，且立即用 `apply` 调用它（在进入下一次循环前），所以这里**不会**出现常见的"闭包陷阱"（所有函数都引用最后一个循环变量的值）。但如果把 `encode` 函数存起来延迟调用，就需要注意这个问题。

> ⚠️ **常见问题**：为什么 `encode` 函数能正确识别"当前循环的 `le`"？
>
> 因为 `apply(encode)` 是在当前循环内**立即执行**的，此时 `le` 和 `most_common` 还是当前特征的值。等下一次循环时，虽然 `le` 和 `most_common` 被重新赋值，但当前特征的编码已经完成了。这种"立即调用"的写法是安全的。

---

## 八、转浮点数

```python
df_feat = df_feat.astype(float)
```

### 8.1 为什么要转成 float？

经过 LabelEncoder 编码后，`df_feat` 中的列类型可能不一致：

- `Age`、`year`、`Code.Profession`：原本是 `int64` 或 `float64`。
- `Gender`、`Diagnostic.means`、`Raca.Color`：编码后是 `int64`（但含 NaN，所以实际可能是 `float64`）。
- `target`：`int64`（0 或 1）。

`astype(float)` 把所有列统一转成 `float64`，原因有三：

1. **sklearn 要求数值输入**：sklearn 的 `SimpleImputer`、`StandardScaler`、`RobustScaler`、`LogisticRegression` 都要求输入是数值类型，不能是字符串或 object。
2. **支持 NaN**：`float64` 类型可以表示 NaN，而 `int64` 不能（pandas 的整数类型遇到 NaN 会自动升级为 `float64`）。统一转 float 可以避免类型混乱。
3. **计算精度**：标准化（除以标准差或 IQR）会产生小数，float 类型能保留精度。

### 8.2 转换后的数据预览

转换后，`df_feat` 是一个 80,000 行 × 7 列（6 特征 + 1 目标）的纯浮点数 DataFrame，大致如下：

| Age | year | Gender | Code.Profession | Diagnostic.means | Raca.Color | target |
|-----|------|--------|-----------------|------------------|------------|--------|
| 62.0 | 2010.0 | 1.0 | 5142.0 | 3.0 | 2.0 | 0.0 |
| 55.0 | 2008.0 | 0.0 | 8731.0 | 1.0 | NaN | 1.0 |
| 70.0 | 2013.0 | 1.0 | 1234.0 | 0.0 | 4.0 | 0.0 |
| ... | ... | ... | ... | ... | ... | ... |

注意 `Raca.Color` 列有 NaN（缺失率 15.31%），这些 NaN 会在后续的 `SimpleImputer` 中被均值插补。

> 💡 **小贴士**：`astype(float)` 是喂给 sklearn 前的"最后一道工序"。如果你的数据中还有任何非数值列（如日期、字符串），sklearn 会直接报错。养成"在交给 sklearn 前先 `astype(float)`"的习惯，可以避免很多莫名其妙的错误。

---

## 九、本模块运行结果

运行本模块代码后，控制台输出大致如下：

```
======================================================================
案例教程 4: 特征工程
======================================================================

[0] 加载数据...
    样本量: 80,000  (VIVO: 32,760 | 40.95%)
    基础特征: ['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Raca.Color']
```

### 关键数字解读

| 指标 | 数值 | 说明 |
|------|------|------|
| 有标签样本 | 209,758 | Status.Vital 非缺失的样本（与案例教程 2/3 一致） |
| 采样样本量 | 80,000 | 从 209,758 条中无放回采样（与案例教程 3 一致） |
| VIVO（存活） | 32,760 | 占 40.95%，与全量数据的 40.95% 一致 |
| MORTO（死亡） | 47,240 | 占 59.05%，与全量数据的 59.05% 一致 |
| 基础特征数 | 6 | Age、year、Gender、Code.Profession、Diagnostic.means、Raca.Color |

### 为什么采样后标签比例与全量数据一致？

采样后 VIVO 仍占 40.95%、MORTO 占 59.05%，与全量数据完全一致。这是因为：

1. **样本量足够大**：8 万条样本的随机采样，大数定律保证比例会接近真实分布。
2. **无放回采样**：每个样本被选中的概率相等，不改变分布。
3. **随机种子固定**：这个比例是可复现的，与案例教程 3 完全相同。

### 为什么 VIVO 是 32,760 与案例教程 3 完全一样？

因为本教程与案例教程 3 使用**完全相同的采样逻辑**：

- 同样的 `RANDOM_STATE = 42`
- 同样的 `N_SAMPLES = 80000`
- 同样的 `np.random.seed(RANDOM_STATE)` + `np.random.choice(len(df), N_SAMPLES, replace=False)`

所以选中的样本索引完全相同，VIVO 数量也完全相同（32,760）。这种一致性是**跨案例比较的基础**——案例教程 3 的最佳 AUC 和本教程的 AUC 可以直接对比。

> ⚠️ **常见问题**：为什么采样后 VIVO 是 32,760 而不是其他数字？
>
> 因为固定了 `RANDOM_STATE=42` 后，`np.random.choice` 会生成确定的随机索引序列，每次运行都选中相同的 80,000 条样本，其中恰好有 32,760 条是 VIVO。这个数字是可复现的，与案例教程 3 完全一致。

---

## 十、小贴士与常见问题

### 💡 小贴士汇总

1. **`RobustScaler` 是本教程的新增重点**：用中位数和 IQR 替代均值和标准差，对异常值更鲁棒。记住公式 `z = (x - median) / IQR`，其中 `IQR = Q3 - Q1`。
2. **本教程精简了插补方法**：只用 `SimpleImputer(strategy='mean')`，不再对比 KNN 和 MICE。这是为了"控制变量"——聚焦于标准化和特征构造，不让插补差异干扰结论。
3. **随机种子要全局固定**：不仅 `np.random.seed`，`train_test_split`、`LogisticRegression` 等也要传 `random_state`，才能完全可复现。
4. **跨案例保持数据一致**：本教程与案例教程 3 用完全相同的 80,000 条样本，便于跨案例比较 AUC 等指标。
5. **`Code.Profession` 是标准化的"试金石"**：它的量纲跨度约 9999，是本教程中量纲最大的特征，让标准化比较的效果非常显著。
6. **`base_features + ['target']` 的列表拼接**：保持 `base_features` 列表"纯特征"，后续特征工程不会被 `target` 污染。
7. **`encode` 函数处理三种情况**：缺失值返回 NaN、已知类别正常编码、未知类别用众数兜底。这是防御性编程的好习惯。
8. **`astype(float)` 是喂给 sklearn 前的最后一道工序**：统一类型、支持 NaN、保证计算精度。

### ⚠️ 常见问题

**Q1：`RobustScaler` 和 `StandardScaler` 到底该用哪个？**

A：没有绝对答案，取决于数据。如果数据近似正态分布且无显著异常值，`StandardScaler` 更合适（结果更标准）。如果数据有重尾分布或异常值，`RobustScaler` 更鲁棒。本教程的实验就是要在同一数据上对比两者，观察哪个效果更好。经验法则：先看数据的箱线图，如果有明显的离群点，优先考虑 `RobustScaler`。

**Q2：为什么本教程不再用 KNN 插补和 MICE 插补？**

A：因为本教程的核心主题是"标准化比较"和"特征构造"，不再是"插补方法对比"。为了隔离出这两个变量，插补方法必须固定（用最简单的均值插补）。如果同时变化插补方法和标准化方法，就无法判断性能差异来自哪个因素。这是"控制变量法"的实验设计。

**Q3：`Code.Profession` 的数值大小有顺序意义吗？**

A：**没有**。职业编码 5000 并不比编码 1000"大 5 倍"，它只是一个分类标签。但因为它被编码成大数值，逻辑回归会把它当作连续数值处理，数值大小会影响模型。这正是标准化的动机之一——标准化后，数值的"绝对大小"被消除，模型更关注"相对模式"。更严谨的做法是用 OneHotEncoder，但本教程为了制造量纲差异的教学效果，保留了数值编码。

**Q4：`RANDOM_STATE = 42` 和 `np.random.seed(42)` 有什么区别？**

A：`np.random.seed(42)` 设置的是 numpy 全局随机数生成器的种子，影响所有 numpy 随机函数（如 `np.random.choice`）。`RANDOM_STATE = 42` 只是一个变量，需要传给 sklearn 函数的 `random_state` 参数才起作用。两者配合使用，才能保证全流程可复现。

**Q5：为什么 `encode` 函数里要单独处理 `pd.isna(x)`？**

A：因为 `le.transform([xs])` 不能处理 NaN——如果直接把 NaN 转成字符串 `'nan'` 再 transform，要么报错（如果 `'nan'` 不在 `classes_` 中），要么被错误地映射成某个整数。所以必须先用 `pd.isna(x)` 拦截 NaN，返回 `np.nan` 保持缺失状态，让后续的 `SimpleImputer` 处理。

**Q6：为什么 `cat_cols` 不包含 `Code.Profession`？**

A：因为 `Code.Profession` 在原始数据中**已经是数值**（0–9999），不需要 LabelEncoder 转换。`cat_cols` 只包含原始数据中是字符串的分类特征（`Gender`、`Diagnostic.means`、`Raca.Color`）。

**Q7：为什么采样后标签比例和全量数据几乎一样？**

A：因为无放回随机采样中，每个样本被选中的概率相等，不改变分布。加上样本量大（8 万），大数定律保证采样比例接近真实比例。

**Q8：`astype(float)` 会不会丢失信息？**

A：不会。`int64` 转 `float64` 是安全的（float64 能精确表示所有 int64 在 ±2^53 范围内的整数）。本教程的数据值都在这个范围内，所以转换无损。

---

## 十一、本模块小结

### 核心知识点回顾

1. **本教程的 import 变化**：
   - **新增**：`RobustScaler`（用中位数和 IQR，抗异常值）
   - **精简**：移除 `KNNImputer`、`IterativeImputer`、`enable_iterative_imputer`、`calibration_curve`、`gridspec`
   - **原因**：聚焦于标准化比较和特征构造，用最简单的均值插补固定变量

2. **`RobustScaler` vs `StandardScaler`**：
   - `StandardScaler`：`z = (x - μ) / σ`，对异常值敏感
   - `RobustScaler`：`z = (x - median) / IQR`，对异常值鲁棒
   - 本教程的核心实验之一就是对比两者

3. **随机种子与采样**：`RANDOM_STATE = 42` + `N_SAMPLES = 80000` 与案例教程 3 完全一致，保证跨案例比较的公平性。

4. **目标变量创建**：继续用 `map` 而非 `==`，让缺失值保持 NaN，`dropna` 才能正确过滤。

5. **`base_features` 列表的设计哲学**：
   - 从案例教程 3 的 5 个特征扩展到 6 个，新增 `Code.Profession`
   - 新增 `Code.Profession` 的教学意图：制造量纲极端的特征（跨度约 9999），让标准化比较的动机更鲜明
   - 6 个特征兼顾数据类型（数值+分类）、缺失率梯度（0%–15.31%）、临床意义

6. **量纲差异是标准化的动机**（本模块最重要的知识点）：
   - 6 个特征的量纲跨度从 ~1（Gender）到 ~9999（Code.Profession），差距约 9999 倍
   - 这种巨大差异会导致梯度下降失衡、收敛极慢、系数不可解释
   - 正是本教程"标准化比较"实验要解决的核心问题

7. **LabelEncoder 循环**：
   - `encode` 函数处理三种情况：缺失值返回 NaN、已知类别正常编码、未知类别用众数兜底
   - 编码器存进 `label_encoders` 字典，便于后续 `inverse_transform` 还原

8. **`astype(float)`**：喂给 sklearn 前的最后一道工序，统一类型、支持 NaN、保证计算精度。

### 数据概况

- 原始 209,758 条有标签样本 → 采样 80,000 条（与案例教程 3 完全相同）
- VIVO 32,760 (40.95%)，MORTO 47,240 (59.05%)
- 6 个基础特征：
  - `Age`（连续数值，0–120，0.15% 缺失）
  - `year`（离散数值，2000–2015，0% 缺失）
  - `Gender`（二分类，编码后 0–1，~0% 缺失）
  - `Code.Profession`（分类编码为数值，0–9999，本教程新增）
  - `Diagnostic.means`（多分类 7 类，编码后 0–6，0.36% 缺失）
  - `Raca.Color`（多分类 5 类，编码后 0–4，15.31% 缺失）

### 下一部分预告

本模块完成了数据加载、采样、特征选择和标签编码，得到了一个 80,000 条样本、6 个基础特征的纯浮点数 DataFrame `df_feat`。接下来的教程分为两个核心部分：

**第一部分：数据标准化比较**
- 对比三种方案：`Raw`（不标准化）、`StandardScaler`、`RobustScaler`
- 用同一个逻辑回归模型，观察 AUC、Recall、Brier Score、收敛速度、系数大小的变化
- 重点观察 `Code.Profession` 的极端量纲如何影响未标准化模型的收敛
- 讨论"为什么逻辑回归对量纲敏感"以及"什么时候该用 RobustScaler"

**第二部分：特征构造**
- 基于医学领域知识构造 7 个新特征：`Age_Group`、`Age_Sq`、`Year_Decade`、`Year_From_2000`、`Gender_x_AgeGroup`、`Is_Child`、`Age_Centered`
- 对比"基础 6 特征" vs "构造后 13 特征"的模型性能
- 讨论"领域知识为什么重要"——特征质量决定模型上限

让我们带着"量纲差异如何影响模型"这个问题，进入第一部分的标准化比较实验！

---
