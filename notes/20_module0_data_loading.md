# 模块 0：数据加载与特征选择

> 本模块是案例教程 3「数据预处理与缺失值插补」的起点，承接案例教程 1（EDA）和案例教程 2（统计学分析）。在比较不同插补方法之前，我们必须先把数据加载进内存、构造目标变量、精选一组有代表性的特征，并对样本做合理采样。
>
> 本模块最核心的知识点有两个：**一是 sklearn 的"实验性 API"机制**（为什么用 `IterativeImputer` 必须先 `import enable_iterative_imputer`）；**二是特征选择的设计哲学**——为什么在 5 个特征里要刻意保留一个缺失率高达 15.31% 的 `Raca.Color`。

***

## 学习目标

学完本模块后，你将能够：

1. **理解每个 import 语句的作用**：特别是 sklearn 的 `model_selection`、`preprocessing`、`impute`、`metrics`、`calibration` 等子模块分别在机器学习流程中扮演什么角色。
2. **掌握 sklearn 实验性 API 的启用机制**：明白为什么 `IterativeImputer` 必须先 `from sklearn.experimental import enable_iterative_imputer` 才能使用，以及这背后的工程考量。
3. **理解随机种子和采样规模的设计**：知道 `RANDOM_STATE = 42` 如何保证可复现性，以及为什么采样 8 万条而非用全量 20 万条。
4. **回顾** **`map`** **vs** **`==`** **的本质区别**：明白为什么本教程继续用 `map` 创建目标变量，缺失值如何被正确保留。
5. **深入理解** **`features_config`** **字典的设计哲学**：明白为什么只选 5 个特征，每个特征的临床含义和缺失率，以及为什么刻意保留高缺失率特征。
6. **掌握字典推导式分离数值/分类特征**：理解 `[k for k, v in features_config.items() if v == 'numerical']` 这种简洁写法。
7. **掌握无放回采样**：理解 `np.random.choice` 的 `replace=False` 参数，以及 `.iloc[]` 索引和 `.copy()` 的作用。

***

## 一、导入必要的库

本教程相比前两个案例教程，新增了大量 sklearn 模块，因为我们要做的是完整的机器学习流程（划分→编码→插补→建模→评估）。下面是完整的导入代码：

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
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, roc_curve,
                             brier_score_loss, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与前两个教程相同）

### 1.2 sklearn 模块详解（本教程新增重点）

sklearn（scikit-learn）是 Python 最主流的机器学习库。本教程从 sklearn 的 5 个子模块导入了多个类和函数，下面逐一解释。

#### `from sklearn.model_selection import train_test_split` — 数据划分

**`model_selection`** 子模块提供了数据划分、交叉验证、超参数调优等工具。

- **`train_test_split`** 是其中最常用的函数，用于把数据集划分成训练集和测试集。
- 在本教程中，我们会把 80,000 条样本划分成 70% 训练集（56,000 条）和 30% 测试集（24,000 条）。
- 关键参数 `stratify=y` 会做分层抽样，保持训练集和测试集的标签分布一致（详见模块 1）。

#### `from sklearn.preprocessing import StandardScaler, LabelEncoder` — 标准化和标签编码

**`preprocessing`** 子模块提供了数据预处理工具，包括缩放、编码、二值化等。

- **`StandardScaler`**：标准化器，把数值特征缩放成均值 0、标准差 1 的分布。许多机器学习模型（如逻辑回归、SVM、KNN）在标准化后表现更好、收敛更快。
- **`LabelEncoder`**：标签编码器，把分类变量的字符串类别（如 `'M'`、`'F'`）映射成整数（如 `0`、`1`）。本教程用它对 `Gender`、`Diagnostic.means`、`Raca.Color` 等分类特征做编码（详见模块 1）。

#### `from sklearn.impute import SimpleImputer, KNNImputer` — 简单插补和 KNN 插补

**`impute`** 子模块提供了缺失值插补工具——这是本教程的核心。

- **`SimpleImputer`**：简单插补器，用单一统计量（均值、中位数、众数、常数）填充缺失值。本教程用它实现"均值插补"策略。
- **`KNNImputer`**：K 近邻插补器，利用样本间的相似性插补缺失值。对于某个缺失值，找到 K 个最相似的样本（基于其他特征计算距离），用它们的均值填充。本教程用它实现"KNN 插补"策略。

#### `from sklearn.experimental import enable_iterative_imputer` — 启用实验性 API（重点）

这是本教程**最特殊**的一行 import，需要重点解释。

**什么是 sklearn 的"实验性 API"？**

sklearn 把一些还在试验阶段、API 可能变动的类放在 `sklearn.experimental` 命名空间下。这些类的功能已经实现，但开发者还不保证其接口在未来版本中稳定——可能改名、改参数、甚至移除。为了提醒用户"这个功能还不稳定，使用需谨慎"，sklearn 设计了一种**强制启用机制**：

- 你**不能**直接 `from sklearn.impute import IterativeImputer`，否则会报错。
- 你**必须**先 `from sklearn.experimental import enable_iterative_imputer`，这行代码会"解锁" `IterativeImputer`，然后才能从 `sklearn.impute` 中导入它。

**为什么要有这种机制？**

这是一种工程上的"软约束"：

1. **明确告知用户风险**：强迫用户多写一行 import，等于让用户主动声明"我知道这是实验性功能，愿意承担 API 变动的风险"。
2. **防止误用**：如果直接能用，很多初学者会不知不觉地用上不稳定的功能，等 sklearn 升级后代码报错却不知道原因。
3. **未来迁移**：当实验性功能稳定后，sklearn 会把它"转正"——届时可以去掉 `enable_iterative_imputer` 这行，直接 `from sklearn.impute import IterativeImputer` 即可。

> 💡 **重点概念：sklearn 的实验性 API 机制**
>
> `from sklearn.experimental import enable_iterative_imputer` 这行代码**本身不直接提供任何功能**，它的作用是"注册一个开关"，告诉 sklearn："我要启用 IterativeImputer 这个实验性功能"。只有先执行这行，下一行 `from sklearn.impute import IterativeImputer` 才不会报错。
>
> 这种"先 enable 再 import"的模式是 sklearn 管理实验性功能的统一做法。类似的还有 `enable_halving_search_cv` 等。如果你在 sklearn 文档里看到某个类标注了 "experimental"，通常都需要这种启用方式。
>
> **记忆口诀**：实验性功能 = `enable_xxx`（开关） + `import xxx`（实际导入），缺一不可。

#### `from sklearn.impute import IterativeImputer` — MICE 多重插补

这行必须在 `enable_iterative_imputer` 之后执行。

- **`IterativeImputer`**：迭代插补器，也叫 MICE（Multivariate Imputation by Chained Equations，链式方程多重插补）。它把每个有缺失的特征当作因变量、其他特征当作自变量，用回归模型迭代预测缺失值，循环多次直到收敛。
- 这是医学研究中最严谨的插补方法之一，因为它**保留了变量间的联合分布关系**，且能引入适当的随机性（不会像均值插补那样低估方差）。
- 本教程用它实现"MICE 插补"策略，作为与均值插补、KNN 插补对比的"高级方法"。

#### `from sklearn.linear_model import LogisticRegression` — 逻辑回归

**`linear_model`** 子模块提供了各种线性模型。

- **`LogisticRegression`**：逻辑回归，二分类任务最经典的模型。本教程用它作为"评估插补方法好坏"的统一模型——所有插补方法都喂给同一个逻辑回归，比较它们的 AUC、Recall、Brier Score，这样能隔离出"插补方法"这个变量对性能的影响。

#### `from sklearn.metrics import (...)` — 评估指标函数

**`metrics`** 子模块提供了大量模型评估指标。本教程用括号跨行导入了一组函数：

- **`roc_auc_score`**：计算 ROC 曲线下面积（AUC），衡量模型区分正负类的能力。AUC=1 完美，AUC=0.5 等同随机。
- **`recall_score`**：计算召回率（Recall），即真正例占所有实际正例的比例。本教程关注 `Recall(VIVO)`——有多少存活患者被正确识别。
- **`roc_curve`**：计算 ROC 曲线的三个序列（FPR、TPR、阈值），用于绘制 ROC 曲线图。
- **`brier_score_loss`**：Brier 分数，衡量预测概率的校准程度（预测概率和真实标签的均方误差）。Brier 越低，预测概率越可靠。
- **`precision_recall_curve`**：计算 PR 曲线，用于不平衡数据集的评估。
- **`average_precision_score`**：平均精度（AP），即 PR 曲线下面积，对不平衡数据比 AUC 更敏感。

#### `from sklearn.calibration import calibration_curve` — 校准曲线

**`calibration`** 子模块提供了概率校准工具。

- **`calibration_curve`**：计算校准曲线的数据点。它把预测概率分箱，计算每个箱内的平均预测概率和实际正例率，然后画出来对比。完美校准的模型，校准曲线应该贴近对角线。
- 本教程用它评估"插补方法是否影响模型预测概率的可靠性"——因为医学场景下，预测概率的准确性（而非只是分类对错）非常重要。

> 💡 **小贴士**：sklearn 的子模块划分遵循机器学习流程的顺序：`model_selection`（划分）→ `preprocessing`（预处理）→ `impute`（插补）→ `linear_model`（建模）→ `metrics`（评估）→ `calibration`（校准）。记住这个流程，就能快速定位到需要的子模块。

***

## 二、路径配置与目录创建

```python
BASE_DIR = ""
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

这部分与前两个教程完全一致

***

## 三、随机种子与采样规模

```python
RANDOM_STATE = 42
N_SAMPLES = 80000  # 采样样本数 (平衡计算开销与统计可靠性)
```

### 3.1 `RANDOM_STATE = 42` — 可复现性的基石

**`RANDOM_STATE`** 是一个整数种子，用于控制所有涉及随机性的操作，确保**结果可复现**。

在本教程中，`RANDOM_STATE` 会被用在以下地方：

| 使用位置                                               | 作用               |
| -------------------------------------------------- | ---------------- |
| `np.random.seed(RANDOM_STATE)`                     | 固定 numpy 的随机数生成器 |
| `np.random.choice(...)`                            | 固定采样时选中的样本索引     |
| `train_test_split(..., random_state=RANDOM_STATE)` | 固定训练集/测试集的划分     |

**为什么需要固定随机种子？**

机器学习流程中有大量随机操作（采样、划分、模型初始化等）。如果不固定种子，每次运行脚本结果都会不同——这会让调试变得困难，也让"对比不同插补方法"失去意义（你无法判断性能差异是来自插补方法还是随机波动）。

固定种子后，**任何人、任何时间、任何机器上运行这段代码，都会得到完全相同的结果**，这就是"可复现性"。

**为什么是 42？**

42 是编程界的"梗"——来自《银河系漫游指南》（The Hitchhiker's Guide to the Galaxy），书中"生命、宇宙和一切的终极答案"就是 42。它已经成为机器学习社区最常用的随机种子（仅次于 0 和 1）。

> ⚠️ **常见问题**：我设置了 `RANDOM_STATE = 42`，为什么不同版本的 sklearn 结果还是略有不同？
>
> 因为不同版本的 sklearn 内部实现可能有细微变化（如算法优化、默认参数调整）。`RANDOM_STATE` 只保证**同一版本、同一环境**下可复现。跨版本对比时，仍可能有微小差异。

### 3.2 `N_SAMPLES = 80000` — 采样规模的设计

本教程从 209,758 条有标签样本中**采样 80,000 条**进行分析，而不是用全量数据。

**为什么不直接用全量 20 万条？**

1. **计算开销**：本教程要对比 4 种插补方法（Complete Case、Mean、KNN、MICE），其中 **KNN 插补的计算复杂度是 O(n²)**——在 20 万条数据上跑 KNN 可能要几分钟甚至更久，而 8 万条只需约 20 秒。
2. **教学效率**：教程需要能快速跑通，让学生聚焦于"方法对比"而非"等待计算"。
3. **统计可靠性**：8 万条样本已经足够大，统计检验和模型评估的置信区间都很窄，结论不会因样本量不足而失真。

**为什么不用更少，比如 1 万条？**

1. **统计效力**：样本太少会导致 AUC、Brier Score 等指标的方差变大，难以区分不同插补方法的差异。
2. **缺失值数量**：`Raca.Color` 缺失率 15.31%，8 万条样本中约有 12,248 条缺失——这个量足够让插补方法"有活干"，能体现不同方法的效果差异。如果只有 1 万条，缺失样本只有约 1,500 条，差异可能被噪声淹没。
3. **类别覆盖**：分类特征（如 `Diagnostic.means` 有 7 类）需要每类都有足够样本，8 万条能保证每个类别都有数百条以上。

> 💡 **小贴士**：在实际研究中，应该用全量数据。采样只是为了教学演示。生产环境中，如果 KNN 太慢，可以考虑：(1) 用 `n_neighbors` 较小的 KNN；(2) 用近似最近邻算法（如 `faiss`）；(3) 改用更快的插补方法（如 MICE）。

***

## 四、加载数据与创建目标变量

```python
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')

df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
df = df.dropna(subset=['target'])
```

### 4.1 读取 CSV

`pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')` 与前两个教程完全一致：

- `low_memory=False`：一次性读取整个文件，避免类型推断警告。
- `encoding='latin-1'`：处理可能残留的葡萄牙语特殊字符。

### 4.2 创建目标变量——回顾 `map` vs `==`

本教程继续用 `map` 而非 `==` 创建目标变量：

```python
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

这是从案例教程 2 继承的关键改进。两种方法的本质区别在于**对缺失值的处理**：

| Status.Vital 原值 | `(== 'VIVO').astype(int)` | `.map({'VIVO':1, 'MORTO':0})` |
| --------------- | ------------------------- | ----------------------------- |
| VIVO            | 1                         | 1                             |
| MORTO           | 0                         | 0                             |
| NaN（缺失）         | **0** ⚠️ 被误判为 MORTO       | **NaN** ✅ 保持缺失                |

**为什么** **`==`** **会把 NaN 当作 0？**

因为 IEEE 754 浮点数标准规定：**任何与 NaN 的比较都返回 False**，包括 `NaN == 'VIVO'`。所以 `(NaN == 'VIVO')` 为 `False`，`.astype(int)` 后变成 `0`，缺失值被错误地标记为"死亡"。

**为什么** **`map`** **更安全？**

`map()` 对字典中**没有定义的值**（包括 NaN）返回 NaN，保持缺失状态。这样后续 `dropna(subset=['target'])` 才能正确识别并删除这些样本。

> 💡 **核心知识点（回顾案例教程 2）**：
>
> 当原始标签列可能存在缺失值时，**必须用** **`map`** **而不是** **`==`** 创建目标变量。`map` 让缺失值保持 NaN，`==` 会把缺失值误判为 0。这是从案例教程 1 升级到案例教程 2/3 的关键改进。

### 4.3 过滤缺失样本

```python
df = df.dropna(subset=['target'])
```

`dropna(subset=['target'])` 删除 `target` 列为 NaN 的行。由于前面用 `map` 创建 target，`Status.Vital` 缺失的样本在 target 列也是 NaN，会被这一步正确删除。

处理后：原始 1,778,176 条 → 保留 209,758 条有标签样本。

***

## 五、特征选择（本模块核心）

这是本模块**最重要**的部分。我们先看代码：

```python
features_config = {
    'Age': 'numerical',           # 0.15% 缺失 — 连续数值
    'year': 'numerical',          # 0.00% 缺失 — 连续数值
    'Gender': 'categorical',      # ~0% 缺失 — 二分类
    'Diagnostic.means': 'categorical',  # 0.36% 缺失 — 诊断方式(7类)
    'Raca.Color': 'categorical',  # 15.31% 缺失 — 人种(5类)
}
```

### 5.1 为什么只选 5 个特征？

原始数据集有 38 个字段，为什么本教程只选 5 个？这是基于三个原则的**精心设计**：

**原则 1：兼顾缺失率的多样性**

本教程的核心目标是**对比不同插补方法**。如果所有特征都没有缺失值，插补方法就无从施展；如果所有特征缺失率都很高，又可能引入过多噪声。所以需要选择**缺失率有梯度**的特征：

| 特征                 | 缺失率    | 在对比中的作用            |
| ------------------ | ------ | ------------------ |
| `year`             | 0.00%  | 作为"无缺失"基线          |
| `Gender`           | \~0%   | 几乎无缺失，提供稳定信息       |
| `Age`              | 0.15%  | 低缺失，测试少量缺失的处理      |
| `Diagnostic.means` | 0.36%  | 低缺失，多类别分类变量        |
| `Raca.Color`       | 15.31% | **高缺失**，是插补方法发挥的关键 |

**原则 2：兼顾数据类型的多样性**

需要同时包含数值型和分类型特征，因为两种类型的插补逻辑不同：

- 数值型（`Age`、`year`）：可以用均值、中位数、KNN 回归插补
- 分类型（`Gender`、`Diagnostic.means`、`Raca.Color`）：通常用众数或 KNN 分类插补

**原则 3：兼顾临床意义**

每个特征都要有明确的临床含义，不能是无关变量：

| 特征                 | 临床含义                       |
| ------------------ | -------------------------- |
| `Age`              | 患者年龄，癌症生存的最强预测因子之一         |
| `year`             | 诊断年份，反映医疗技术进步的影响           |
| `Gender`           | 性别，某些癌症有性别差异               |
| `Diagnostic.means` | 诊断方式（如病理确诊、临床诊断等），反映确诊的确定性 |
| `Raca.Color`       | 人种/肤色，反映社会经济因素和遗传差异对生存的影响  |

### 5.2 每个特征的缺失率和临床含义详解

#### `Age`（年龄）— 0.15% 缺失

- **类型**：连续数值（数值型）
- **缺失率**：0.15%（8 万样本中约 120 条缺失）
- **临床含义**：患者确诊时的年龄。年龄是癌症生存预测最重要的因子之一——老年患者通常预后更差。
- **缺失原因**：极少数记录登记时漏填年龄，属于随机缺失（MCAR）。

#### `year`（诊断年份）— 0.00% 缺失

- **类型**：连续数值（数值型）
- **缺失率**：0%（无缺失）
- **临床含义**：患者确诊的年份。随着医疗技术进步，近年确诊的患者生存率通常更高。
- **作用**：作为"无缺失基线"特征，帮助隔离出其他特征缺失的影响。

#### `Gender`（性别）— \~0% 缺失

- **类型**：二分类（分类型）
- **缺失率**：接近 0%
- **临床含义**：患者性别。某些癌症（如乳腺癌）有显著性别差异。
- **取值**：通常为 `M`（男性）和 `F`（女性）两类。

#### `Diagnostic.means`（诊断方式）— 0.36% 缺失

- **类型**：多分类（分类型，7 个类别）
- **缺失率**：0.36%（8 万样本中约 288 条缺失）
- **临床含义**：癌症的确诊方式，如病理确诊（最可靠）、临床诊断、影像学诊断等。诊断方式反映了确诊的确定性，间接影响生存预测。
- **缺失原因**：少数记录的诊断方式字段未填写。

#### `Raca.Color`（人种/肤色）— 15.31% 缺失（重点）

- **类型**：多分类（分类型，5 个类别）
- **缺失率**：15.31%（8 万样本中约 12,248 条缺失）——**本教程中缺失率最高的特征**
- **临床含义**：患者的人种/肤色分类（巴西人口普查标准，如白人、黑人、棕色人种、亚裔、原住民）。人种反映遗传背景和社会经济地位，两者都会影响癌症生存。
- **缺失原因**：人种信息属于敏感数据，部分患者拒绝提供；且巴西早期登记系统对人种字段不强制要求。

### 5.3 为什么 `Raca.Color` 缺失率高达 15.31% 仍被选中？

这是本模块**最关键的设计决策**。初学者可能会想："缺失率这么高，直接删掉这个特征不就行了？"——但这样做会让本教程失去意义。

> 💡 **重点概念：为什么需要保留一定缺失率的特征？**
>
> 本教程的核心目标是**对比不同插补方法的效果**。如果所有特征都没有缺失值，那么：
>
> - Mean Imputation、KNN Imputation、MICE 都没有用武之地（没有缺失可插补）
> - Complete Case（删除缺失行）和 Mean Imputation 的结果会完全一样
> - 无法展示"不同插补方法如何影响模型性能"
>
> 因此，**必须刻意保留一个缺失率较高的特征**，让插补方法"有活干"。`Raca.Color` 的 15.31% 缺失率恰到好处：
>
> 1. **缺失量足够大**：8 万样本中约 12,248 条缺失，插补方法的效果差异能被模型感知到。
> 2. **缺失率不过高**：15% 不算极端，不会让特征失去统计意义（仍有 85% 的数据可用）。
> 3. **临床意义明确**：人种确实是癌症生存的相关因素，不是凑数的变量。
> 4. **类别数适中**：5 个类别，既不太少（二分类太简单）也不太多（类别过多会让编码复杂）。
>
> 如果选一个缺失率 50% 以上的特征，插补的"猜测成分"太大，结果不可靠；如果选缺失率 1% 的特征，插补方法差异不明显。15% 是一个"甜点区间"（sweet spot）。

### 5.4 `features_config` 字典的设计巧思

注意 `features_config` 是一个**字典**，键是特征名，值是类型（`'numerical'` 或 `'categorical'`）。这种设计有几个好处：

1. **集中管理**：所有特征及其类型在一个地方定义，修改方便。
2. **可派生**：从这一个字典可以派生出 `feature_names`、`numerical_features`、`categorical_features` 三个列表，无需重复定义。
3. **可读性强**：一眼就能看出每个特征的类型，注释也能直接写在旁边。

***

## 六、从字典派生特征列表

```python
feature_names = list(features_config.keys())
numerical_features = [k for k, v in features_config.items() if v == 'numerical']
categorical_features = [k for k, v in features_config.items() if v == 'categorical']
```

### 6.1 `list(features_config.keys())` — 获取所有特征名

`features_config.keys()` 返回字典的所有键（特征名），`list()` 把它转成列表。

结果：`feature_names = ['Age', 'year', 'Gender', 'Diagnostic.means', 'Raca.Color']`

### 6.2 字典推导式分离数值和分类特征

```python
numerical_features = [k for k, v in features_config.items() if v == 'numerical']
categorical_features = [k for k, v in features_config.items() if v == 'categorical']
```

这是 Python 的**列表推导式**（list comprehension）语法，用于从字典中筛选出符合条件的键。

**拆解这段代码**：

- `features_config.items()`：返回字典的所有键值对，如 `[('Age', 'numerical'), ('year', 'numerical'), ('Gender', 'categorical'), ...]`。
- `for k, v in ...`：遍历每个键值对，`k` 是特征名，`v` 是类型。
- `if v == 'numerical'`：筛选条件，只保留类型为 `'numerical'` 的特征。
- 最前面的 `k`：对每个满足条件的键值对，取出键（特征名）放入结果列表。

**执行过程示意**：

```
features_config.items():
  ('Age', 'numerical')              → v=='numerical' ✓ → 取 'Age'
  ('year', 'numerical')             → v=='numerical' ✓ → 取 'year'
  ('Gender', 'categorical')         → v=='numerical' ✗ → 跳过
  ('Diagnostic.means', 'categorical') → v=='numerical' ✗ → 跳过
  ('Raca.Color', 'categorical')     → v=='numerical' ✗ → 跳过

结果: numerical_features = ['Age', 'year']
```

同理，`categorical_features = ['Gender', 'Diagnostic.means', 'Raca.Color']`。

> 💡 **小贴士**：列表推导式是 Python 的特色语法，比传统的 `for` 循环 + `append` 更简洁高效。等价的传统写法是：
>
> ```python
> numerical_features = []
> for k, v in features_config.items():
>     if v == 'numerical':
>         numerical_features.append(k)
> ```
>
> 列表推导式把 4 行压缩成 1 行，且执行速度更快（底层 C 实现）。

***

## 七、采样

```python
np.random.seed(RANDOM_STATE)
if len(df) > N_SAMPLES:
    sample_idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df_sample = df.iloc[sample_idx].copy()
else:
    df_sample = df.copy()
```

### 7.1 `np.random.seed(RANDOM_STATE)` — 固定随机数生成器

在调用任何 numpy 随机函数之前，先设置随机种子。这样 `np.random.choice` 每次运行都会选中**相同的样本索引**，保证可复现。

### 7.2 `np.random.choice(len(df), N_SAMPLES, replace=False)` — 无放回采样

**`np.random.choice`** 是 numpy 的随机采样函数，参数含义：

- 第 1 个参数 `len(df)`：从 `0` 到 `len(df)-1` 的整数序列中采样（即所有样本的行索引）。
- 第 2 个参数 `N_SAMPLES`：采样数量，这里是 80,000。
- `replace=False`：**无放回采样**——每个样本最多被选中一次。

**`replace=False`** **vs** **`replace=True`** **的区别**：

| 参数              | 含义  | 结果                              |
| --------------- | --- | ------------------------------- |
| `replace=False` | 无放回 | 80,000 个**不重复**的样本（子集）          |
| `replace=True`  | 有放回 | 80,000 个样本，但可能有重复（bootstrap 采样） |

本教程用 `replace=False`，因为我们要得到一个**真实数据的子集**，不希望有重复样本。如果用 `replace=True`（有放回），同一个样本可能出现多次，这会人为放大某些样本的影响，破坏统计推断。

> 💡 **小贴士**：`replace=False` 类似于"从一副牌中抽 5 张"——每张牌只能抽一次；`replace=True` 类似于"掷骰子 5 次"——每次都是独立的，可能掷出相同的点数。机器学习中，训练集划分用无放回，Bootstrap 集成方法（如随机森林的 bootstrap sampling）用有放回。

### 7.3 `df.iloc[sample_idx]` — 按位置索引选行

**`df.iloc[]`** 是 pandas 的**基于位置**的索引器（integer location），用整数位置选取行/列。

- `sample_idx` 是一个包含 80,000 个整数的 numpy 数组，表示被选中的行位置。
- `df.iloc[sample_idx]` 返回这些行组成的 DataFrame。

**`iloc`** **vs** **`loc`** **的区别**：

| 索引器    | 基于          | 示例                         |
| ------ | ----------- | -------------------------- |
| `iloc` | 整数位置（第几行）   | `df.iloc[0]` 取第 1 行        |
| `loc`  | 标签（index 值） | `df.loc[0]` 取 index 为 0 的行 |

由于 `sample_idx` 是位置整数，必须用 `iloc`。如果用 `loc`，pandas 会把整数当作 index 标签查找，可能出错（尤其是 index 不连续时）。

### 7.4 `.copy()` — 创建独立副本

```python
df_sample = df.iloc[sample_idx].copy()
```

`.copy()` 创建一个**独立的副本**，后续对 `df_sample` 的修改不会影响原始的 `df`。

**为什么需要** **`.copy()`？**

`df.iloc[sample_idx]` 返回的可能是原始 DataFrame 的**视图**（view，共享内存），也可能是**副本**（copy，独立内存）。pandas 不保证返回哪种。如果返回的是视图，修改 `df_sample` 会意外改变 `df`。显式调用 `.copy()` 强制创建副本，避免这种副作用，也避免 `SettingWithCopyWarning` 警告。

> ⚠️ **常见问题**：什么时候需要加 `.copy()`？
>
> 经验法则：**当你打算后续修改一个从原 DataFrame 派生出来的子集时，就应该加** **`.copy()`**。本教程后续会对 `df_sample` 做大量修改（提取特征、划分、编码），所以必须加 `.copy()`。如果只是读取子集而不修改，可以不加（节省内存）。

### 7.5 `if len(df) > N_SAMPLES` 的容错设计

```python
if len(df) > N_SAMPLES:
    sample_idx = np.random.choice(len(df), N_SAMPLES, replace=False)
    df_sample = df.iloc[sample_idx].copy()
else:
    df_sample = df.copy()
```

这个 `if-else` 是一种**容错设计**：

- 如果原始数据量大于 `N_SAMPLES`（80,000），执行采样。
- 如果原始数据量小于等于 `N_SAMPLES`（比如换了一个小数据集），直接用全量数据，避免 `np.random.choice` 因采样数大于总数而报错。

这种写法让代码更健壮——即使数据集规模变化，代码也能正常运行。

***

## 八、检查缺失情况

```python
missing_info = []
for col in feature_names:
    n_miss = df_sample[col].isnull().sum()
    pct_miss = n_miss / len(df_sample) * 100
    missing_info.append({'Feature': col, 'Missing': n_miss, 'Pct': pct_miss})
```

### 8.1 逐行解释

- `missing_info = []`：创建一个空列表，用于存储每个特征的缺失信息。
- `for col in feature_names:`：遍历 5 个特征。
- `df_sample[col].isnull()`：返回一个布尔 Series，True 表示该位置是缺失值。
- `.sum()`：对布尔 Series 求和（True=1，False=0），得到缺失值的总数。
- `pct_miss = n_miss / len(df_sample) * 100`：计算缺失率（百分比）。
- `missing_info.append({...})`：把该特征的缺失信息存成字典，追加到列表。

### 8.2 `isnull().sum()` 的优雅之处

`df_sample[col].isnull().sum()` 是统计缺失值的**标准惯用法**：

- `isnull()` 返回布尔 Series（True=缺失，False=非缺失）。
- `.sum()` 把 True 当 1、False 当 0 求和，结果就是缺失值的个数。

这比 `len(df[df[col].isnull()])` 更简洁高效。

***

## 九、本模块运行结果

运行本模块代码后，控制台输出大致如下：

```
======================================================================
案例教程 3: 数据预处理与缺失值插补
======================================================================

[0] 加载数据与特征选择...
    有标签样本: 209,758

    所选特征: ['Age', 'year', 'Gender', 'Diagnostic.means', 'Raca.Color']
    数值型: ['Age', 'year']
    分类型: ['Gender', 'Diagnostic.means', 'Raca.Color']
    分析样本量: 80,000
    VIVO: 32,760 (40.95%)
    MORTO: 47,240 (59.05%)
      Age                 缺失:    120 (0.15%)
      year                缺失:      0 (0.00%)
      Gender              缺失:      0 (0.00%)
      Diagnostic.means    缺失:    288 (0.36%)
      Raca.Color          缺失: 12,248 (15.31%)
```

### 关键数字解读

| 指标                  | 数值              | 说明                              |
| ------------------- | --------------- | ------------------------------- |
| 有标签样本               | 209,758         | Status.Vital 非缺失的样本（与案例教程 2 一致） |
| 采样样本量               | 80,000          | 从 209,758 条中无放回采样               |
| VIVO（存活）            | 32,760          | 占 40.95%，与全量数据的 40.95% 一致       |
| MORTO（死亡）           | 47,240          | 占 59.05%，与全量数据的 59.05% 一致       |
| Age 缺失              | 120 (0.15%)     | 极少缺失                            |
| year 缺失             | 0 (0.00%)       | 无缺失                             |
| Gender 缺失           | 0 (0.00%)       | 无缺失                             |
| Diagnostic.means 缺失 | 288 (0.36%)     | 少量缺失                            |
| Raca.Color 缺失       | 12,248 (15.31%) | 高缺失率，是插补方法发挥的关键                 |

### 为什么采样后标签比例与全量数据一致？

采样后 VIVO 仍占 40.95%、MORTO 占 59.05%，与全量数据完全一致。这是因为：

1. **样本量足够大**：8 万条样本的随机采样，大数定律保证比例会接近真实分布。
2. **无放回采样**：每个样本被选中的概率相等，不改变分布。
3. **随机种子固定**：这个比例是可复现的。

> ⚠️ **常见问题**：为什么采样后 VIVO 是 32,760 而不是精确的 32,760？
>
> 实际上，由于随机采样的波动，采样后的比例可能与全量数据有微小差异（如 40.93% 或 40.97%）。但因为样本量大（8 万），波动很小，通常在小数点后两位内。固定 `RANDOM_STATE=42` 后，每次运行都会得到完全相同的 32,760。

***

## 十、小贴士与常见问题

### 💡 小贴士汇总

1. **实验性 API 的启用模式**：`from sklearn.experimental import enable_xxx` + `from sklearn.xxx import XxxImputer`，两行缺一不可。看到 sklearn 文档标注 "experimental" 的类，都要这样启用。
2. **随机种子要全局固定**：不仅 `np.random.seed`，`train_test_split`、`LogisticRegression` 等也要传 `random_state`，才能完全可复现。
3. **采样规模的"甜点区间"**：教学场景下，8 万条既能控制计算开销，又能保证统计可靠性。实际研究应该用全量数据。
4. **特征选择要兼顾缺失率梯度**：对比插补方法时，需要刻意保留一个缺失率较高的特征（如本教程的 `Raca.Color`），否则插补方法无差异可对比。
5. **`iloc`** **vs** **`loc`**：`iloc` 按位置索引（整数），`loc` 按标签索引。用 `np.random.choice` 生成的整数索引必须配合 `iloc`。

### ⚠️ 常见问题

**Q1：为什么我直接** **`from sklearn.impute import IterativeImputer`** **会报错？**

A：因为 `IterativeImputer` 是实验性功能，必须先 `from sklearn.experimental import enable_iterative_imputer` 启用，才能导入。这是 sklearn 的强制机制，提醒你这是不稳定 API。

**Q2：`RANDOM_STATE = 42`** **和** **`np.random.seed(42)`** **有什么区别？**

A：`np.random.seed(42)` 设置的是 numpy 全局随机数生成器的种子，影响所有 numpy 随机函数（如 `np.random.choice`）。`RANDOM_STATE = 42` 只是一个变量，需要传给 sklearn 函数的 `random_state` 参数才起作用。两者配合使用，才能保证全流程可复现。

**Q3：为什么** **`Raca.Color`** **缺失率这么高（15.31%）还要保留？**

A：因为本教程的目标是对比插补方法。如果删掉所有有缺失的特征，插补方法就没有用武之地。15.31% 是一个"甜点区间"——缺失量足够大让插补方法有差异，又不过高导致特征失去意义。

**Q4：`replace=False`** **和** **`replace=True`** **该用哪个？**

A：划分训练集/测试集、采样分析子集时用 `replace=False`（无放回，避免重复样本）；Bootstrap 采样（如随机森林的 bootstrap）用 `replace=True`（有放回，允许重复）。

**Q5：为什么采样后标签比例和全量数据几乎一样？**

A：因为无放回随机采样中，每个样本被选中的概率相等，不改变分布。加上样本量大（8 万），大数定律保证采样比例接近真实比例。

**Q6：`features_config`** **为什么用字典而不用两个列表（一个数值、一个分类）？**

A：字典的好处是"一个特征只定义一次"，避免在两个列表里重复维护。且能通过推导式派生出各种列表，修改时只需改一处。

***

## 十一、本模块小结

### 核心知识点回顾

1. **sklearn 库导入**：
   - `model_selection.train_test_split`：数据划分
   - `preprocessing.StandardScaler, LabelEncoder`：标准化和标签编码
   - `impute.SimpleImputer, KNNImputer`：简单插补和 KNN 插补
   - `experimental.enable_iterative_imputer` + `impute.IterativeImputer`：MICE 多重插补（实验性 API，必须先 enable）
   - `linear_model.LogisticRegression`：逻辑回归模型
   - `metrics.*`：AUC、Recall、Brier Score 等评估指标
   - `calibration.calibration_curve`：校准曲线
2. **实验性 API 机制**：`enable_iterative_imputer` 是"开关"，`IterativeImputer` 是实际功能，缺一不可。这是 sklearn 管理不稳定 API 的统一模式。
3. **随机种子与采样**：`RANDOM_STATE = 42` 保证可复现；`N_SAMPLES = 80000` 平衡计算开销与统计可靠性。
4. **目标变量创建**：继续用 `map` 而非 `==`，让缺失值保持 NaN，`dropna` 才能正确过滤。
5. **特征选择设计哲学**：
   - 5 个特征兼顾缺失率梯度（0% \~ 15.31%）、数据类型（数值+分类）、临床意义。
   - 刻意保留高缺失率的 `Raca.Color`，让插补方法有差异可对比。
6. **字典推导式**：`[k for k, v in features_config.items() if v == 'numerical']` 从一个字典派生出数值/分类特征列表。
7. **无放回采样**：`np.random.choice(len(df), N_SAMPLES, replace=False)` + `df.iloc[idx].copy()`。

### 数据概况

- 原始 209,758 条有标签样本 → 采样 80,000 条
- VIVO 32,760 (40.95%)，MORTO 47,240 (59.05%)
- 5 个特征：Age (0.15%)、year (0%)、Gender (\~0%)、Diagnostic.means (0.36%)、Raca.Color (15.31%)

### 下一模块预告

本模块完成了数据加载、特征选择和采样，得到了一个 80,000 条样本、5 个特征的分析数据集。下一模块（**模块 1：数据划分与编码**）将：

- 把 80,000 条样本划分成训练集（56,000 条）和测试集（24,000 条），用 `stratify=y` 保持标签分布
- 对 3 个分类特征做 LabelEncoder 编码，处理测试集中的"未知类别"
- 保存缺失掩码，用于后续分析插补前后的变化
- 讨论"为什么要在插补前做编码"和"为什么用 LabelEncoder 而非 OneHotEncoder"

***

