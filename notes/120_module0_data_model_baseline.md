# 模块 0：数据加载、预处理与模型训练基础

> 本模块是案例教程 12「模型解释 — SHAP + LIME + 综合可解释性分析」的**起点模块**。在打开机器学习模型的"黑箱"之前，我们必须先把数据加载进内存、构造目标变量、精选一组基础特征、做标签编码、划分训练/测试集、做缺失值插补和标准化，最后训练一个**随机森林分类器**作为后续所有解释工作的"被解释对象"。
>
> 本模块最核心的知识点有三个：**一是 SHAP 与 LIME 两个解释库的导入**——这是本教程相比此前案例新增的核心内容，包括 `shap`、`lime`、`lime.lime_tabular`；**二是** **`permutation_importance`** **的导入位置**——它来自 `sklearn.inspection`，是模型无关的全局解释工具；**三是为什么本教程选择随机森林而不是逻辑回归作为被解释模型**——因为 SHAP 的 `TreeExplainer` 只对树模型高效，且 RF 在本数据集上 AUC 最高、稳定性最好。

***

## 学习目标

学完本模块后，你将能够：

1. **理解模型解释的整体框架**：知道"全局解释"和"局部解释"的区别，以及 SHAP、LIME、Permutation Importance 各自的角色。
2. **掌握 SHAP 与 LIME 库的导入**：明白 `import shap`、`import lime`、`import lime.lime_tabular` 各自提供什么能力。
3. **理解** **`permutation_importance`** **的来源**：知道它来自 `sklearn.inspection`，是 sklearn 官方提供的模型无关解释工具。

   <br />

***

## 一、导入必要的库

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, warnings, time
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.inspection import permutation_importance

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
```

### 1.1 基础库（与前序教程相同）

 

### 1.2 SHAP 与 LIME 库（本教程核心新增）

#### `import shap`

**SHAP**（SHapley Additive exPlanations）是模型解释的**核心工具**，基于博弈论的 Shapley Value。本教程用它做：

- `shap.TreeExplainer(rf_model)`：树模型专用解释器（高效）。
- `shap.summary_plot()`：蜂群图（Bee Swarm）。
- `shap.waterfall_plot()`：瀑布图（局部解释）。
- `shap.Explanation()`：构造解释对象。

> 💡 **SHAP 的三句话原理**
>
> 1. **base value（基准值）** = 数据集中平均预测概率（本实验中 ≈ 4.9%，即 VIVO 比例）。
> 2. 对于每个样本：**预测概率 = base value + Σ(各特征的 SHAP 值)**。
> 3. SHAP 值可正可负：**正值 → 推高预测概率（→ VIVO）**，**负值 → 推低预测概率（→ MORTO）**。

#### `import lime` 和 `import lime.lime_tabular`

**LIME**（Local Interpretable Model-agnostic Explanations）是另一种局部解释工具。它的思路与 SHAP 不同：

1. 在你关心的样本附近，**随机生成扰动样本**。
2. 用黑箱模型对这些扰动样本预测。
3. 在扰动样本上训练一个**简单模型**（线性回归），简单模型的系数 = 局部特征贡献。

`lime.lime_tabular` 是 LIME 处理表格数据的子模块，提供 `LimeTabularExplainer` 类。

 

***


## 二、训练主模型 — 随机森林（本模块核心）

```python
# ============================================================================
# 训练主模型 — Random Forest (AUC最高+稳定性最好)
# ============================================================================
print("\n[1] 训练主模型 (Random Forest)...")
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1)
rf_model.fit(X_tr_final, y_tr)
y_prob = rf_model.predict_proba(X_te_final)[:, 1]
auc = roc_auc_score(y_te, y_prob)
print(f"    RF 测试 AUC = {auc:.4f}")
```

### 2.1 为什么选择随机森林？

> 💡 **重点概念：为什么用 RF 作为被解释模型？**
>
> 本教程选择 RF 有三个原因：
>
> 1. **AUC 最高**：在前序模型对比实验中，RF 在本数据集上 AUC 表现最好且最稳定。
> 2. **SHAP TreeExplainer 高效**：SHAP 对树模型有专用解释器 `TreeExplainer`，复杂度 O(n×m)，而对任意模型用 `KernelExplainer` 复杂度 O(2^m×n)，慢得多。
> 3. **非线性建模能力**：RF 能捕捉特征间的非线性关系和交互效应，SHAP 解释能揭示这些非线性结构（如 year 与 Diagnostic.means 的交互）。

### 2.2 RandomForestClassifier 参数详解

#### `n_estimators=200`

**树的数量**。200 棵决策树。树越多，模型越稳定（方差越小），但计算成本越高。200 是一个常用的平衡值。

#### `max_depth=8`

**树的最大深度**。限制每棵树最多 8 层。这是**正则化**手段——防止树过深导致过拟合。深度 8 在 6 个特征的数据集上能捕捉足够的交互，又不会过拟合。

> 💡 **为什么限制深度？**
>
> 不限制深度的 RF 容易过拟合——每棵树会一直分裂到叶节点纯度极高，导致模型记住训练数据的噪声。`max_depth=8` 让每棵树适度"早停"，提高泛化能力。这也让 SHAP 解释更稳定（过拟合模型的 SHAP 值会非常极端）。

#### `class_weight='balanced'`

**类别权重**。自动根据类别频率调整权重，使少数类获得更大权重。本数据集 VIVO 占 49.30%、MORTO 占 50.70%，虽然接近平衡，但用 `balanced` 可以让模型更均衡地对待两类。

> 💡 **`class_weight='balanced'`** **对 SHAP 的影响**
>
> `class_weight` 会改变模型的预测概率分布，从而改变 SHAP 的 base value。本实验中，base value ≈ 0.049（即 RF 预测的平均 VIVO 概率约 4.9%）。这个数字远低于真实的 VIVO 比例（49.30%），是因为 `class_weight='balanced'` 让模型对两类"一视同仁"，预测概率被压低了。这不影响 SHAP 的相对解释力，但要注意 base value 不等于真实正类比例。

#### `random_state=RANDOM_STATE`

**随机种子**。保证 RF 的训练可复现。RF 的随机性来自：每棵树的样本采样（bootstrap）和每棵树分裂时的特征采样。

#### `n_jobs=-1`

**并行计算**。`-1` 表示用所有 CPU 核心并行训练。200 棵树可以完全并行，所以 `n_jobs=-1` 能大幅加速训练。

### 2.3 训练与评估

```python
rf_model.fit(X_tr_final, y_tr)
y_prob = rf_model.predict_proba(X_te_final)[:, 1]
auc = roc_auc_score(y_te, y_prob)
print(f"    RF 测试 AUC = {auc:.4f}")
```

- `rf_model.fit(X_tr_final, y_tr)`：在训练集上训练 RF。
- `rf_model.predict_proba(X_te_final)[:, 1]`：预测测试集每个样本的 VIVO 概率。`predict_proba` 返回形状 `(n_samples, 2)` 的数组，第 0 列是 MORTO 概率，第 1 列是 VIVO 概率，所以取 `[:, 1]`。
- `roc_auc_score(y_te, y_prob)`：计算 AUC。

实际运行输出：

```
[1] 训练主模型 (Random Forest)...
    RF 测试 AUC = 0.8220
```

> 💡 **AUC = 0.8220 意味着什么？**
>
> AUC = 0.82 表示：随机选一个 VIVO 样本和一个 MORTO 样本，模型有 82% 的概率给 VIVO 样本更高的预测分数。这是一个**良好**但非**优秀**的性能——医学预测模型通常要求 AUC > 0.8 才有临床价值。这个 AUC 水平足够让我们做有意义的 SHAP 解释。

***


## 三、SHAP 的理论基础：Shapley Value（提前预习）

在进入下一模块的 SHAP 实战之前，我们先了解 SHAP 的理论基础——**Shapley Value**。

### 3.1 博弈论起源

Shapley Value 由诺贝尔经济学奖得主 Lloyd Shapley 于 1953 年提出，源自**合作博弈论**。考虑这样一个问题：

> 一群玩家合作完成一项任务，获得一笔收益。如何**公平地**把收益分配给每个玩家？

Shapley 给出了一个数学上唯一的答案，满足四条公理（见下文）。

### 3.2 SHAP 的类比映射

| 博弈论概念 | 类比            | SHAP 对应                       |
| ----- | ------------- | ----------------------------- |
| 玩家    | 特征            | year, Age, Code.Profession... |
| 联盟    | 特征子集          | 用部分特征预测                       |
| 收益    | 预测结果          | 预测概率                          |
| 贡献分配  | Shapley Value | SHAP 值                        |

### 3.3 Shapley Value 的四条公理

Shapley Value 是**唯一**满足以下四条公理的特征贡献分配方法：

| 公理                  | 含义                    | 为什么重要    |
| ------------------- | --------------------- | -------- |
| **效率（Efficiency）**  | 所有特征贡献之和 = 总预测 - 基准值  | 保证解释是完整的 |
| **对称性（Symmetry）**   | 两个特征贡献相同 → 分配相同       | 公平       |
| **虚拟性（Dummy）**      | 不贡献的特征 → 分配 0         | 不引入噪声    |
| **可加性（Additivity）** | 合并两个模型的 SHAP = 分别计算的和 | 一致性保证    |

> 💡 **为什么这四条公理重要？**
>
> 这四条公理保证了 SHAP 解释的**公平性**和**一致性**。其他解释方法（如 LIME）不满足全部公理，所以解释结果可能不稳定。SHAP 是目前**唯一**满足这四条公理的解释方法，这也是它在医学 AI 论文中成为标准的原因。

### 3.4 SHAP 的计算公式（直觉理解）

对于特征 i，它的 SHAP 值 ϕ\_i 计算如下：

```
ϕ_i = Σ [S ⊆ N \ {i}]  (|S|! × (|N|-|S|-1)!) / |N|! × [v(S ∪ {i}) - v(S)]
```

其中：

- N 是所有特征的集合。
- S 是 N 的一个子集（不包含 i）。
- v(S) 是用子集 S 的特征预测的"价值"（如平均预测概率）。
- v(S ∪ {i}) - v(S) 是**加入特征 i 后的边际贡献**。
- 前面的阶乘项是**权重**，保证所有子集都被公平考虑。

直觉上：SHAP 值 = 特征 i 在所有可能的特征组合中的**平均边际贡献**。

> 💡 **为什么 SHAP 计算慢？**
>
> 上面公式的求和是对**所有子集** S 进行的。如果有 m 个特征，子集数量是 2^m。对于 m=6，2^6=64 个子集，还能算；但对于 m=20，2^20 ≈ 100 万，太慢了。
>
> 这就是为什么 SHAP 对树模型用 `TreeExplainer`——它利用树结构的特性，把复杂度从 O(2^m) 降到 O(n×m)，大幅加速。对于任意模型，只能用 `KernelExplainer`，它用蒙特卡洛采样近似，慢但通用。

***

## 小贴士

1. **SHAP vs LIME 的本质区别**：SHAP 基于博弈论的 Shapley Value，有严谨数学保证；LIME 是局部线性近似，是启发式方法。在最重要的判断上两者通常一致，但细节有差异。
2. **`feature_names_short`** **的命名**：本教程用 `feature_names_short` 而不是 `feature_names`，是为了和 SHAP 库内部的变量名区分。SHAP 的很多函数接受 `feature_names` 参数，传入这个数组即可。
3. **`n_jobs=-1`** **的注意事项**：并行训练会占用所有 CPU 核心，训练期间电脑可能卡顿。如果你的电脑内存不足，可以改成 `n_jobs=4` 或更小。
4. **`max_depth=8`** **的选择**：这个值是经验性的。对于 6 个特征的数据集，深度 8 足够捕捉所有二阶交互（2^6=64 < 2^8=256）。如果特征更多，可以适当增大深度。
5. **SHAP 的 base value 不等于真实正类比例**：本实验中 base value ≈ 0.049，但 VIVO 真实比例是 49.30%。这是因为 RF 用了 `class_weight='balanced'`，预测概率被压低。这不影响 SHAP 的相对解释力，但解读 base value 时要注意。

***
 
