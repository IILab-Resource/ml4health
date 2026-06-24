# 模块 2：七个模型定义详解

> 本模块是案例教程 9「机器学习建模 — 7 模型对比」的第三部分，承接模块 1（Pipeline 构建与 CV 评估框架）。在评估框架就绪后，本模块定义 7 个待比较的模型：Logistic Regression、SVM、KNN、Decision Tree、Random Forest、XGBoost、LightGBM。每个模型都有不同的数学原理、适用场景和参数设置。
>
> 本模块最核心的知识点有三个：**一是 7 个模型的数学直觉和参数含义**——从线性模型（LR/SVM）到非参数模型（KNN）到树模型（DT/RF）到梯度提升（XGBoost/LightGBM），每个模型的关键参数（如 `n_estimators=200`、`max_depth=10`、`C=1.0`、`learning_rate=0.1`）如何影响模型行为；**二是不平衡数据的处理方式**——`class_weight='balanced'` 和 `scale_pos_weight` 的区别，为什么 KNN 不支持 class\_weight；**三是 XGBoost 和 LightGBM 的差异**——分裂策略（Level-wise vs Leaf-wise）、采样策略（GOSS + EFB）如何让 LightGBM 更快。

***

## 学习目标

学完本模块后，你将能够：

1. **理解 7 个模型的数学原理**：从线性决策边界（LR）到最大间隔（SVM）到距离投票（KNN）到递归二分（DT）到 Bagging（RF）到梯度提升（XGBoost/LightGBM）。
2. **掌握每个模型的关键参数**：`class_weight='balanced'`、`max_iter=5000`、`C=1.0`、`n_neighbors=15`、`max_depth=10`、`n_estimators=200`、`learning_rate=0.1`、`scale_pos_weight` 等。
3. **理解** **`class_weight='balanced'`** **的原理**：少数类权重 = n\_samples / (n\_classes \* n\_counts)，如何让模型更关注少数类。
4. **理解** **`scale_pos_weight`** **的用法**：XGBoost 不用 class\_weight，而用 scale\_pos\_weight = n\_negative / n\_positive。
5. **理解 KNN 为什么不支持 class\_weight**：KNN 是基于距离的非参数方法，没有损失函数可以加权。
6. **掌握 XGBoost 和 LightGBM 的参数差异**：分裂策略、采样策略、特征处理。
7. **理解** **`try/except ImportError`** **的容错设计**：XGBoost 和 LightGBM 是第三方库，可能未安装。
8. **理解** **`models`** **列表的数据结构**：每个元素是 `(name, model, scale_needed)` 三元组。

***

## 一、模型列表的数据结构

```python
# ============================================================================
# 2. 定义 7 个模型
# ============================================================================
models = [
    ('Logistic Regression',
     LogisticRegression(class_weight='balanced', max_iter=5000,
                        random_state=RANDOM_STATE),
     True),
    ('SVM (Linear)',
     LinearSVC(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE,
               loss='hinge'),
     True),
    ('KNN (k=15)',
     KNeighborsClassifier(n_neighbors=15, n_jobs=N_JOBS),
     True),
    ('Decision Tree',
     DecisionTreeClassifier(class_weight='balanced', max_depth=10,
                            random_state=RANDOM_STATE),
     False),
    ('Random Forest',
     RandomForestClassifier(n_estimators=200, max_depth=12,
                            class_weight='balanced',
                            random_state=RANDOM_STATE, n_jobs=N_JOBS),
     False),
]
```

`models` 是一个列表，每个元素是一个三元组 `(name, model, scale_needed)`：

- `name`：模型名称字符串，用于结果展示和绘图标签。
- `model`：分类器实例，已配置好所有参数。
- `scale_needed`：是否需要标准化（传给 `build_pipeline`）。

这种数据结构的好处是：可以用一个循环统一评估所有模型（模块 3 会看到）。

***

## 二、模型 1：Logistic Regression（逻辑回归）

```python
('Logistic Regression',
 LogisticRegression(class_weight='balanced', max_iter=5000,
                    random_state=RANDOM_STATE),
 True),
```

### 2.1 模型原理

**逻辑回归**是二分类任务最经典的线性模型。它的核心思想是：用一条直线（或超平面）把两类分开，然后用 Sigmoid 函数把线性组合转成概率。

**数学公式**：

$$p = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w^T x + b)}}$$

- $x$：特征向量。
- $w$：权重向量（系数）。
- $b$：偏置（截距）。
- $\sigma$：Sigmoid 函数，把实数压缩到 (0, 1)。
- $p$：预测为正类的概率。

**决策边界**：当 $p = 0.5$ 时，$w^T x + b = 0$，这是一个线性超平面。

### 2.2 参数详解

#### `class_weight='balanced'` — 类别权重（重点）

> 💡💡💡 **重点概念：`class_weight='balanced'`** **的原理**
>
> 默认情况下，逻辑回归对每个样本权重相同。在不平衡数据上，模型会偏向多数类——因为"多预测多数类"能降低整体损失。
>
> `class_weight='balanced'` 自动调整权重：
> $$w\_j = \frac{n\_{samples}}{n\_{classes} \cdot n\_j}$$
>
> 其中 $n\_j$ 是类别 $j$ 的样本数。
>
> 本数据集：
>
> - VIVO: 8,230 样本 → 权重 = 20000 / (2 × 8230) ≈ 1.21
> - MORTO: 11,770 样本 → 权重 = 20000 / (2 × 11770) ≈ 0.85
>
> 少数类（VIVO）的权重更高，模型会更关注"找存活患者"，从而提升 Recall。
>
> 实验结果：LR 的 Recall = 0.8786，如果不加 `balanced`，Recall 会显著降低。

#### `max_iter=5000` — 最大迭代次数

逻辑回归用梯度下降（或拟牛顿法）优化，`max_iter` 是最大迭代次数。默认 100，本教程设为 5000。

**为什么设这么大？**

- 本教程有 9 个特征，量纲差异大（Age 约 120，Code.Profession 约 9999），未标准化的数据上收敛慢。
- 即使有 StandardScaler，某些情况下（如特征高度相关）也可能需要更多迭代。
- 5000 是一个安全的上限，避免 `ConvergenceWarning`。

#### `random_state=RANDOM_STATE` — 随机种子

逻辑回归在某些求解器（如 `saga`、`liblinear`）中有随机性，固定种子确保可复现。

### 2.3 适用场景与优缺点

| 维度        | 评价                               |
| --------- | -------------------------------- |
| **优点**    | 可解释性强（系数直接反映特征重要性）、训练快、概率输出有校准意义 |
| **缺点**    | 只能学习线性关系、对异常值敏感、需要标准化            |
| **适用场景**  | 基线模型、可解释性要求高、线性关系足够的问题           |
| **本教程角色** | 线性基线参照，AUC=0.8936（最低之一）          |

***

## 三、模型 2：SVM（Linear，线性支持向量机）

```python
('SVM (Linear)',
 LinearSVC(class_weight='balanced', max_iter=5000, random_state=RANDOM_STATE,
           loss='hinge'),
 True),
```

### 3.1 模型原理

**支持向量机（SVM）** 的核心思想是：找到一条"边界最宽"的分隔线，让两类点到分隔线的间隔最大化。

**数学直觉**：

```
        │     ●●●●●  (MORTO)
        │   ●●●●●●●
─────────┼─────────────  ← 决策边界 w^T x + b = 0
        │   ○○○○○○○
        │     ○○○○○  (VIVO)
        │
        ↑ 支持向量到边界的距离（间隔，要最大化）
```

- **支持向量**：离决策边界最近的几个点，它们决定了边界的位置。
- **间隔**：支持向量到决策边界的距离，SVM 要最大化这个间隔。
- **决策函数**：$f(x) = w^T x + b$，正值预测正类，负值预测负类。

### 3.2 参数详解

#### `class_weight='balanced'` — 同 LR

与逻辑回归相同，自动调整类别权重，让模型更关注少数类（VIVO）。

#### `max_iter=5000` — 最大迭代次数

LinearSVC 用 liblinear 库求解，`max_iter` 是最大迭代次数。默认 1000，本教程设为 5000 以确保收敛。

#### `loss='hinge'` — 损失函数（重点）

> 💡 **重点概念：`loss='hinge'`** **的含义**
>
> LinearSVC 支持多种损失函数：
>
> - `loss='hinge'`：标准 SVM 的合页损失，$\max(0, 1 - y \cdot f(x))$。这是经典 SVM 的损失。
> - `loss='squared_hinge'`：合页损失的平方，$\max(0, 1 - y \cdot f(x))^2$。默认值。
>
> 本教程用 `loss='hinge'`，这是最标准的 SVM 损失。`squared_hinge` 对误分类的惩罚更重（平方），可能导致过拟合。
>
> **注意**：`loss='hinge'` 只支持 `dual=True`（对偶问题），不支持 `penalty='l1'`。本教程用默认的 `penalty='l2'` 和 `dual=True`。

#### `random_state=RANDOM_STATE` — 随机种子

LinearSVC 在某些情况下有随机性（如数据打乱），固定种子确保可复现。

### 3.3 为什么用 LinearSVC 而不是 SVC？

> 💡 **重点概念：LinearSVC vs SVC(kernel='linear')**
>
> - **`SVC(kernel='linear')`**：通用 SVM 实现，用 libsvm 库，时间复杂度 O(n²)–O(n³)，在 20,000 样本上非常慢。
> - **`LinearSVC`**：专门为线性核优化的实现，用 liblinear 库，时间复杂度近似 O(n)，能高效处理大样本。
>
> 本教程只比较线性核 SVM，不涉及 RBF 核（RBF 核在 20,000 样本上太慢）。实验结果：SVM 训练 0.18s（5 折），非常快。

### 3.4 适用场景与优缺点

| 维度        | 评价                                    |
| --------- | ------------------------------------- |
| **优点**    | 间隔最大化使泛化好、支持向量提供稀疏解、线性核速度快            |
| **缺点**    | 只能线性分割（本教程）、没有原生概率输出、对量纲敏感            |
| **适用场景**  | 中等样本量的线性分类、需要最大间隔保证的场景                |
| **本教程角色** | 线性模型对比，AUC=0.8946，Recall=0.8984（并列最高） |

***

## 四、模型 3：KNN（K 近邻）

```python
('KNN (k=15)',
 KNeighborsClassifier(n_neighbors=15, n_jobs=N_JOBS),
 True),
```

### 4.1 模型原理

**K 近邻（KNN）** 是最简单的非参数方法。它的核心思想是：看一个样本最近的 K 个邻居怎么投票，就把它分到哪一类。

**算法流程**：

1. 计算待预测样本到所有训练样本的距离（如欧氏距离）。
2. 找出距离最近的 K 个训练样本。
3. 这 K 个邻居中，哪一类多就预测哪一类（多数投票）。

**数学直觉**：

```
待预测样本 X:
        ●●●●●
        ●●●●●●●     ← 最近的 15 个邻居中：
        ●●X○○○○        ● (MORTO) 有 9 个
        ●●●○○○         ○ (VIVO) 有 6 个
        ●●●●●         → 多数投票：预测为 MORTO
```

### 4.2 参数详解

#### `n_neighbors=15` — 邻居数 K（重点）

> 💡💡💡 **重点概念：K 的选择与偏差-方差权衡**
>
> K 是 KNN 最重要的参数，控制偏差-方差权衡：
>
> - **K 太小（如 K=1）**：偏差低（每个点只看最近邻居）、方差高（对噪声敏感，过拟合）。
> - **K 太大（如 K=100）**：偏差高（决策边界过于平滑）、方差低（欠拟合）。
> - **K=15**：本教程的选择，平衡偏差与方差。
>
> 经验法则：K ≈ √n，其中 n 是样本量。本教程 n=16000（训练折），√16000 ≈ 126，但 15 已经足够好——因为本数据集信号强，不需要太大的 K。
>
> **为什么不用 class\_weight？** KNN 不支持 `class_weight` 参数，因为它是基于距离的方法，没有损失函数可以加权。这是 KNN 在不平衡数据上的劣势之一。

#### `n_jobs=N_JOBS` — 并行计算

KNN 的 `predict` 阶段需要计算待预测样本到所有训练样本的距离，可以并行化。`n_jobs=-1` 用所有 CPU 核。

### 4.3 适用场景与优缺点

| 维度        | 评价                                   |
| --------- | ------------------------------------ |
| **优点**    | 简单直观、无需训练（懒学习）、决策边界非线性               |
| **缺点**    | 预测慢（要计算所有距离）、对量纲敏感、不支持 class\_weight |
| **适用场景**  | 小到中等数据、需要非线性但不想训练复杂模型                |
| **本教程角色** | 非参数基线，AUC=0.9111，Recall=0.8335（最低）   |

### 4.4 KNN 在不平衡数据上的表现（重点发现）

> 💡 **重点概念：不平衡程度决定 KNN 是否失效**
>
> 实验结果显示，KNN 的 Recall=0.8335（七个模型中最低），但已远非"灾难"级别。原因：
>
> - **严重不平衡下**（如 1:20）：KNN 的近邻中几乎全是多数类，少数类样本被"淹没"，Recall 接近 0。
> - **轻度不平衡下**（本数据集 41:59）：KNN 的近邻中有足够的正类样本，仍能识别多数存活病例，Recall=0.8335。
>
> **不平衡程度是决定 KNN 是否失效的关键**——这是本教程的核心发现之一。

***

## 五、模型 4：Decision Tree（决策树）

```python
('Decision Tree',
 DecisionTreeClassifier(class_weight='balanced', max_depth=10,
                        random_state=RANDOM_STATE),
 False),
```

### 5.1 模型原理

**决策树**通过递归二分把特征空间划分成若干区域，每个区域对应一个预测。它的核心思想是：提一系列"是否"问题，逐步缩小范围。

**算法流程**：

1. 从根节点开始，选择一个特征和阈值，把数据分成两份。
2. 选择标准：最大化信息增益（或最小化基尼不纯度）。
3. 对每个子节点递归重复，直到停止条件（如深度限制、样本数过少）。

**数学直觉**：

```
                    Age < 60?
                   /         \
              是                否
              /                  \
     Diagnostic.means < 3?    Age < 75?
      /         \              /         \
   是            否          是            否
   /              \          /              \
预测: VIVO     预测: MORTO  预测: VIVO    预测: MORTO
```

### 5.2 参数详解

#### `class_weight='balanced'` — 同 LR/SVM

自动调整类别权重，让模型更关注少数类（VIVO）。

#### `max_depth=10` — 最大深度（重点）

> 💡 **重点概念：`max_depth`** **与过拟合**
>
> `max_depth` 控制树的最大深度，是最重要的正则化参数：
>
> - **深度太小（如 3）**：欠拟合，决策边界过于简单，AUC 低。
> - **深度太大（如 30）**：过拟合，树记住训练数据的噪声，泛化差。
> - **max\_depth=10**：本教程的选择，平衡拟合与泛化。
>
> 实验结果：DT 的 AUC=0.9149，σ=0.0018（较稳定）。教学文档指出："Decision Tree（σ=0.0018）反而较稳定。这与旧印象不同——在本数据集上单棵树的方差并不突出，可能因为深度限制（max\_depth=10）起到了正则化作用。"

#### `random_state=RANDOM_STATE` — 随机种子

决策树在某些分裂点选择上有随机性（当多个分裂点增益相同时），固定种子确保可复现。

### 5.3 为什么 `scale_needed=False`？

决策树的分裂基于特征的阈值（如 `Age < 60`），与特征的量纲无关。标准化不会改变分裂点（`Age < 60` 等价于 `StandardScaler(Age) < StandardScaler(60)`）。所以树模型不需要标准化。

### 5.4 适用场景与优缺点

| 维度        | 评价                                         |
| --------- | ------------------------------------------ |
| **优点**    | 可解释性强（可视化决策规则）、无需标准化、能处理非线性                |
| **缺点**    | 容易过拟合、方差高（小变化导致不同树）、不稳定                    |
| **适用场景**  | 探索性分析、需要可解释规则、作为集成的基学习器                    |
| **本教程角色** | 单棵树基线，AUC=0.9149，比集成树（RF/XGB/LGBM）低约 0.025 |

***

## 六、模型 5：Random Forest（随机森林）

```python
('Random Forest',
 RandomForestClassifier(n_estimators=200, max_depth=12,
                        class_weight='balanced',
                        random_state=RANDOM_STATE, n_jobs=N_JOBS),
 False),
```

### 6.1 模型原理

**随机森林**是 Bagging 集成的树模型。它的核心思想是：建一批决策树，让它们投票，降低单棵树的方差。

**算法流程**：

1. **Bootstrap 采样**：从训练集中有放回采样，生成 N 个子训练集。
2. **训练 N 棵树**：每个子训练集训练一棵决策树。每棵树分裂时，只随机选择部分特征（特征子空间）。
3. **投票**：预测时，N 棵树投票，多数票为最终预测。

**数学直觉**：

```
训练集 → Bootstrap 采样 → 200 个子训练集
                            │
                    ┌───────┼───────┐
                    ▼       ▼       ▼
                 树1     树2  ...  树200
                    │       │       │
                    └───────┼───────┘
                            ▼
                        多数投票
                            │
                            ▼
                        最终预测
```

### 6.2 参数详解

#### `n_estimators=200` — 树的数量（重点）

> 💡 **重点概念：`n_estimators`** **与偏差-方差权衡**
>
> `n_estimators` 是随机森林最重要的参数：
>
> - **树太少（如 10）**：方差高，性能不稳定。
> - **树太多（如 1000）**：性能提升边际递减，但训练时间和内存增加。
> - **n\_estimators=200**：本教程的选择，平衡性能和成本。
>
> 经验法则：树越多越好，但边际收益递减。200 棵树通常足够稳定。
>
> 实验结果：RF 的 AUC=0.9391，σ=0.0022，训练 1.83s（5 折），性价比极高。

#### `max_depth=12` — 最大深度

每棵树的最大深度。比单棵决策树的 `max_depth=10` 稍深，因为集成能降低方差，允许单棵树稍复杂。

#### `class_weight='balanced'` — 同前

自动调整类别权重。

#### `random_state=RANDOM_STATE` — 随机种子

固定 bootstrap 采样和特征子空间选择的随机性。

#### `n_jobs=N_JOBS` — 并行训练

200 棵树可以并行训练，`n_jobs=-1` 用所有 CPU 核，显著加速。

### 6.3 随机森林的"随机"在哪里？

> 💡 **重点概念：随机森林的两层随机性**
>
> 随机森林通过两层随机性降低方差：
>
> 1. **样本随机（Bootstrap）**：每棵树用有放回采样的子训练集，约 63% 的样本被选中（37% 是"袋外样本"）。
> 2. **特征随机（特征子空间）**：每次分裂时，只考虑随机选择的 √m 个特征（m 是总特征数）。本教程 m=9，√9=3，每次分裂只看 3 个特征。
>
> 这两层随机性让 200 棵树"足够不同"，投票时方差大幅降低。

### 6.4 适用场景与优缺点

| 维度        | 评价                                           |
| --------- | -------------------------------------------- |
| **优点**    | 性能强、方差低、无需标准化、能估计特征重要性                       |
| **缺点**    | 训练比单棵树慢、模型大（200 棵树）、可解释性不如单棵树                |
| **适用场景**  | 性价比之选、特征重要性分析、医学论文基线                         |
| **本教程角色** | 性价比之王，AUC=0.9391（第三），训练 1.83s（比 LGBM 快 35 倍） |

***

## 七、XGBoost 和 LightGBM 的容错导入

```python
# XGBoost & LightGBM
try:
    import xgboost as xgb
    models.append((
        'XGBoost',
        xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
            eval_metric='logloss', use_label_encoder=False
        ),
        True  # XGBoost 对特征尺度不敏感, 标准化不影响性能
    ))
except ImportError:
    print("  XGBoost not installed, skipping")

try:
    import lightgbm as lgb
    models.append((
        'LightGBM',
        lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1
        ),
        True  # LightGBM 也对尺度不敏感
    ))
except ImportError:
    print("  LightGBM not installed, skipping")
```

### 7.1 `try/except ImportError` — 容错设计

XGBoost 和 LightGBM 是第三方库，不在 sklearn 中，可能未安装。用 `try/except ImportError` 容错：

- 如果库已安装：`import` 成功，模型添加到 `models` 列表。
- 如果库未安装：`import` 失败，跳过该模型，打印提示。

这种设计让脚本在没有 XGBoost/LightGBM 的环境下也能运行（只比较 5 个模型），提高了鲁棒性。

### 7.2 打印模型数量

```python
print(f"\n  共 {len(models)} 个模型待评估")
```

如果 XGBoost 和 LightGBM 都安装，`len(models)` 是 7；如果都未安装，是 5。

***

## 八、模型 6：XGBoost

```python
xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0,
    eval_metric='logloss', use_label_encoder=False
)
```

### 8.1 模型原理

**XGBoost（eXtreme Gradient Boosting）** 是梯度提升的标杆实现。它的核心思想是：一棵一棵地建树，每棵树纠正上一棵的错误。

**算法流程**：

1. 初始化预测为常数（如对数几率）。
2. 计算残差（负梯度）。
3. 拟合一棵树到残差。
4. 把这棵树的预测加到总预测上（学习率控制步长）。
5. 重复步骤 2–4，直到 N 棵树。

**数学直觉**：

```
第1棵树: 拟合原始数据 → 预测 F1(x) = y1
第2棵树: 拟合残差 (y - F1) → 预测 F2(x) = F1(x) + lr * y2
第3棵树: 拟合新残差 (y - F2) → 预测 F3(x) = F2(x) + lr * y3
...
第N棵树: 每棵树都在纠正前面所有树的累积错误
```

### 8.2 参数详解

#### `n_estimators=300` — 树的数量

XGBoost 用 300 棵树，比随机森林的 200 多。因为梯度提升是串行的，每棵树贡献较小，需要更多树。

#### `max_depth=6` — 最大深度

每棵树的最大深度。比随机森林的 12 浅，因为梯度提升是"弱学习器"集成，每棵树应该是浅的。

> 💡 **重点概念：RF vs XGBoost 的深度差异**
>
> - **随机森林**：每棵树是"强学习器"，深度大（12），单独就能拟合好。集成降低方差。
> - **XGBoost**：每棵树是"弱学习器"，深度小（6），单独拟合不好。集成降低偏差。
>
> 这是 Bagging（RF）和 Boosting（XGBoost）的本质区别：Bagging 降方差，Boosting 降偏差。

#### `learning_rate=0.1` — 学习率（重点）

> 💡 **重点概念：`learning_rate`** **与** **`n_estimators`** **的权衡**
>
> `learning_rate`（也叫 `eta`）控制每棵树的贡献：
>
> $$F\_t(x) = F\_{t-1}(x) + \text{lr} \cdot h\_t(x)$$
>
> - **学习率小（如 0.01）**：每棵树贡献小，需要更多树（n\_estimators 大），但泛化好。
> - **学习率大（如 0.3）**：每棵树贡献大，需要更少树，但可能过拟合。
> - **learning\_rate=0.1**：本教程的选择，是经验上的"黄金值"。
>
> 经验法则：`n_estimators × learning_rate` 大致恒定。如 300×0.1 ≈ 100×0.3。

#### `scale_pos_weight=(y == 0).sum() / (y == 1).sum()` — 正类权重（重点）

> 💡💡💡 **重点概念：`scale_pos_weight`** **vs** **`class_weight='balanced'`**
>
> XGBoost 不用 `class_weight`，而用 `scale_pos_weight`：
>
> $$\text{scale\_pos\_weight} = \frac{n\_{negative}}{n\_{positive}} = \frac{n\_{MORTO}}{n\_{VIVO}} = \frac{11770}{8230} \approx 1.43$$
>
> 这告诉 XGBoost："正确预测一个存活病例的重要性 ≈ 正确预测 1.43 个死亡病例的权重。"
>
> **与** **`class_weight='balanced'`** **的区别**：
>
> - `class_weight='balanced'`：sklearn 的接口，自动计算每类权重。
> - `scale_pos_weight`：XGBoost 的接口，只调整正类权重（负类权重为 1）。
>
> 在轻度不平衡下（1.43），scale\_pos\_weight 的作用减弱——因为权重接近 1。教学文档指出："如果数据变为严重不平衡（如 1:20），scale\_pos\_weight 应如何调整？"——答案是 scale\_pos\_weight 应设为 20。

#### `random_state=RANDOM_STATE` — 随机种子

固定 XGBoost 内部的随机性（如列采样）。

#### `n_jobs=N_JOBS` — 并行训练

XGBoost 支持并行化树构建，`n_jobs=-1` 用所有 CPU 核。

#### `verbosity=0` — 静默模式

`verbosity=0` 关闭所有日志输出，让控制台更干净。

#### `eval_metric='logloss'` — 评估指标

指定评估指标为对数损失（logloss），用于内部早停（本教程未用早停，但指定 metric 是好习惯）。

#### `use_label_encoder=False` — 禁用旧版标签编码

XGBoost 1.3+ 不再需要旧版标签编码器，设为 `False` 避免警告。

### 8.3 适用场景与优缺点

| 维度        | 评价                            |
| --------- | ----------------------------- |
| **优点**    | 性能强（梯度提升）、正则化防过拟合、处理缺失值、特征重要性 |
| **缺点**    | 训练慢（串行）、参数多、可能过拟合             |
| **适用场景**  | 表格数据竞赛、性能优先、中等以上样本量           |
| **本教程角色** | 性能亚军，AUC=0.9415，训练 22.79s     |

***

## 九、模型 7：LightGBM

```python
lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1
)
```

### 9.1 模型原理

**LightGBM** 是微软开源的高效梯度提升框架。它的核心思想是：和 XGBoost 一样，但用 GOSS 和 EFB 优化让它更快。

**与 XGBoost 的关键差异**：

| 特性   | XGBoost          | LightGBM        |
| ---- | ---------------- | --------------- |
| 分裂策略 | Level-wise（逐层分裂） | Leaf-wise（逐叶分裂） |
| 采样策略 | 无特殊采样            | GOSS（梯度单边采样）    |
| 特征处理 | 预排序              | EFB（互斥特征捆绑）     |
| 速度   | 基线               | 快 2-10 倍        |
| 内存   | 中等               | 更低              |
| 小样本  | 更好               | 可能过拟合           |

### 9.2 参数详解

#### `n_estimators=300`、`max_depth=6`、`learning_rate=0.1` — 与 XGBoost 相同

为了公平对比，LightGBM 用和 XGBoost 相同的核心参数。

#### `class_weight='balanced'` — 用 sklearn 风格（与 XGBoost 不同）

> 💡 **重点概念：LightGBM 用** **`class_weight`** **而非** **`scale_pos_weight`**
>
> LightGBM 的 sklearn 接口（`LGBMClassifier`）支持 `class_weight='balanced'`，与 sklearn 的其他模型一致。这是 LightGBM 比 XGBoost 更"sklearn 友好"的一点。
>
> 本教程中 LightGBM 用 `class_weight='balanced'`，XGBoost 用 `scale_pos_weight=1.43`，两者效果类似——都是让模型更关注少数类。

#### `random_state=RANDOM_STATE` — 随机种子

#### `n_jobs=N_JOBS` — 并行训练

#### `verbosity=-1` — 静默模式

LightGBM 的 `verbosity` 取值不同：`-1` 是静默，`0` 是警告，`1` 是信息。

### 9.3 Leaf-wise vs Level-wise（重点）

> 💡💡💡 **重点概念：LightGBM 为什么更快？**
>
> **Level-wise（XGBoost 默认）**：逐层分裂，同一层的所有节点都分裂后再进入下一层。树是"平衡"的。
>
> **Leaf-wise（LightGBM 默认）**：每次选择增益最大的叶节点分裂，树是"不平衡"的。
>
> ```
> Level-wise (XGBoost):          Leaf-wise (LightGBM):
>       ●                              ●
>      / \                            / \
>     ●   ●                          ●   ●
>    / \ / \                        / \
>   ●  ●●  ●                       ●   ●
>  (平衡，4 层)                    / \
>                                 ●   ●
>                                (不平衡，5 层，但增益更高)
> ```
>
> Leaf-wise 收敛更快（每分裂都选最优），但可能过拟合（树太深）。LightGBM 用 `max_depth` 限制深度来防止过拟合。

### 9.4 GOSS 和 EFB 优化

> 💡 **重点概念：LightGBM 的两大速度优化**
>
> **GOSS（Gradient-based One-Side Sampling，梯度单边采样）**：
>
> - 保留梯度大的样本（信息量大），随机采样梯度小的样本（信息量小）。
> - 减少训练样本数，加速分裂。
>
> **EFB（Exclusive Feature Bundling，互斥特征捆绑）**：
>
> - 把"互斥"的特征（很少同时非零）捆绑成一个特征。
> - 减少特征数，加速分裂。
> - 对稀疏特征（如 one-hot 编码）特别有效。
>
> 这两个优化让 LightGBM 在大数据上比 XGBoost 快 2-10 倍。但在本教程的小数据（20,000 样本）上，LightGBM 反而比 XGBoost 慢（64.64s vs 22.79s）——因为小数据上优化的开销超过了收益。

### 9.5 适用场景与优缺点

| 维度        | 评价                                   |
| --------- | ------------------------------------ |
| **优点**    | 大数据上极快、性能强、内存低、支持类别特征                |
| **缺点**    | 小数据可能过拟合、参数多、leaf-wise 可能过拟合         |
| **适用场景**  | 大数据竞赛、性能优先、百万级样本                     |
| **本教程角色** | 性能冠军，AUC=0.9423（最高），Brier=0.0974（最低） |

***

## 十、七模型对比总览

| 模型                  | 类型   | 关键参数                                        | scale\_needed | class\_weight 处理          |
| ------------------- | ---- | ------------------------------------------- | ------------- | ------------------------- |
| Logistic Regression | 线性   | `max_iter=5000`                             | True          | `class_weight='balanced'` |
| SVM (Linear)        | 线性   | `loss='hinge'`, `max_iter=5000`             | True          | `class_weight='balanced'` |
| KNN (k=15)          | 非参数  | `n_neighbors=15`                            | True          | 不支持                       |
| Decision Tree       | 树    | `max_depth=10`                              | False         | `class_weight='balanced'` |
| Random Forest       | 集成树  | `n_estimators=200`, `max_depth=12`          | False         | `class_weight='balanced'` |
| XGBoost             | 梯度提升 | `n_estimators=300`, `max_depth=6`, `lr=0.1` | True          | `scale_pos_weight=1.43`   |
| LightGBM            | 梯度提升 | `n_estimators=300`, `max_depth=6`, `lr=0.1` | True          | `class_weight='balanced'` |

***

## 小贴士

1. **`class_weight='balanced'`** **自动调整权重**：少数类权重 = n\_samples / (n\_classes × n\_counts)，让模型更关注少数类。
2. **KNN 不支持 class\_weight**：因为它是基于距离的方法，没有损失函数可以加权。
3. **XGBoost 用** **`scale_pos_weight`** **而非** **`class_weight`**：scale\_pos\_weight = n\_negative / n\_positive。
4. **LightGBM 用** **`class_weight='balanced'`**：与 sklearn 接口一致，更友好。
5. **`max_depth`** **控制树复杂度**：RF 用 12（强学习器），XGBoost/LightGBM 用 6（弱学习器）。
6. **`n_estimators`** **控制集成大小**：RF 用 200，XGBoost/LightGBM 用 300。
7. **`learning_rate`** **控制每棵树贡献**：0.1 是经验"黄金值"，与 n\_estimators 反比。
8. **LinearSVC 比 SVC(kernel='linear') 快得多**：大样本上用 LinearSVC。
9. **`try/except ImportError`** **容错**：XGBoost/LightGBM 是第三方库，可能未安装。
10. **RF 是 Bagging（降方差），XGBoost/LightGBM 是 Boosting（降偏差）**：这是集成的两大流派。

***

## 常见问题

**Q1：为什么 LR 和 SVM 的** **`max_iter=5000`，而默认是 100/1000？**

A：本教程有 9 个特征，量纲差异大。即使有 StandardScaler，某些情况下（如特征高度相关）也可能需要更多迭代。5000 是安全上限，避免 `ConvergenceWarning`。如果收敛快，实际迭代次数会远小于 5000。

**Q2：KNN 的** **`n_neighbors=15`** **怎么选的？**

A：经验法则 K ≈ √n，本教程 n=16000，√16000 ≈ 126。但 15 已经足够好——因为本数据集信号强，不需要太大的 K。K 太大会欠拟合（决策边界过于平滑），K 太小会过拟合（对噪声敏感）。可以用交叉验证选最优 K，但本教程固定 15 保持对比公平。

**Q3：为什么 RF 的** **`max_depth=12`，而 XGBoost/LightGBM 的** **`max_depth=6`？**

A：因为 RF 是 Bagging（降方差），每棵树是"强学习器"，单独就能拟合好，所以深度大。XGBoost/LightGBM 是 Boosting（降偏差），每棵树是"弱学习器"，单独拟合不好，所以深度小。这是两种集成流派的本质区别。

**Q4：`scale_pos_weight=1.43`** **是怎么算的？**

A：`scale_pos_weight = n_negative / n_positive = n_MORTO / n_VIVO = 11770 / 8230 ≈ 1.43`。这告诉 XGBoost："正确预测一个存活病例的重要性 ≈ 1.43 个死亡病例。"在轻度不平衡下（1.43），权重接近 1，作用减弱。

**Q5：为什么 XGBoost 用** **`scale_pos_weight`，LightGBM 用** **`class_weight='balanced'`？**

A：这是两个库的接口差异。XGBoost 的原生接口用 `scale_pos_weight`（只调整正类权重），LightGBM 的 sklearn 接口支持 `class_weight='balanced'`（自动计算每类权重）。两者效果类似，都是让模型更关注少数类。

**Q6：`loss='hinge'`** **和默认的** **`loss='squared_hinge'`** **有什么区别？**

A：`hinge` 是标准 SVM 的合页损失 $\max(0, 1 - y \cdot f(x))$，`squared_hinge` 是其平方 $\max(0, 1 - y \cdot f(x))^2$。平方版本对误分类惩罚更重，可能导致过拟合。本教程用 `hinge` 是最标准的 SVM 损失。

**Q7：XGBoost 和 LightGBM 的** **`scale_needed=True`，但它们对尺度不敏感，为什么还标准化？**

A：代码注释说明了："XGBoost 对特征尺度不敏感, 标准化不影响性能"。设为 True 是为了统一流程，标准化不会影响它们的性能。实际上设为 False 也行，但统一流程更简洁。

**Q8：为什么 LightGBM 在本教程比 XGBoost 慢（64.64s vs 22.79s）？**

A：LightGBM 的 GOSS 和 EFB 优化在大数据上才有效。本教程只有 20,000 样本，优化的开销超过了收益。在百万级样本上，LightGBM 会比 XGBoost 快 2-10 倍。这是"小数据上 XGBoost 更好，大数据上 LightGBM 更好"的体现。

**Q9：`use_label_encoder=False`** **是什么意思？**

A：XGBoost 1.3+ 不再需要旧版标签编码器（`LabelEncoder`）。设为 `False` 避免警告。如果用 `True`，XGBoost 会内部对标签做编码，可能产生 deprecation 警告。

**Q10：为什么** **`verbosity=0`（XGBoost）和** **`verbosity=-1`（LightGBM）？**

A：两个库的 verbosity 取值不同。XGBoost：`0` 是静默，`1` 是警告，`2` 是信息，`3` 是调试。LightGBM：`-1` 是静默，`0` 是警告，`1` 是信息。设为静默让控制台更干净，只保留本教程自己的 print 输出。

***

## 本模块小结

### 核心知识点回顾

1. **7 个模型的数学原理**：
   - LR：线性决策边界 + Sigmoid
   - SVM：最大间隔分隔
   - KNN：距离投票
   - DT：递归二分
   - RF：Bagging 集成（降方差）
   - XGBoost：梯度提升（降偏差）+ 正则化
   - LightGBM：梯度提升 + GOSS + EFB（更快）
2. **关键参数**：
   - `class_weight='balanced'`：LR/SVM/DT/RF/LightGBM 用，自动调整类别权重
   - `scale_pos_weight=1.43`：XGBoost 用，= n\_negative / n\_positive
   - `n_estimators`：RF=200，XGBoost/LightGBM=300
   - `max_depth`：DT=10，RF=12，XGBoost/LightGBM=6
   - `learning_rate=0.1`：XGBoost/LightGBM 的"黄金值"
   - `n_neighbors=15`：KNN 的邻居数
3. **不平衡数据处理**：
   - `class_weight='balanced'`：sklearn 接口，自动计算每类权重
   - `scale_pos_weight`：XGBoost 接口，只调整正类权重
   - KNN 不支持任何权重调整（基于距离）
4. **Bagging vs Boosting**：
   - RF（Bagging）：每棵树是强学习器（深度大），集成降方差
   - XGBoost/LightGBM（Boosting）：每棵树是弱学习器（深度小），集成降偏差
5. **LinearSVC vs SVC**：
   - LinearSVC 用 liblinear，O(n)，大样本快
   - SVC(kernel='linear') 用 libsvm，O(n²)–O(n³)，大样本慢
6. **LightGBM 的速度优化**：
   - Leaf-wise 分裂（vs Level-wise）
   - GOSS（梯度单边采样）
   - EFB（互斥特征捆绑）
   - 大数据上快 2-10 倍，小数据上可能更慢
   <br />

### 七模型参数对比表

| 模型       | n\_estimators | max\_depth | learning\_rate | class\_weight           | scale\_needed |
| -------- | ------------- | ---------- | -------------- | ----------------------- | ------------- |
| LR       | —             | —          | —              | balanced                | True          |
| SVM      | —             | —          | —              | balanced                | True          |
| KNN      | —             | —          | —              | 不支持                     | True          |
| DT       | —             | 10         | —              | balanced                | False         |
| RF       | 200           | 12         | —              | balanced                | False         |
| XGBoost  | 300           | 6          | 0.1            | scale\_pos\_weight=1.43 | True          |
| LightGBM | 300           | 6          | 0.1            | balanced                | True          |

### 下一部分预告

本模块完成了 7 个模型的定义，得到了 `models` 列表（每个元素是 `(name, model, scale_needed)` 三元组）。接下来的模块 3 将运行训练和评估：

**模块 3：训练评估与结果分析**

- 运行 7 模型 × 5 折 = 35 次训练
- 分析 AUC、Recall、Brier、Time、σ 五个维度
- 解读 LightGBM 性能冠军、KNN Recall 最低等核心发现
- 讨论稳定性与性能的二维分析

让我们带着"7 个模型在同一数据上表现如何"的疑问，进入模块 3 的训练评估！

***

