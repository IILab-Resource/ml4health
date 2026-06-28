# 模块 1：DROrthoForest 训练与 CATE 估计

> 本模块是案例教程 18「异质性处理效应 (HTE) — 双重机器学习 (DML) 方法」的**第二个模块**。在模块 0 完成数据准备（Y/T/X/W 和 X_test）之后，本模块开始训练第一个 DML 模型——**DROrthoForest（Double Robust Orthogonal Random Forest，双重稳健正交随机森林）**。我们将计算正则化参数 `lambda_reg`、配置模型的 8 个关键参数、训练模型、估计 CATE（条件平均处理效应）及其 95% 置信区间。 
>
> 本模块最核心的知识点有三个：**一是 DROrthoForest 的双重稳健性原理**——为什么"结果模型或倾向模型至少一个正确"就能保证估计一致；**二是正则化参数 `lambda_reg` 的计算公式**——`np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))` 的每个组成部分如何影响正则化强度；**三是 `est.fit(Y, T, X=X, W=W)` 的参数顺序和 `est.effect(X_test)` / `est.effect_interval(X_test, alpha=0.05)` 的用法**——如何估计 CATE 点值和置信区间。

---

## 学习目标

学完本模块后，你将能够：

1. **理解 DROrthoForest 的双重稳健性原理**：知道"双重稳健"的含义是"结果模型或倾向模型至少一个正确即可"，以及为什么这在实践中很重要。
2. **掌握正则化参数 `lambda_reg` 的计算公式**：理解 `np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))` 中每个变量的含义和作用。
3. **理解 DROrthoForest 的 8 个关键参数**：`n_trees`、`min_leaf_size`、`max_depth`、`subsample_ratio`、`propensity_model`、`model_Y`、`propensity_model_final`、`model_Y_final` 各自的作用。
4. **理解为什么用 LogisticRegression (L1) 作为 propensity_model**：知道倾向得分模型预测 P(T=1|X, W)，L1 正则化防止过拟合。
5. **理解为什么用 Lasso 作为 model_Y**：知道结果模型预测 E[Y|X, W]，Lasso 自动做特征选择。
6. **理解 WeightedLasso 的作用**：知道 `model_Y_final` 在叶子节点内根据样本权重做加权估计。
7. **掌握 `est.fit(Y, T, X=X, W=W)` 的参数顺序**：知道 Y、T、X、W 的传入顺序和关键字参数的用法。
8. **掌握 `est.effect(X_test)` 和 `est.effect_interval(X_test, alpha=0.05)` 的用法**：知道如何估计 CATE 点值和 95% 置信区间，以及异常处理。

---

## 一、DROrthoForest 的双重稳健性原理（核心概念）

在进入代码之前，我们先理解 DROrthoForest 的核心原理。这是本模块最重要的概念。

### 1.1 什么是"双重稳健"？

> 💡 **重点概念：双重稳健性 (Double Robustness)**
>
> "双重稳健"的含义是：**只要结果模型 E[Y|X, W] 或倾向模型 P(T=1|X, W) 中至少一个正确设定，因果效应估计就是一致的（渐近无偏）**。
>
> 数学表述：
> - 如果 E[Y|X, W] 正确，或 P(T=1|X, W) 正确，或两者都正确 → 估计一致 ✅
> - 如果两者都错误 → 估计可能有偏 ❌
>
> 这在实践中非常重要，因为我们很少能确保模型完全正确。双重稳健性给了我们"两次机会"——即使一个模型设定错误，只要另一个正确，估计仍然可靠。

### 1.2 什么是"正交"？

"正交"（Orthogonal）的含义是：把"效应估计"与"讨厌参数估计"分离，让效应估计对模型误差不敏感。

具体来说，DROrthoForest 用 **Neyman 正交化**：
1. 先用 W 预测 Y，得到残差 Y_res = Y - E[Y|X, W]
2. 先用 W 预测 T，得到残差 T_res = T - P(T=1|X, W)
3. 在叶子节点内对 Y_res ~ T_res 做回归，斜率就是 CATE

这种正交化让 CATE 估计对 E[Y|X, W] 和 P(T=1|X, W) 的误差不敏感——即使模型略有偏差，CATE 估计仍然稳定。

### 1.3 DROrthoForest 的工作流程

```
DROrthoForest 工作流程:

1. Bootstrap 采样: 从训练集有放回抽取 subsample_ratio 比例的样本
2. 对每个 Bootstrap 样本训练一棵"正交随机森林"树:
   a. 用 W 预测 Y, 得到 Y_res
   b. 用 W 预测 T, 得到 T_res
   c. 在每个叶子节点内, 对 Y_res ~ T_res 做加权 Lasso 回归
   d. 斜率 = 该叶子节点的 CATE 估计
3. 聚合所有树的结果: 对同一 X_test 点, 取所有树 CATE 的加权平均
```

---
 
---

## 二、计算正则化参数 lambda_reg（本模块核心）

```python
# 计算正则化参数 lambda
subsample_ratio = 0.3
n_features_W = W.shape[1]
n_samples = W.shape[0]
lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))
print(f"    正则化参数 lambda_reg = {lambda_reg:.6f}")
```

这是本模块**最重要的数学公式**。我们逐行解释。

### 2.1 `subsample_ratio = 0.3`

**子采样比例**。每棵树从训练集中有放回抽取 30% 的样本。这个值影响：

- **方差**：subsample_ratio 越小，树之间的差异越大，聚合后方差越小（类似随机森林的 max_samples）。
- **偏差**：subsample_ratio 越小，每棵树用的数据越少，单棵树的偏差越大。
- **0.3 是经验值**：在偏差和方差之间取得平衡。

### 2.2 `n_features_W = W.shape[1]`

W 的特征数。本教程中 W 有 5 列（Gender, Raca.Color, year, Laterality, Diagnostic.means），所以 `n_features_W = 5`。

- `W.shape`：返回 `(行数, 列数)` 元组。
- `W.shape[1]`：取列数（5）。

### 2.3 `n_samples = W.shape[0]`

样本数。本教程中 `n_samples = 3000`。

- `W.shape[0]`：取行数（3000）。

### 2.4 `lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))`

**正则化参数 lambda 的计算公式**。这是 econml 文档推荐的经验公式，用于 Lasso 和 LogisticRegression 的正则化强度。

我们拆解这个公式：

```
lambda_reg = sqrt( log(n_features_W) / (10 * subsample_ratio * n_samples) )
           = sqrt( log(5) / (10 * 0.3 * 3000) )
           = sqrt( 1.609 / 9000 )
           = sqrt( 0.000179 )
           = 0.01338
```

#### 公式各部分的含义：

| 部分 | 含义 | 本教程值 |
|------|------|---------|
| `np.log(n_features_W)` | W 特征数的对数，随特征数增加缓慢增长 | log(5) ≈ 1.609 |
| `10` | 经验常数，放大分母，让 lambda 更小 | 10 |
| `subsample_ratio` | 子采样比例 | 0.3 |
| `n_samples` | 样本数 | 3000 |
| `np.sqrt(...)` | 开根号，让 lambda 与样本数的平方根成反比 | — |

#### 为什么是这个公式？

> 💡 **重点概念：lambda_reg 的设计逻辑**
>
> 1. **与样本数成反比**：样本越多，lambda 越小，正则化越弱。因为大样本下模型不容易过拟合，不需要强正则化。
> 2. **与特征数的对数成正比**：特征越多，lambda 略大，正则化略强。因为高维更容易过拟合。
> 3. **与 subsample_ratio 成反比**：子采样比例越小，每棵树用的数据越少，需要更强的正则化防止过拟合。
> 4. **开根号**：让 lambda 与 1/sqrt(n) 成正比，这是统计学中正则化参数的经典缩放（与 Lasso 的理论保证一致）。
>
> 本教程的 lambda_reg ≈ 0.0134，是一个较小的值，表示轻度正则化。

---

## 三、配置 DROrthoForest 模型（本模块核心）

```python
print("\n    开始训练 DROrthoForest...")
t0 = time.time()

est = DROrthoForest(
    n_trees=100,
    min_leaf_size=10,
    max_depth=20,
    subsample_ratio=subsample_ratio,
    # 倾向得分模型: 带 L1 惩罚的逻辑回归
    propensity_model=LogisticRegression(
        C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga', max_iter=1000
    ),
    # 结果模型: Lasso 回归
    model_Y=Lasso(alpha=lambda_reg, max_iter=1000),
    # DML 最后阶段的模型
    propensity_model_final=LogisticRegression(
        C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga', max_iter=1000
    ),
    model_Y_final=WeightedLasso(alpha=lambda_reg),
    n_jobs=1,
    random_state=RANDOM_STATE
)
```

### 3.1 `t0 = time.time()`

记录训练开始时间。训练结束后用 `time.time() - t0` 计算耗时。DROrthoForest 训练较慢，记录耗时有助于评估计算成本。

### 3.2 DROrthoForest 的参数详解

#### `n_trees=100`

**树的数量**。DROrthoForest 是一个集成模型，包含 100 棵正交随机森林树。

- 树越多，估计越稳定（方差越小），但训练越慢。
- 100 是一个合理的默认值，类似随机森林的 `n_estimators=100`。
- 如果 CATE 曲线波动太大，可以增加到 200 或 500。

#### `min_leaf_size=10`

**叶子节点的最小样本数**。每个叶子节点至少包含 10 个样本。

- 叶子节点太小（如 1）会导致 CATE 估计方差大（过拟合）。
- 叶子节点太大（如 100）会导致 CATE 估计偏差大（欠拟合，无法捕捉局部异质性）。
- 10 是经验值，在偏差和方差之间取得平衡。

#### `max_depth=20`

**树的最大深度**。控制树的复杂度。

- 深度越大，树越复杂，能捕捉更细的异质性，但容易过拟合。
- 深度越小，树越简单，CATE 估计更平滑，但可能欠拟合。
- 20 足够深，能捕捉 Age 的非线性效应。

#### `subsample_ratio=subsample_ratio`

**子采样比例**（0.3）。每棵树从训练集中有放回抽取 30% 的样本（即 900 个样本）。

- 这个值在前面已经定义（0.3），这里直接引用。
- 子采样增加树之间的多样性，降低聚合后的方差。

### 3.3 propensity_model（倾向得分模型）

```python
propensity_model=LogisticRegression(
    C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga', max_iter=1000
),
```

**倾向得分模型**：预测 P(T=1|X, W)，即给定特征下患者转移的概率。

> 💡 **重点概念：为什么用 LogisticRegression (L1) 作为 propensity_model？**
>
> 1. **二分类任务**：T 是二值的（0 或 1），LogisticRegression 是二分类的标准模型。
> 2. **L1 正则化**：`penalty='l1'` 让模型自动做特征选择，对高维 W 鲁棒。不重要的 W 变量系数会被压到 0。
> 3. **防止过拟合**：DML 的理论保证要求 nuisance model 不过拟合，L1 正则化是经典选择。
> 4. **计算高效**：相比随机森林，LogisticRegression 训练快，适合在交叉拟合中多次调用。

#### LogisticRegression 的参数详解：

##### `C=1 / (X.shape[0] * lambda_reg)`

**正则化强度的倒数**。C 越小，正则化越强。

- `X.shape[0]`：样本数（3000）。
- `lambda_reg`：前面计算的正则化参数（≈0.0134）。
- `C = 1 / (3000 * 0.0134) = 1 / 40.2 ≈ 0.0249`。

> 💡 **重点概念：C 和 alpha 的关系**
>
> sklearn 的 LogisticRegression 用 `C`（正则化强度的倒数），Lasso 用 `alpha`（正则化强度本身）。两者关系：
> - LogisticRegression: `C = 1 / (n_samples * lambda)`
> - Lasso: `alpha = lambda`
>
> 这种差异源于不同的优化目标公式。econml 文档建议用 `C = 1 / (n_samples * lambda_reg)` 让 LogisticRegression 和 Lasso 的正则化强度"对齐"。

##### `penalty='l1'`

**惩罚类型**：L1 惩罚（Lasso）。让不重要的特征系数为 0，实现特征选择。

- L1 惩罚：`loss + alpha * Σ|β|`
- L2 惩罚：`loss + alpha * Σβ²`
- L1 能产生稀疏解（部分系数为 0），L2 不能。

##### `solver='saga'`

**优化算法**。`saga` 是支持 L1 惩罚的随机梯度下降变体。

- `saga` 适合大样本 + L1 惩罚的场景。
- 其他选项：`liblinear`（小数据集）、`lbfgs`（不支持 L1）、`newton-cg`（不支持 L1）。
- 本教程用 `saga` 因为它支持 L1 且适合中等规模数据。

##### `max_iter=1000`

**最大迭代次数**。saga 算法的最大迭代次数。

- 默认是 100，但 L1 惩罚收敛较慢，增加到 1000 确保收敛。
- 如果出现 `ConvergenceWarning`，可以增加到 2000 或 5000。

### 3.4 model_Y（结果模型）

```python
model_Y=Lasso(alpha=lambda_reg, max_iter=1000),
```

**结果模型**：预测 E[Y|X, W]，即给定特征下患者的存活概率（用回归预测）。

> 💡 **重点概念：为什么用 Lasso 作为 model_Y？**
>
> 1. **回归任务**：Y 是 0/1 二值，但 DML 内部用回归（而非分类）预测 E[Y|X, W]。这是因为 DML 的理论基于回归残差。
> 2. **L1 正则化**：`alpha=lambda_reg` 让模型自动做特征选择，对高维 W 鲁棒。
> 3. **线性模型**：Lasso 是线性模型，假设 E[Y|X, W] 是 W 的线性函数。如果关系非线性，可以用 `RandomForestRegressor` 等替代，但 Lasso 是 DML 的经典选择。
> 4. **计算高效**：Lasso 训练快，适合在交叉拟合中多次调用。

#### Lasso 的参数详解：

##### `alpha=lambda_reg`

**正则化强度**。alpha 越大，正则化越强。

- `alpha = lambda_reg ≈ 0.0134`，是较小的值，表示轻度正则化。
- Lasso 的损失函数：`Σ(y - ŷ)² + alpha * Σ|β|`。

##### `max_iter=1000`

**最大迭代次数**。Lasso 用坐标下降法，1000 次迭代足够收敛。

### 3.5 propensity_model_final 和 model_Y_final（DML 最后阶段的模型）

```python
propensity_model_final=LogisticRegression(
    C=1 / (X.shape[0] * lambda_reg), penalty='l1', solver='saga', max_iter=1000
),
model_Y_final=WeightedLasso(alpha=lambda_reg),
```

**DML 最后阶段的模型**：在叶子节点内做最终的 CATE 估计。

> 💡 **重点概念：propensity_model vs propensity_model_final 的区别**
>
> - **`propensity_model` / `model_Y`**：用于"第一层" nuisance model 估计——在整个 Bootstrap 样本上训练，预测 P(T=1|X, W) 和 E[Y|X, W]，计算残差。
> - **`propensity_model_final` / `model_Y_final`**：用于"第二层"最终估计——在叶子节点内，用残差做最终的 CATE 估计。
>
> 为什么要分两层？因为第一层用全部数据训练 nuisance model，第二层在局部（叶子节点）做加权估计。两层的模型可以不同，但本教程用相同的配置（除了 model_Y_final 用 WeightedLasso）。

#### `model_Y_final=WeightedLasso(alpha=lambda_reg)`

**WeightedLasso**：econml 提供的加权 Lasso，支持样本权重。

- 在叶子节点内，不同样本的权重不同（离叶子中心越近，权重越大）。
- 普通 Lasso 不支持样本权重，所以用 WeightedLasso。
- `alpha=lambda_reg`：与 model_Y 相同的正则化强度。

> 💡 **小贴士：WeightedLasso 来自哪里？**
>
> `WeightedLasso` 来自 `econml.sklearn_extensions.linear_model`，是 econml 对 sklearn Lasso 的扩展，增加了 `sample_weight` 支持。sklearn 的 Lasso 虽然在 `fit` 方法中支持 `sample_weight`，但 econml 的 WeightedLasso 在内部更深度地集成了权重，适合 DML 的加权回归场景。

### 3.6 `n_jobs=1` 和 `random_state=RANDOM_STATE`

#### `n_jobs=1`

**并行数**。1 表示单线程，不并行。

- 设为 1 是为了可复现性——多线程可能导致结果不可复现。
- 如果想加速，可以设为 -1（用所有 CPU），但结果可能略有差异。

#### `random_state=RANDOM_STATE`

**随机种子**。控制 DROrthoForest 内部的所有随机性（Bootstrap 采样、树的生长等）。

- 设为 42 保证可复现。

---

## 四、训练模型

```python
est.fit(Y, T, X=X, W=W)
print(f"    训练完成, 耗时: {time.time() - t0:.1f}s")
```

### 4.1 `est.fit(Y, T, X=X, W=W)`

**训练 DROrthoForest 模型**。这是本模块的核心步骤。

#### 参数顺序：

| 参数 | 含义 | 形状 |
|------|------|------|
| `Y` | 结果变量 | (3000,) |
| `T` | 处理变量 | (3000,) |
| `X` | 异质性特征（关键字参数） | (3000, 1) |
| `W` | 控制变量（关键字参数） | (3000, 5) |

> ⚠️ **注意：Y 和 T 是位置参数，X 和 W 是关键字参数**
>
> `est.fit(Y, T, X=X, W=W)` 中，Y 和 T 是位置参数（按顺序传入），X 和 W 是关键字参数（用 `X=` 和 `W=` 显式指定）。这种设计是为了与 sklearn 的 `fit(X, y)` 接口区分——DML 有四个变量，不能用单纯的 `fit(X, y)`。
>
> 如果你写成 `est.fit(Y, T, X, W)`（X 和 W 也用位置参数），会报错，因为 econml 要求 X 和 W 用关键字参数。

#### 训练过程：

```
est.fit(Y, T, X=X, W=W) 内部执行:

1. 对每棵树 (n_trees=100):
   a. Bootstrap 采样: 从 (Y, T, X, W) 有放回抽取 subsample_ratio=0.3 比例的样本
   b. 用 Bootstrap 样本训练 propensity_model: P(T=1|X, W)
   c. 用 Bootstrap 样本训练 model_Y: E[Y|X, W]
   d. 计算残差: Y_res = Y - E[Y|X, W], T_res = T - P(T=1|X, W)
   e. 用残差和 X 生长一棵正交随机森林树:
      - 分裂准则: 最大化 CATE 的异质性
      - 每个叶子节点内, 用 model_Y_final (WeightedLasso) 做加权回归
      - 斜率 = 该叶子节点的 CATE 估计
2. 聚合所有树: 对同一 X_test 点, 取所有树 CATE 的加权平均
```
 

---

## 五、估计 CATE（条件平均处理效应）

```python
# 估计 CATE
treatment_effects = est.effect(X_test)
try:
    te_lower, te_upper = est.effect_interval(X_test, alpha=0.05)
except Exception:
    # 某些版本的 DROrthoForest 可能不支持 effect_interval
    print("    (effect_interval 不可用, 跳过置信区间估计)")
    te_lower = treatment_effects - 1.96 * np.std(treatment_effects)
    te_upper = treatment_effects + 1.96 * np.std(treatment_effects)
print("    CATE 估计完成")
```

### 5.1 `treatment_effects = est.effect(X_test)`

**估计 CATE 点值**。对 X_test 中的每个 Age 值，估计条件平均处理效应 CATE(Age)。

- `X_test`：形状 (50, 1)，50 个 Age 测试点。
- 返回值 `treatment_effects`：形状 (50,)，每个 Age 点的 CATE 估计。

> 💡 **重点概念：CATE 的含义**
>
> CATE(x) = E[Y(1) - Y(0) | X=x]
>
> 含义：在 Age=x 的患者中，转移（T=1）相比局部（T=0）对存活概率的平均因果效应。
>
> - CATE = -0.3：转移使存活概率降低 30%
> - CATE = 0：转移对存活无影响
> - CATE = +0.1：转移使存活概率增加 10%（不太可能，但理论上可能）
>
> 本教程预期 CATE 为负（转移降低存活概率），且随 Age 变化（异质性）。

### 5.2 `est.effect_interval(X_test, alpha=0.05)` — 置信区间

```python
try:
    te_lower, te_upper = est.effect_interval(X_test, alpha=0.05)
except Exception:
    print("    (effect_interval 不可用, 跳过置信区间估计)")
    te_lower = treatment_effects - 1.96 * np.std(treatment_effects)
    te_upper = treatment_effects + 1.96 * np.std(treatment_effects)
```

**估计 95% 置信区间**。对每个 Age 点，计算 CATE 的 95% 置信区间 [te_lower, te_upper]。

#### `alpha=0.05`

**显著性水平**。alpha=0.05 表示 95% 置信区间。

- alpha=0.05 → 95% CI
- alpha=0.01 → 99% CI
- alpha=0.10 → 90% CI

#### 返回值

`effect_interval` 返回两个数组：
- `te_lower`：置信区间下界，形状 (50,)。
- `te_upper`：置信区间上界，形状 (50,)。

#### DROrthoForest 的置信区间方法

> 💡 **重点概念：BLB (Bag of Little Bootstraps)**
>
> DROrthoForest 用 **BLB（Bag of Little Bootstraps）** 方法计算置信区间：
> 1. 对每棵树，用 Bootstrap 重采样估计 CATE 的分布
> 2. 聚合所有树的 Bootstrap 分布
> 3. 取 2.5% 和 97.5% 分位数作为 95% CI
>
> BLB 是一种高效的 Bootstrap 方法，比传统 Bootstrap 快得多，适合 DML 场景。

### 5.3 异常处理 try/except

```python
try:
    te_lower, te_upper = est.effect_interval(X_test, alpha=0.05)
except Exception:
    print("    (effect_interval 不可用, 跳过置信区间估计)")
    te_lower = treatment_effects - 1.96 * np.std(treatment_effects)
    te_upper = treatment_effects + 1.96 * np.std(treatment_effects)
```

**异常处理**：某些版本的 econml 可能不支持 `effect_interval`，或者在某些边缘情况下会报错。用 try/except 捕获异常，回退到"近似置信区间"。

#### 回退方案

```python
te_lower = treatment_effects - 1.96 * np.std(treatment_effects)
te_upper = treatment_effects + 1.96 * np.std(treatment_effects)
```

如果 `effect_interval` 不可用，用 **正态近似** 计算置信区间：

- `1.96`：95% 正态分位数（Φ⁻¹(0.975) ≈ 1.96）。
- `np.std(treatment_effects)`：CATE 估计的标准差（跨 50 个测试点）。
- `treatment_effects ± 1.96 * std`：近似 95% CI。

> ⚠️ **注意：回退方案是粗略近似**
>
> 这种回退方案假设 CATE 估计服从正态分布，且用跨测试点的标准差代替每个点的标准误。这是一个粗略近似，不如 BLB 准确。但在 `effect_interval` 不可用时，至少能提供一个置信区间的估计。
>
> 如果你看到 `(effect_interval 不可用, 跳过置信区间估计)` 这行输出，说明用了回退方案，置信区间仅供参考。
 
---

## 六、CATE 估计结果的解读

虽然本模块不绘制图表（图表在模块 2 的模型比较中绘制），但我们可以解读 CATE 估计的数值。

### 6.1 ATE（平均处理效应）的近似

在模块 2 中，我们会计算 ATE ≈ `np.mean(treatment_effects)`。根据结果文件：

```
DROrthoForest 估计的平均处理效应 (ATE): -0.2630
```

这意味着：**平均而言，转移使存活概率降低约 26.3%**。

> 💡 **重点概念：ATE vs CATE**
>
> - **ATE（Average Treatment Effect）**：总体平均效应，一个数字。ATE = -0.2630 表示平均效应是 -26.3%。
> - **CATE（Conditional Average Treatment Effect）**：条件平均效应，一个函数 CATE(x)。CATE(Age=30) 可能是 -0.15，CATE(Age=70) 可能是 -0.40，表示效应随年龄变化。
>
> ATE 是 CATE 在 X 分布上的平均：ATE = E[CATE(X)]。本教程用 `np.mean(treatment_effects)` 近似 ATE，其中 treatment_effects 是 50 个测试点的 CATE。

### 6.2 CATE 的异质性

如果 CATE 在不同 Age 点差异很大，说明存在异质性（转移效应随年龄变化）。如果 CATE 几乎是常数，说明异质性弱。

具体的 CATE 曲线会在模块 2 的模型比较图中可视化。

---
 

---

## 小贴士

> 💡 **小贴士 1：DROrthoForest 训练较慢，请耐心等待**
>
> DROrthoForest 训练 100 棵树，每棵树都要训练多个 nuisance model（LogisticRegression + Lasso）。3000 样本约需 30–60 秒。如果觉得太慢，可以：
> - 减少 `n_trees` 到 50（更快，但估计方差略大）。
> - 减少 `N_SAMPLES` 到 2000（更快，但统计效力下降）。
> - 增加 `n_jobs` 到 -1（并行，但结果可能不可复现）。

> 💡 **小贴士 2：lambda_reg 是 econml 推荐的经验公式**
>
> `lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))` 是 econml 文档推荐的正则化参数计算公式。它让正则化强度与样本数成反比、与特征数的对数成正比。你可以调整公式中的常数（如 10）来改变正则化强度，但建议保持默认值。

> 💡 **小贴士 3：propensity_model 和 model_Y 的选择影响估计质量**
>
> DROrthoForest 的双重稳健性意味着"结果模型或倾向模型至少一个正确即可"。但两个模型都正确时，估计最有效。本教程用 Lasso 和 LogisticRegression (L1)，是经典选择。如果关系非线性，可以用 `RandomForestRegressor` 和 `RandomForestClassifier` 替代，但计算成本更高。

> 💡 **小贴士 4：effect_interval 可能不可用**
>
> 某些版本的 econml 不支持 DROrthoForest 的 `effect_interval`。如果报错，代码会自动回退到正态近似。但正态近似是粗略的，置信区间仅供参考。如果需要准确的置信区间，建议升级 econml 或改用 CausalForestDML（模块 2）。

> 💡 **小贴士 5：CATE 的符号含义**
>
> CATE 为负表示转移降低存活概率（符合临床预期）。CATE 为正可能意味着：
> 1. 存在未观测到的混淆因素。
> 2. 样本量不足，估计不稳定。
> 3. 选择偏差（如转移患者接受了更好的治疗）。
>
> 本教程的 ATE = -0.2630（负），符合临床预期。

---

## 常见问题

> ❓ **Q1：DROrthoForest 的"双重稳健"到底是什么意思？**
>
> **A**："双重稳健"意味着"结果模型 E[Y|X, W] 或倾向模型 P(T=1|X, W) 至少一个正确设定，因果效应估计就是一致的"。这给了我们"两次机会"——即使一个模型设定错误，只要另一个正确，估计仍然可靠。这在实践中很重要，因为我们很少能确保模型完全正确。

> ❓ **Q2：lambda_reg 的公式是怎么来的？**
>
> **A**：`lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))` 是 econml 文档推荐的经验公式。它基于 Lasso 的理论保证：正则化参数应与 `sqrt(log(p) / n)` 成正比，其中 p 是特征数，n 是样本数。公式中的 10 是经验常数，subsample_ratio 反映子采样的影响。

> ❓ **Q3：为什么 propensity_model 用 LogisticRegression，model_Y 用 Lasso？**
>
> **A**：因为 T 是二值的（0/1），用 LogisticRegression 预测 P(T=1|X, W) 是二分类任务。Y 虽然也是 0/1，但 DML 内部用回归（而非分类）预测 E[Y|X, W]，所以用 Lasso。这是 DML 的标准做法——把 Y 当作连续值回归，即使 Y 是二值的。

> ❓ **Q4：propensity_model 和 propensity_model_final 有什么区别？**
>
> **A**：`propensity_model` 用于"第一层" nuisance model 估计——在整个 Bootstrap 样本上训练，预测 P(T=1|X, W) 计算残差。`propensity_model_final` 用于"第二层"最终估计——在叶子节点内做最终的 CATE 估计。两层的模型可以不同，但本教程用相同的配置。

> ❓ **Q5：为什么 `est.fit(Y, T, X=X, W=W)` 中 X 和 W 要用关键字参数？**
>
> **A**：因为 econml 的 DML 接口要求 X 和 W 用关键字参数显式指定。Y 和 T 是位置参数（按顺序传入），X 和 W 是关键字参数（用 `X=` 和 `W=` 指定）。这种设计是为了避免混淆——DML 有四个变量，不能用单纯的 `fit(X, y)` 接口。如果写成 `est.fit(Y, T, X, W)`（全位置参数），会报错。

> ❓ **Q6：`effect_interval` 用什么方法计算置信区间？**
>
> **A**：DROrthoForest 用 **BLB（Bag of Little Bootstraps）** 方法计算置信区间。BLB 是一种高效的 Bootstrap 方法：对每棵树用 Bootstrap 重采样估计 CATE 的分布，聚合后取分位数作为 CI。BLB 比传统 Bootstrap 快得多，适合 DML 场景。

> ❓ **Q7：如果 `effect_interval` 报错怎么办？**
>
> **A**：代码用 try/except 捕获异常，回退到正态近似：`te_lower = treatment_effects - 1.96 * np.std(treatment_effects)`。这是粗略近似，不如 BLB 准确。如果需要准确的置信区间，建议升级 econml 或改用 CausalForestDML（模块 2），后者的 `effect_interval` 更稳定。

> ❓ **Q8：DROrthoForest 的 ATE = -0.2630 是什么意思？**
>
> **A**：ATE = -0.2630 表示**平均而言，转移使存活概率降低约 26.3%**。这是 50 个测试点 CATE 的平均值 `np.mean(treatment_effects)`。负号符合临床预期——转移患者预后更差。注意这是因果效应（去除混淆后），不是简单的相关性。

---

## 本模块小结

本模块完成了 DROrthoForest 的**训练和 CATE 估计**：

1. **理解了 DROrthoForest 的双重稳健性原理**：结果模型或倾向模型至少一个正确即可保证估计一致。

2. **计算了正则化参数 lambda_reg**：
   - 公式：`lambda_reg = np.sqrt(np.log(n_features_W) / (10 * subsample_ratio * n_samples))`
   - 值：≈ 0.0134
   - 含义：与样本数成反比、与特征数的对数成正比。

3. **配置了 DROrthoForest 的 8 个关键参数**：
   - `n_trees=100`：100 棵树。
   - `min_leaf_size=10`：叶子最小 10 个样本。
   - `max_depth=20`：最大深度 20。
   - `subsample_ratio=0.3`：每棵树用 30% 样本。
   - `propensity_model=LogisticRegression(L1)`：倾向模型。
   - `model_Y=Lasso`：结果模型。
   - `propensity_model_final=LogisticRegression(L1)`：最终倾向模型。
   - `model_Y_final=WeightedLasso`：最终结果模型（支持权重）。

4. **训练了模型**：`est.fit(Y, T, X=X, W=W)`，耗时约 30–60 秒。

5. **估计了 CATE 点值**：`treatment_effects = est.effect(X_test)`，形状 (50,)。

6. **估计了 95% 置信区间**：`est.effect_interval(X_test, alpha=0.05)`，返回 te_lower 和 te_upper。带异常处理回退。

**核心结果**：
- DROrthoForest ATE ≈ -0.2630（平均转移使存活概率降低 26.3%）。
- CATE 曲线将在模块 2 的模型比较图中可视化。

**下一模块**将训练第二个 DML 模型——CausalForestDML，并与 DROrthoForest 做并排可视化比较，计算两个模型的 ATE 并写入结果文件。

---

 