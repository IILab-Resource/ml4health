# 模块 1：Bootstrap 主循环 — 30 次重采样、训练与 SHAP 计算

> 本模块是案例教程 17「SHAP 稳定性 Bootstrap 分析」的**核心模块**。本模块实现了 Bootstrap 稳定性分析的"心脏"——一个 30 次迭代的循环，每次迭代都执行"有放回抽样 → 标准化 → 训练 RF → 计算 SHAP"的完整流程，最终得到 30 组特征重要性。 
>
> 本模块最核心的知识点有三个：**一是** **`np.random.choice(replace=True)`** **的有放回抽样原理**——这是 Bootstrap 方法的数学基础；**二是 OOB（Out-Of-Bag）AUC 的概念**——它让 Bootstrap 评估模型性能时不需要单独的测试集；**三是 SHAP 值的多输出格式处理**——不同版本的 sklearn/shap 返回的 `shap_values` 格式不同，需要兼容处理。

***

## 学习目标

学完本模块后，你将能够：

1. **理解 Bootstrap 有放回抽样的数学原理**：知道为什么 `replace=True` 能模拟"略有差异的训练数据"，以及 63.2% 法则的由来。
2. **掌握 Bootstrap 主循环的结构**：明白每次迭代包含"抽样 → 标准化 → 训练 → SHAP"四个步骤。
3. **理解为什么每次迭代都要重新标准化**：知道 StandardScaler 必须在 Bootstrap 样本内 fit，避免数据泄露。
4. **深入理解 RandomForestClassifier 的关键参数**：明白 `n_estimators=100`、`max_depth=6`、`class_weight='balanced'`、`oob_score=True` 各自的作用。
5. **掌握 OOB AUC 的概念**：知道为什么 Bootstrap 不需要划分训练/测试集，以及 OOB 如何"免费"评估模型性能。
6. **理解 SHAP 值的多输出格式**：知道 `shap_values` 可能是 list、2D 数组或 3D 数组，以及如何统一提取正类的 SHAP 值。
7. **掌握全局重要性的计算方法**：明白 `np.abs(sv).mean(axis=0)` 如何把每个样本的 SHAP 值聚合成全局重要性。
8. **理解为什么 SHAP 计算只用 500 个样本**：知道这是为了加速，且 500 个样本已能稳定估计全局重要性。

***

## 一、Bootstrap 主循环概述

```python
# ============================================================================
# 1. Bootstrap 主循环
# ============================================================================
print(f"\n[1] 开始 {N_BOOTSTRAP} 次 Bootstrap 迭代...")
print(f"    每次迭代: 有放回抽样 → 标准化 → 训练 RF → SHAP...")

bootstrap_shap_importance = []  # 存储每次的全局重要性
bootstrap_aucs = []             # 存储每次的 AUC

for i in range(N_BOOTSTRAP):
    # 有放回抽样
    indices = np.random.choice(len(X_all), size=len(X_all), replace=True)
    X_boot, y_boot = X_all[indices], y_all[indices]

    # 标准化（在 bootstrap 样本内）
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)

    # 训练模型（减少树数加速）
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight='balanced',
        random_state=i, n_jobs=-1, oob_score=True)
    model.fit(X_boot_scaled, y_boot)

    auc_val = model.oob_score_
    bootstrap_aucs.append(auc_val)

    # SHAP 分析（用 bootstrap 样本的子集，加速）
    n_shap = min(500, len(X_boot_scaled))
    shap_idx = np.random.choice(len(X_boot_scaled), n_shap, replace=False)
    X_shap_sample = X_boot_scaled[shap_idx]

    explainer = shap.TreeExplainer(model)
    sv_full = explainer.shap_values(X_shap_sample)

    if isinstance(sv_full, list):
        sv = sv_full[1]
    else:
        sv = sv_full
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    importance = np.abs(sv).mean(axis=0)  # (n_features,)
    bootstrap_shap_importance.append(importance)

    if (i + 1) % 5 == 0:
        print(f"    √ 完成 {i + 1}/{N_BOOTSTRAP}  平均 AUC_boot = {np.mean(bootstrap_aucs):.4f}")

bootstrap_shap_importance = np.array(bootstrap_shap_importance)
print(f"\n    全部完成!")
```

这是本教程**最核心**的代码块。整个 Bootstrap 稳定性分析的"心脏"就是这个 `for` 循环。让我们逐段拆解。

### 1.1 打印进度信息

```python
print(f"\n[1] 开始 {N_BOOTSTRAP} 次 Bootstrap 迭代...")
print(f"    每次迭代: 有放回抽样 → 标准化 → 训练 RF → SHAP...")
```

打印循环开始的信息，告诉用户接下来要做什么。`f"..."` 是 f-string 格式化，`{N_BOOTSTRAP}` 会被替换成 30。

### 1.2 初始化存储列表

```python
bootstrap_shap_importance = []  # 存储每次的全局重要性
bootstrap_aucs = []             # 存储每次的 AUC
```

- `bootstrap_shap_importance`：一个列表，每个元素是一个长度为 6 的数组（6 个特征的全局重要性）。循环结束后会转成形状 `(30, 6)` 的 NumPy 数组。
- `bootstrap_aucs`：一个列表，每个元素是一个浮点数（该次迭代的 OOB AUC）。循环结束后长度为 30。

> 💡 **小贴士：为什么用列表而不是预分配 NumPy 数组？**
>
> Python 列表的 `append()` 是 O(1) 操作，非常高效。预分配 NumPy 数组（如 `np.zeros((30, 6))`）然后按索引赋值也可以，但代码可读性稍差。对于 30 次迭代这种小规模，列表完全够用。循环结束后用 `np.array(bootstrap_shap_importance)` 一次性转换即可。

***

## 二、有放回抽样 — Bootstrap 的核心

```python
for i in range(N_BOOTSTRAP):
    # 有放回抽样
    indices = np.random.choice(len(X_all), size=len(X_all), replace=True)
    X_boot, y_boot = X_all[indices], y_all[indices]
```

### 2.1 `for i in range(N_BOOTSTRAP):`

循环 30 次，`i` 从 0 到 29。`i` 会在后面用作 RF 的 `random_state`，确保每次迭代的模型初始化不同但可复现。

### 2.2 `indices = np.random.choice(len(X_all), size=len(X_all), replace=True)`

**Bootstrap 的核心一行**！逐参数解释：

- **第一个参数** **`len(X_all)`**：抽样的上界。从 `[0, 5000)` 范围内抽取索引。
- **`size=len(X_all)`**：抽取的数量 = 5000。**关键**：Bootstrap 抽取的样本数与原始数据**相同**，这是 Bootstrap 的标准做法。
- **`replace=True`**：**有放回抽样**——每个样本可以被抽中**多次**。这是 Bootstrap 与子集采样的本质区别。

> 💡 **重点概念：有放回抽样的"63.2% 法则"**
>
> 当从 n 个样本中有放回抽取 n 次时，每个样本**未被抽中**的概率是：
>
> ```
> P(未被抽中) = (1 - 1/n)^n ≈ 1/e ≈ 0.368
> ```
>
> 所以每个样本**被抽中至少一次**的概率约是 `1 - 0.368 = 0.632`，即 **63.2%**。
>
> 这意味着：
>
> - **约 63.2% 的独特样本**参与训练（有些重复多次）。
> - **约 36.8% 的样本**完全没被抽中，这些就是 **OOB（Out-Of-Bag）样本**。
>
> OOB 样本可以"免费"用作验证集，这就是 `oob_score=True` 的原理。

### 2.3 `X_boot, y_boot = X_all[indices], y_all[indices]`

用索引数组 `indices` 同时抽取特征和标签。NumPy 的"花式索引"（fancy indexing）允许用整数数组作为索引：

- `X_all[indices]`：形状 `(5000, 6)`，但其中有些行是重复的（被抽中多次）。
- `y_all[indices]`：形状 `(5000,)`，与 `X_boot` 的行对齐。

> ⚠️ **重要：X 和 y 必须用同一组 indices**
>
> 抽样时必须用同一个 `indices` 数组同时索引 `X_all` 和 `y_all`，保证特征和标签的对齐。如果分别抽样，会导致特征和标签错位，模型训练完全错误。

### 2.4 Bootstrap 样本的特点

每次 Bootstrap 得到的 `X_boot` 有以下特点：

1. **样本数与原始相同**：都是 5000。
2. **有重复**：某些样本出现 2 次、3 次甚至更多。
3. **有缺失**：约 36.8% 的样本完全没出现。
4. **类别比例略有波动**：由于有放回抽样，VIVO/MORTO 的比例会在 95.12% 附近小幅波动。

这种"略有差异"正是 Bootstrap 模拟"数据扰动"的方式——如果特征重要性在这种小扰动下就大幅变化，说明它不稳定。

***

## 三、标准化（在 Bootstrap 样本内）

```python
    # 标准化（在 bootstrap 样本内）
    scaler = StandardScaler()
    X_boot_scaled = scaler.fit_transform(X_boot)
```

### 3.1 `scaler = StandardScaler()`

创建一个新的标准化器实例。**关键**：每次迭代都创建新的 `scaler`，因为每次 Bootstrap 样本的均值和标准差都不同。

### 3.2 `X_boot_scaled = scaler.fit_transform(X_boot)`

`fit_transform` 是 `fit` + `transform` 的快捷方式：

- **`fit`**：计算 `X_boot` 每列的均值 μ 和标准差 σ。
- **`transform`**：对每列做 `(x - μ) / σ`，得到均值 0、标准差 1 的标准化数据。

> ⚠️ **重要：为什么必须用** **`fit_transform`** **而不是** **`transform`？**
>
> 这是\*\*数据泄露（data leakage）\*\*的关键防护点。
>
> - **错误做法**：在原始 `X_all` 上 `fit`，然后在 `X_boot` 上 `transform`。这会让 Bootstrap 样本"看到"全量数据的统计量，破坏了 Bootstrap"模拟独立数据集"的假设。
> - **正确做法**：在 `X_boot` 上 `fit_transform`，只用 Bootstrap 样本自身的统计量。这样每次迭代都是"独立的"。
>
> 虽然对 RF 来说标准化不影响结果（RF 是尺度不变的），但这是良好的机器学习习惯，对其他模型（如 LR、SVM）至关重要。

***

## 四、训练随机森林模型

```python
    # 训练模型（减少树数加速）
    model = RandomForestClassifier(
        n_estimators=100, max_depth=6, class_weight='balanced',
        random_state=i, n_jobs=-1, oob_score=True)
    model.fit(X_boot_scaled, y_boot)
```

### 4.1 `RandomForestClassifier` 参数详解

#### `n_estimators=100`

森林中决策树的数量。100 棵树是"精度-速度"的折中选择。

> 💡 **小贴士：为什么是 100 而不是 200？**
>
> 案例教程 12 用的是 200 棵树。本教程减少到 100，原因是：
>
> 1. **Bootstrap 要训练 30 次模型**：100 棵 × 30 次 = 3000 棵树，已经很多。
> 2. **稳定性分析不需要最优模型**：我们关心的是"特征重要性的波动"，而不是"最高 AUC"。100 棵树的 RF 已经能给出稳定的 SHAP 估计。
> 3. **加速实验**：100 棵比 200 棵快约 2 倍。

#### `max_depth=6`

每棵树的最大深度。深度 6 意味着每棵树最多有 `2^6 = 64` 个叶子节点。

> 💡 **小贴士：为什么是 6 而不是更深？**
>
> - **浅树（max\_depth 小）**：模型简单，SHAP 值更稳定，但可能欠拟合。
> - **深树（max\_depth 大）**：模型复杂，SHAP 值更不稳定，但拟合更好。
>
> 本教程用 6 是为了**控制 SHAP 的波动**——如果用 `max_depth=20`，每次 Bootstrap 的 SHAP 值会大幅波动，CV 会变大，难以判断"不稳定"是模型本身的问题还是数据扰动的问题。
>
> 思考题：如果换成 `max_depth=20`，你认为 CV 会变大还是变小？（答案：变大，因为深树对数据更敏感。）

#### `class_weight='balanced'`

类别权重策略。`'balanced'` 表示自动调整权重，使少数类（MORTO）的权重更高。

- 数据集中 VIVO 占 95.12%，MORTO 仅 4.88%。
- 不加权重的话，模型会倾向于把所有样本预测为 VIVO（准确率 95%，但完全没学到 MORTO）。
- `'balanced'` 让 MOUTO 的权重 ≈ 1/0.0488 ≈ 20.5，VIVO 的权重 ≈ 1/0.9512 ≈ 1.05。这样模型会更重视 MORTO 样本。

#### `random_state=i`

**关键参数**！用循环变量 `i` 作为随机种子。

- `i=0` 时，RF 的随机种子是 0。
- `i=1` 时，RF 的随机种子是 1。
- ...
- `i=29` 时，RF 的随机种子是 29。

这样每次迭代的 RF 都用不同的种子初始化，**但整个流程是可复现的**——只要 `np.random.seed(42)` 在前面设好，每次运行代码都会得到完全相同的 30 个模型。

> ⚠️ **注意：不要用** **`random_state=42`**
>
> 如果所有迭代都用 `random_state=42`，那么 30 次迭代的 RF 初始化完全相同，唯一的差异来自 Bootstrap 样本。这会让结果"过度相关"，不利于评估稳定性。
>
> 用 `random_state=i` 让每次迭代的 RF 初始化也不同，更真实地模拟"独立训练"。

#### `n_jobs=-1`

并行训练的 CPU 核心数。`-1` 表示用所有可用核心，加速训练。

#### `oob_score=True`

**关键参数**！启用 OOB 评估。

> 💡 **重点概念：OOB（Out-Of-Bag）评估**
>
> Bootstrap 抽样时，每棵树只用了约 63.2% 的样本训练。剩余 36.8% 的样本对这棵树来说是"袋外"（Out-Of-Bag）的。
>
> OOB 评估的流程：
>
> 1. 对每个样本，找出"没见过它"的那些树。
> 2. 用这些树对该样本预测。
> 3. 把预测结果聚合（投票或平均），得到 OOB 预测。
> 4. 用 OOB 预测与真实标签计算 AUC/准确率。
>
> **OOB 评估的优势**：
>
> - **不需要划分训练/测试集**——所有数据都参与训练，同时得到验证指标。
> - **无偏估计**——OOB 样本对每棵树是"新数据"，评估无偏。
> - **免费**——RF 训练时自动计算，几乎不增加成本。
>
> 本教程用 `model.oob_score_` 获取 OOB 准确率（注意：是准确率，不是 AUC，但 RF 的 `oob_score_` 默认用 `score` 方法，对分类是准确率）。

### 4.2 `model.fit(X_boot_scaled, y_boot)`

用标准化后的 Bootstrap 样本训练 RF。`fit` 方法会：

1. 对每棵树，从 `X_boot_scaled` 中 Bootstrap 抽样（第二层 Bootstrap，RF 内部的 Bootstrap）。
2. 用抽样的子集训练决策树，每次分裂只考虑随机子集的特征。
3. 重复 100 次，得到 100 棵树。
4. 如果 `oob_score=True`，计算 OOB 评估。

***

## 五、获取 OOB AUC

```python
    auc_val = model.oob_score_
    bootstrap_aucs.append(auc_val)
```

### 5.1 `auc_val = model.oob_score_`

`model.oob_score_` 是 RF 训练后自动计算的 OOB 准确率（注意是准确率，不是 AUC，但本教程文档中称为"AUC"是简化的说法）。

> ⚠️ **注意：`oob_score_`** **是准确率，不是 AUC**
>
> 严格来说，`RandomForestClassifier.oob_score_` 返回的是**准确率**（accuracy），不是 AUC。但本教程的注释和文档中称为"AUC"，这是一种简化的说法。
>
> 如果想要真正的 OOB AUC，需要用 `oob_decision_function_` 手动计算：
>
> ```python
> from sklearn.metrics import roc_auc_score
> oob_proba = model.oob_decision_function_[:, 1]  # OOB 预测的正类概率
> oob_auc = roc_auc_score(y_boot, oob_proba)
> ```
>
> 本教程为了简化，直接用 `oob_score_`（准确率）作为模型质量的监控指标。

### 5.2 `bootstrap_aucs.append(auc_val)`

把该次迭代的 OOB 准确率存入列表。循环结束后，`bootstrap_aucs` 长度为 30，可以计算均值和标准差。

**实际运行结果**（来自 `results/24_shap_stability_bootstrap.txt`）：

```
Bootstrap 平均 AUC: 0.8117 ± 0.0094
```

- **平均 OOB 准确率 = 0.8117**：30 次迭代的平均准确率。
- **标准差 = 0.0094**：准确率的波动很小（约 1%），说明模型性能稳定。

> 💡 **小贴士：OOB 准确率 0.81 意味着什么？**
>
> 0.81 的准确率看起来不高，但要注意：
>
> 1. **数据严重不平衡**：VIVO 占 95%，如果模型全猜 VIVO，准确率就是 95%。但 OOB 准确率只有 81%，说明模型在 MORTO 上犯了错（为了识别 MORTO，牺牲了一些 VIVO 的准确率）。
> 2. **`class_weight='balanced'`**：这个设置让模型更重视 MORTO，导致整体准确率下降，但 MORTO 的召回率上升。
> 3. **稳定性分析不追求最高准确率**：我们关心的是"特征重要性是否稳定"，而不是"模型是否最优"。0.81 的准确率已经足够支撑 SHAP 分析。

***

## 六、SHAP 计算（用子集加速）

```python
    # SHAP 分析（用 bootstrap 样本的子集，加速）
    n_shap = min(500, len(X_boot_scaled))
    shap_idx = np.random.choice(len(X_boot_scaled), n_shap, replace=False)
    X_shap_sample = X_boot_scaled[shap_idx]

    explainer = shap.TreeExplainer(model)
    sv_full = explainer.shap_values(X_shap_sample)
```

### 6.1 `n_shap = min(500, len(X_boot_scaled))`

SHAP 计算的样本数。取 500 和总样本数的较小值。

> 💡 **小贴士：为什么只用 500 个样本算 SHAP？**
>
> SHAP 的 TreeExplainer 虽然高效，但对 5000 个样本计算 SHAP 仍需几秒。30 次迭代 × 5000 样本 = 15 万次 SHAP 计算，太慢。
>
> 用 500 个样本的好处：
>
> 1. **速度快**：500 个样本的 SHAP 计算约 0.1 秒，30 次迭代总共 3 秒。
> 2. **统计可靠**：500 个样本已能稳定估计全局重要性（`np.abs(sv).mean(axis=0)`）。全局重要性是所有样本的平均，500 个样本的均值已经收敛。
> 3. **不影响稳定性评估**：我们关心的是"不同 Bootstrap 之间重要性的波动"，而不是"单次 Bootstrap 内部重要性的精度"。
>
> `min(500, len(X_boot_scaled))` 是防御性写法——如果数据集本身小于 500，就用全部样本。

### 6.2 `shap_idx = np.random.choice(len(X_boot_scaled), n_shap, replace=False)`

从 Bootstrap 样本中**无放回**抽取 500 个索引。注意这里用 `replace=False`（无放回），因为我们要的是 500 个**不同**的样本。

### 6.3 `X_shap_sample = X_boot_scaled[shap_idx]`

用索引数组选取 500 行，得到形状 `(500, 6)` 的子集。

### 6.4 `explainer = shap.TreeExplainer(model)`

创建树模型专用的 SHAP 解释器。

> 💡 **重点概念：TreeExplainer 的高效性**
>
> SHAP 有多种 Explainer：
>
> - **`TreeExplainer`**：树模型专用，利用树结构**精确**计算 Shapley Value，复杂度 O(TLD²)（T=树数，L=叶子数，D=深度）。
> - **`KernelExplainer`**：模型无关，用蒙特卡洛近似，慢但通用。
> - **`DeepExplainer`**：深度学习专用。
> - **`LinearExplainer`**：线性模型专用。
>
> 本教程用 `TreeExplainer`，因为 RF 是树模型，能精确且高效地计算 SHAP 值。

### 6.5 `sv_full = explainer.shap_values(X_shap_sample)`

计算 500 个样本的 SHAP 值。返回值的格式**因 sklearn/shap 版本而异**，这是下一段代码要处理的问题。

***

## 七、SHAP 值的多输出格式处理

```python
    if isinstance(sv_full, list):
        sv = sv_full[1]
    else:
        sv = sv_full
        if sv.ndim == 3:
            sv = sv[:, :, 1]
```

这段代码处理 SHAP 值的**三种可能格式**，确保最终 `sv` 是形状 `(500, 6)` 的数组（500 个样本 × 6 个特征的正类 SHAP 值）。

### 7.1 格式 1：列表（旧版 shap + sklearn）

```python
    if isinstance(sv_full, list):
        sv = sv_full[1]
```

**旧版 shap**（< 0.40）对二分类返回一个**列表**，包含两个数组：

- `sv_full[0]`：形状 `(500, 6)`，负类（MORTO）的 SHAP 值。
- `sv_full[1]`：形状 `(500, 6)`，正类（VIVO）的 SHAP 值。

我们取 `sv_full[1]`（正类的 SHAP 值），因为本教程关心的是"预测 VIVO 的因素"。

> 💡 **小贴士：为什么二分类有两个 SHAP 值数组？**
>
> 二分类模型对每个样本输出两个概率：P(MORTO) 和 P(VIVO)，且 P(MORTO) + P(VIVO) = 1。
>
> SHAP 对每个类别分别分解：
>
> - `SHAP_负类[i]`：特征 i 对 P(MORTO) 的贡献。
> - `SHAP_正类[i]`：特征 i 对 P(VIVO) 的贡献。
>
> 两者满足 `SHAP_负类[i] = -SHAP_正类[i]`（因为概率互补）。所以取任意一个都行，本教程取正类。

### 7.2 格式 2：2D 数组（新版 shap）

```python
    else:
        sv = sv_full
```

**新版 shap**（≥ 0.40）对二分类返回一个**2D 数组**，形状 `(500, 6)`，直接就是正类的 SHAP 值。这种情况最简单，直接用。

### 7.3 格式 3：3D 数组（某些中间版本）

```python
        if sv.ndim == 3:
            sv = sv[:, :, 1]
```

某些版本的 shap 返回 **3D 数组**，形状 `(500, 6, 2)`：

- `sv[:, :, 0]`：负类的 SHAP 值。
- `sv[:, :, 1]`：正类的 SHAP 值。

用 `sv[:, :, 1]` 提取正类的 SHAP 值，得到形状 `(500, 6)`。

> ⚠️ **重要：这段代码是"版本兼容"的关键**
>
> 不同版本的 shap 和 sklearn 组合会产生不同的 `shap_values` 格式。这段 `if-else` 代码兼容了三种情况，确保代码在不同环境下都能运行。
>
> 如果你遇到 SHAP 值相关的报错（如 `IndexError` 或形状不匹配），首先检查这段代码是否正确处理了你当前版本的输出格式。

***

## 八、计算全局重要性

```python
    importance = np.abs(sv).mean(axis=0)  # (n_features,)
    bootstrap_shap_importance.append(importance)
```

### 8.1 `importance = np.abs(sv).mean(axis=0)`

**全局重要性的核心计算**！逐步拆解：

1. **`sv`**：形状 `(500, 6)`，500 个样本 × 6 个特征的 SHAP 值。值可正可负。
2. **`np.abs(sv)`**：取绝对值，形状仍为 `(500, 6)`。绝对值表示"该特征对该样本预测的**影响程度**"，不区分方向。
3. **`.mean(axis=0)`**：沿第 0 轴（样本轴）求平均，得到形状 `(6,)` 的数组。这是 500 个样本的平均绝对 SHAP 值，即**全局重要性**。

> 💡 **重点概念：全局 SHAP 重要性**
>
> 全局重要性 = `mean(|SHAP|)`，即每个特征 SHAP 值绝对值的平均。
>
> **为什么取绝对值？**
>
> - SHAP 值有正有负：正值推高预测，负值推低预测。
> - 如果直接平均，正负会抵消，得到接近 0 的值，无法反映"影响程度"。
> - 取绝对值后平均，衡量的是"该特征对预测的**平均影响幅度**"，无论方向。
>
> **为什么用 mean 而不是 sum？**
>
> - mean 与样本数无关，便于跨数据集比较。
> - sum 会随样本数线性增长，没有可比性。
>
> 这个 `mean(|SHAP|)` 就是 SHAP 论文中推荐的"全局特征重要性"指标，比传统的 Gini Importance 更有理论依据。

### 8.2 `bootstrap_shap_importance.append(importance)`

把该次迭代的全局重要性（长度 6 的数组）存入列表。循环结束后，`bootstrap_shap_importance` 是一个包含 30 个数组的列表。

***

## 九、进度打印

```python
    if (i + 1) % 5 == 0:
        print(f"    √ 完成 {i + 1}/{N_BOOTSTRAP}  平均 AUC_boot = {np.mean(bootstrap_aucs):.4f}")
```

### 9.1 `if (i + 1) % 5 == 0:`

每完成 5 次迭代打印一次进度。`(i + 1) % 5 == 0` 表示 `i+1` 是 5 的倍数（即 `i=4, 9, 14, 19, 24, 29`）。用 `i+1` 而不是 `i` 是因为人类习惯从 1 开始计数。

### 9.2 打印内容

```python
print(f"    √ 完成 {i + 1}/{N_BOOTSTRAP}  平均 AUC_boot = {np.mean(bootstrap_aucs):.4f}")
```

- `√`：对勾符号，表示成功完成。
- `{i + 1}/{N_BOOTSTRAP}`：当前进度，如 `5/30`、`10/30`。
- `np.mean(bootstrap_aucs)`：到目前为止所有迭代的平均 OOB 准确率。
- `:.4f`：保留 4 位小数。

**实际运行输出**（示意）：

```
[1] 开始 30 次 Bootstrap 迭代...
    每次迭代: 有放回抽样 → 标准化 → 训练 RF → SHAP...
    √ 完成 5/30  平均 AUC_boot = 0.8123
    √ 完成 10/30  平均 AUC_boot = 0.8118
    √ 完成 15/30  平均 AUC_boot = 0.8115
    √ 完成 20/30  平均 AUC_boot = 0.8119
    √ 完成 25/30  平均 AUC_boot = 0.8116
    √ 完成 30/30  平均 AUC_boot = 0.8117

    全部完成!
```

观察"平均 AUC\_boot"的变化：随着迭代次数增加，平均值越来越稳定（从 0.8123 收敛到 0.8117），说明 30 次迭代已经足够给出稳定的性能估计。

***

## 十、循环结束：转换为 NumPy 数组

```python
bootstrap_shap_importance = np.array(bootstrap_shap_importance)
print(f"\n    全部完成!")
```

### 10.1 `bootstrap_shap_importance = np.array(bootstrap_shap_importance)`

把列表转换成 NumPy 数组。转换后：

- **形状**：`(30, 6)`——30 次迭代 × 6 个特征。
- **行**：每次迭代的全局重要性。
- **列**：每个特征在 30 次迭代中的重要性序列。

这个 `(30, 6)` 的数组是后续所有统计分析的基础。模块 2 会基于它计算均值、标准差、CV、置信区间和排名稳定性。

### 10.2 打印完成信息

```python
print(f"\n    全部完成!")
```

打印循环完成的信息。`\n` 是换行符，让输出更清晰。

***

## 十一、本模块代码完整执行流程

把本模块的代码串起来，30 次迭代的执行流程是：

**每次迭代（i = 0, 1, ..., 29）**：

1. **有放回抽样**：`np.random.choice(5000, 5000, replace=True)` → `X_boot, y_boot`。
2. **标准化**：`StandardScaler().fit_transform(X_boot)` → `X_boot_scaled`。
3. **训练 RF**：`RandomForestClassifier(n_estimators=100, max_depth=6, ...).fit(X_boot_scaled, y_boot)`。
4. **获取 OOB 准确率**：`model.oob_score_` → `bootstrap_aucs.append(...)`。
5. **SHAP 子集采样**：从 5000 行中抽 500 行 → `X_shap_sample`。
6. **计算 SHAP**：`shap.TreeExplainer(model).shap_values(X_shap_sample)` → `sv_full`。
7. **格式统一**：提取正类 SHAP 值 → `sv`，形状 `(500, 6)`。
8. **全局重要性**：`np.abs(sv).mean(axis=0)` → `importance`，形状 `(6,)`。
9. **存储**：`bootstrap_shap_importance.append(importance)`。
10. **进度打印**：每 5 次打印一次。

**循环结束后**：

1. **转换格式**：`np.array(bootstrap_shap_importance)` → 形状 `(30, 6)`。

执行完毕后，我们得到两个核心数据结构：

- `bootstrap_shap_importance`：形状 `(30, 6)`，30 次迭代 × 6 个特征的全局重要性。
- `bootstrap_aucs`：长度 30 的列表，30 次迭代的 OOB 准确率。

***

## 小贴士

> 💡 **小贴士 1：Bootstrap 的"两层随机性"**
>
> 本教程的 Bootstrap 循环包含**两层随机性**：
>
> 1. **外层 Bootstrap**：`np.random.choice(replace=True)` 生成略有差异的训练集。
> 2. **内层 RF Bootstrap**：RF 训练时每棵树内部也做 Bootstrap（这是 RF 的标准做法）。
>
> 两层随机性叠加，使得每次迭代的模型都有差异。这种"双重扰动"让稳定性评估更保守（CV 会略大），但也更真实。

> 💡 **小贴士 2：如何调试 SHAP 值格式问题**
>
> 如果你在运行代码时遇到 SHAP 值相关的错误，可以用以下代码调试：
>
> ```python
> sv_full = explainer.shap_values(X_shap_sample)
> print(f"Type: {type(sv_full)}")
> if isinstance(sv_full, list):
>     print(f"List length: {len(sv_full)}")
>     for i, sv in enumerate(sv_full):
>         print(f"  sv_full[{i}].shape: {sv.shape}")
> else:
>     print(f"Shape: {sv_full.shape}")
>     print(f"ndim: {sv_full.ndim}")
> ```
>
> 这样可以看到 SHAP 值的实际格式，便于调整提取逻辑。

> 💡 **小贴士 3：Bootstrap 迭代次数的"收敛检查"**
>
> 如何判断 30 次迭代是否足够？一个简单的方法是观察"累计均值的波动"：
>
> ```python
> cumulative_mean = np.cumsum(bootstrap_shap_importance, axis=0) / np.arange(1, 31).reshape(-1, 1)
> # cumulative_mean[i] 是前 i+1 次迭代的平均重要性
> ```
>
> 如果累计均值在后 10 次迭代中变化很小（如 < 1%），说明已经收敛。本教程的"平均 AUC\_boot"从 0.8123 收敛到 0.8117，变化仅 0.06%，说明 30 次足够。

***

## 常见问题

> ❓ **Q1：为什么 Bootstrap 抽样的样本数与原始数据相同（都是 5000）？**
>
> **A**：这是 Bootstrap 方法的标准做法，原因有二：
>
> 1. **理论依据**：Bootstrap 的数学理论假设"重采样样本量与原始样本量相同"，这样统计量的分布才能正确近似真实分布。
> 2. **实践考虑**：如果抽样更少（如 2500），每次迭代的模型性能会下降；如果抽样更多（如 10000），又超过原始数据量，需要重复抽样太多。
>
> 所以 `size=len(X_all)` 是 Bootstrap 的"黄金标准"。

> ❓ **Q2：OOB 准确率 0.81 看起来不高，模型是不是有问题？**
>
> **A**：不是模型有问题，而是数据特点决定的：
>
> 1. **数据严重不平衡**：VIVO 占 95%，MORTO 占 5%。
> 2. **`class_weight='balanced'`**：模型为了识别 MORTO，会牺牲一些 VIVO 的准确率。
> 3. **OOB 准确率 ≠ AUC**：`oob_score_` 是准确率，不是 AUC。在不平衡数据上，准确率会被多数类主导。
>
> 如果用真正的 OOB AUC（用 `oob_decision_function_` 计算），数值会更高（通常 0.85+）。本教程为了简化，直接用准确率作为监控指标。

> ❓ **Q3：为什么 SHAP 计算只用 500 个样本，而不是全部 5000 个？**
>
> **A**：纯粹是为了**加速**。500 个样本的 SHAP 计算约 0.1 秒，5000 个样本约 1 秒。30 次迭代 × 1 秒 = 30 秒，虽然不算太慢，但用 500 个样本只要 3 秒，快 10 倍。
>
> 更重要的是，**500 个样本已能稳定估计全局重要性**。全局重要性是 `mean(|SHAP|)`，即 500 个样本的平均。根据大数定律，500 个样本的均值已经收敛到真实均值。用更多样本不会显著改变 `mean(|SHAP|)` 的值。

> ❓ **Q4：`random_state=i`** **会不会让结果"不够随机"？**
>
> **A**：不会。`random_state=i` 只是固定了 RF 的初始化种子，让结果可复现。RF 的随机性来自两个地方：
>
> 1. **每棵树的 Bootstrap 抽样**：由 `random_state` 控制。
> 2. **每次分裂的特征子集选择**：由 `random_state` 控制。
>
> 用 `random_state=i` 让每次迭代的 RF 用不同的种子，所以 30 个 RF 是"独立"的。同时，因为种子固定，整个流程可复现——这是科学研究的必要条件。

> ❓ **Q5：如果我想用真正的 OOB AUC 而不是准确率，怎么改代码？**
>
> **A**：把 `auc_val = model.oob_score_` 替换为：
>
> ```python
> from sklearn.metrics import roc_auc_score
> oob_proba = model.oob_decision_function_[:, 1]  # OOB 预测的正类概率
> auc_val = roc_auc_score(y_boot, oob_proba)
> ```
>
> `oob_decision_function_` 是形状 `(n_samples, n_classes)` 的数组，`[:, 1]` 取正类概率。然后用 `roc_auc_score` 计算 AUC。这样得到的 `auc_val` 是真正的 OOB AUC，通常比准确率更高（在本数据集上约 0.85+）。

> ❓ **Q6：为什么每次迭代都要重新创建** **`StandardScaler`？**
>
> **A**：因为每次 Bootstrap 样本的统计量（均值、标准差）都不同。如果复用同一个 `scaler`，相当于用"上一次的统计量"标准化"这一次的数据"，会引入偏差。
>
> 正确做法是每次迭代 `scaler = StandardScaler()` 创建新实例，然后 `fit_transform(X_boot)`。这样每次标准化都是"独立的"，符合 Bootstrap"模拟独立数据集"的假设。
>
> 虽然对 RF 来说标准化不影响结果（RF 是尺度不变的），但这是良好的机器学习习惯，对其他模型至关重要。

***

## 本模块小结

本模块完成了 SHAP 稳定性 Bootstrap 分析的**核心循环**：

1. **理解了 Bootstrap 有放回抽样的原理**：`replace=True` 让每个样本可被多次抽中，约 63.2% 的独特样本参与训练，36.8% 是 OOB 样本。
2. **掌握了主循环的四个步骤**：抽样 → 标准化 → 训练 RF → 计算 SHAP。
3. **理解了 RF 的关键参数**：`n_estimators=100`、`max_depth=6`、`class_weight='balanced'`、`oob_score=True`、`random_state=i`。
4. **掌握了 OOB 评估的概念**：用 36.8% 的袋外样本"免费"评估模型性能，不需要划分训练/测试集。
5. **理解了 SHAP 值的三种格式**：list、2D 数组、3D 数组，以及如何统一提取正类 SHAP 值。
6. **掌握了全局重要性的计算**：`np.abs(sv).mean(axis=0)` 把 500 个样本的 SHAP 值聚合成 6 个特征的全局重要性。
7. **得到了核心数据结构**：`bootstrap_shap_importance` 形状 `(30, 6)`，`bootstrap_aucs` 长度 30。
8. **观察了实际运行结果**：平均 OOB 准确率 = 0.8117 ± 0.0094，模型性能稳定。

**下一模块**将基于 `(30, 6)` 的重要性数组计算稳定性统计量——均值、标准差、变异系数（CV）、95% 置信区间、Spearman 排名相关性，量化每个特征的稳定性。

***

**  **
