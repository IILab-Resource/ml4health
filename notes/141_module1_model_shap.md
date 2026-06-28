# 模块 1：模型训练、SHAP 计算与特征排名

> 本模块是案例教程 14「带特征分布的 SHAP 依赖图」的**模型准备模块**。我们将完成三件事：**一是训练随机森林分类器**——用与前序教程完全相同的参数（`n_estimators=200, max_depth=8, class_weight='balanced'`），保证 SHAP 值可比；**二是计算 SHAP 值**——用 `shap.TreeExplainer` 对测试集前 500 个样本计算 SHAP 值，并处理新版 sklearn 的 list/ndarray 两种输出格式；**三是计算特征重要性并排序**——用 `np.abs(sv).mean(0)` 计算 mean |SHAP|，用 `np.argsort(...)[::-1]` 从大到小排序，选出 Top 6 特征作为后续依赖图的主角。
>
> 本模块最核心的知识点有三个：**一是随机森林的关键参数**——`n_estimators=200` 控制树的数量、`max_depth=8` 控制树的深度、`class_weight='balanced'` 处理类别不平衡、`n_jobs=-1` 启用多核并行；**二是 SHAP 值的多维数组处理**——理解 `sv = shap_values[1]`（list 格式）和 `sv = shap_values[:, :, 1]`（ndarray 格式）两种情况的兼容处理；**三是 `feature_order` 的作用**——后续所有模块都基于这个排序索引选择 Top 6 特征。

---

## 学习目标

学完本模块后，你将能够：

1. **掌握随机森林的关键参数**：理解 `n_estimators=200`、`max_depth=8`、`class_weight='balanced'`、`n_jobs=-1` 各自的作用。
2. **理解为什么用 RF 作为被解释模型**：知道 TreeExplainer 的高效性、RF 的 AUC 优势、以及 SHAP 对树模型的友好支持。
3. **掌握 AUC 的计算**：理解 `roc_auc_score(y_te, model.predict_proba(X_te_final)[:, 1])` 的含义。
4. **理解 SHAP 样本的选择**：知道为什么取测试集前 500 个样本，而不是全部 3000 个。
5. **掌握 `shap.TreeExplainer` 的使用**：理解树模型专用解释器的高效性。
6. **掌握 SHAP 值的多维数组处理**：理解 `isinstance(sv_full, list)` 和 `sv.ndim == 3` 两种兼容情况。
7. **理解 mean |SHAP| 的计算**：知道 `np.abs(sv).mean(0)` 如何把每个特征的 SHAP 值聚合成一个重要性分数。
8. **掌握 `np.argsort(...)[::-1]` 的排序技巧**：理解如何从大到小排序特征。
9. **理解 Top 6 特征的选择**：知道为什么选 6 个（与 2×3 网格对应）。
10. **能够解读特征排名**：理解实际运行结果中 year 排第一、Extension 排最后的含义。

---

## 一、模块 1：训练 Random Forest + SHAP

### 1.1 代码结构

```python
# ============================================================================
# 1. 模型 + SHAP
# ============================================================================
print("\n[1] 训练 Random Forest + SHAP...")
model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_tr_final, y_tr)
auc = roc_auc_score(y_te, model.predict_proba(X_te_final)[:, 1])
print(f"    RF AUC = {auc:.4f}")

n_shap = 500
X_shap = X_te_final[:n_shap]
y_shap = y_te[:n_shap]

explainer = shap.TreeExplainer(model)
sv_full = explainer.shap_values(X_shap)
if isinstance(sv_full, list):
    sv = sv_full[1]
else:
    sv = sv_full
    if sv.ndim == 3:
        sv = sv[:, :, 1]

shap_importance = np.abs(sv).mean(0)
feature_order = np.argsort(shap_importance)[::-1]
print(f"    特征排名: {[feature_names[i] for i in feature_order]}")
print(f"    Mean |SHAP|: {[f'{shap_importance[i]:.4f}' for i in feature_order]}")

top_n = 6
top6_idx = feature_order[:top_n]
```

### 1.2 训练随机森林

```python
model = RandomForestClassifier(
    n_estimators=200, max_depth=8, class_weight='balanced',
    random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_tr_final, y_tr)
```

#### `n_estimators=200`

**树的数量**。随机森林由 200 棵决策树组成。树越多，模型越稳定（方差越小），但计算成本越高。200 是一个常用的平衡值——既保证了稳定性，又不会太慢。

> 💡 **为什么是 200 而不是 100 或 500？**
>
> - 100 棵树：可能不够稳定，AUC 波动较大。
> - 200 棵树：AUC 已经收敛，再多收益递减。
> - 500 棵树：AUC 提升有限，但训练时间增加 2.5 倍。
>
> 经验法则：从 100 开始，逐步增加到 AUC 收敛为止。本教程用 200 是前序教程（案例 8、12）的设定，保持一致。

#### `max_depth=8`

**树的最大深度**。每棵树最多有 8 层。深度越大，模型越复杂（能拟合更复杂的模式），但容易过拟合。

- 深度 8 意味着每棵树最多有 2^8 = 256 个叶子节点。
- 对于 6 个特征的数据集，深度 8 已经足够拟合复杂的交互。
- 如果用 `max_depth=None`（不限制），树会一直生长直到所有叶子纯净，容易过拟合。

> 💡 **为什么限制深度？**
>
> 限制深度是**正则化**的一种方式。在医学数据中，过拟合的模型会在新数据上表现差（AUC 下降）。`max_depth=8` 是一个温和的限制，既允许复杂模式，又防止过拟合。

#### `class_weight='balanced'`

**类别权重**。让 RF 自动调整类别权重，权重与类别频率成反比。

- 本数据集 VIVO 49.30%，MORTO 50.70%，接近平衡。
- 但 `class_weight='balanced'` 仍然有用——它让 RF 在分裂时更关注少数类（虽然这里少数类不明显）。
- 如果数据严重不平衡（如 VIVO 10%，MORTO 90%），`class_weight='balanced'` 会大幅提升少数类的权重，防止模型把所有样本都预测成多数类。

#### `random_state=RANDOM_STATE`

**随机种子**。保证 RF 的训练可复现——同样的数据、同样的参数、同样的种子，每次训练得到完全相同的模型。

RF 的随机性来自两个地方：
1. **Bootstrap 采样**：每棵树从训练集中有放回地采样。
2. **特征子采样**：每个节点分裂时，随机选择一部分特征。

`random_state` 固定了这两个随机过程，保证可复现。

#### `n_jobs=-1`

**并行训练**。`-1` 表示使用所有 CPU 核心。RF 的 200 棵树可以并行训练，`n_jobs=-1` 能显著加速。

- `n_jobs=1`：单核训练（慢）。
- `n_jobs=-1`：所有核并行（快）。
- `n_jobs=4`：用 4 个核。

#### `model.fit(X_tr_final, y_tr)`

在训练集上拟合模型。`X_tr_final` 是标准化后的训练集特征矩阵（形状 7000×6），`y_tr` 是训练集目标向量（形状 7000）。

### 1.3 计算 AUC

```python
auc = roc_auc_score(y_te, model.predict_proba(X_te_final)[:, 1])
print(f"    RF AUC = {auc:.4f}")
```

- `model.predict_proba(X_te_final)`：预测测试集每个样本属于各类的概率，返回形状 `(3000, 2)` 的数组。
  - 第 0 列：MORTO（类别 0）的概率。
  - 第 1 列：VIVO（类别 1）的概率。
- `[:, 1]`：取第 1 列（VIVO 概率），得到形状 `(3000,)` 的数组。
- `roc_auc_score(y_te, ...)`：计算 ROC 曲线下面积（AUC）。

实际运行输出：

```
    RF AUC = 0.8800
```

> 💡 **AUC = 0.8800 的含义**
>
> AUC（Area Under the ROC Curve）是分类模型的常用评估指标，取值范围 [0, 1]：
> - AUC = 0.5：随机猜测（模型没用）。
> - AUC = 0.7：可接受。
> - AUC = 0.8：良好。
> - AUC = 0.88：**优秀**。
> - AUC = 1.0：完美分类。
>
> 本教程的 RF AUC = 0.8800，说明模型在测试集上有优秀的分类性能。这个 AUC 与案例 12、12b 一致（因为模型和数据完全相同）。

### 1.4 选择 SHAP 样本

```python
n_shap = 500
X_shap = X_te_final[:n_shap]
y_shap = y_te[:n_shap]
```

- `n_shap = 500`：取 500 个样本计算 SHAP 值。
- `X_shap = X_te_final[:n_shap]`：取测试集前 500 个样本的特征矩阵，形状 `(500, 6)`。
- `y_shap = y_te[:n_shap]`：取测试集前 500 个样本的目标向量，形状 `(500,)`。

> 💡 **为什么只取 500 个而不是全部 3000 个？**
>
> SHAP 的计算成本与样本数线性相关。`TreeExplainer` 的复杂度是 O(n × T × log T)，其中 n 是样本数，T 是树的大小。对于 200 棵树、500 个样本，计算时间约几秒；如果用 3000 个样本，时间会增加到几十秒。
>
> 500 个样本已经足够稳定地估计 SHAP 值的分布。前序教程（案例 12、12b）也用 500 个样本，保持一致。

> 💡 **为什么取"前 500 个"而不是"随机 500 个"？**
>
> `X_te_final[:n_shap]` 取的是测试集的前 500 个样本（按索引顺序）。由于 `train_test_split` 已经是随机划分，测试集的顺序本身就是随机的，所以"前 500 个"等价于"随机 500 个"。这样写更简洁，也不需要再设置随机种子。

### 1.5 创建 SHAP 解释器

```python
explainer = shap.TreeExplainer(model)
```

`shap.TreeExplainer` 是 SHAP 库为树模型（决策树、随机森林、XGBoost、LightGBM 等）专门设计的解释器。它利用树结构的高效算法，计算 SHAP 值的速度比通用解释器（`KernelExplainer`、`DeepExplainer`）快几个数量级。

> 💡 **为什么用 TreeExplainer 而不是 KernelExplainer？**
>
> - **TreeExplainer**：专门针对树模型，复杂度 O(T × L × D)，T 是树数，L 是叶子数，D 是深度。对于 200 棵树，计算 500 个样本只需几秒。
> - **KernelExplainer**：模型无关，但需要反复扰动样本并预测，复杂度 O(n × M)，M 是扰动次数（通常 1000+）。对于 500 个样本，可能需要几分钟。
>
> 对于 RF 模型，TreeExplainer 是首选。

### 1.6 计算 SHAP 值

```python
sv_full = explainer.shap_values(X_shap)
```

`explainer.shap_values(X_shap)` 计算 500 个样本的 SHAP 值。返回值的格式取决于 sklearn 和 shap 的版本：

- **旧版格式（list）**：返回一个 list，长度等于类别数。`sv_full[0]` 是负类（MORTO）的 SHAP 值，`sv_full[1]` 是正类（VIVO）的 SHAP 值。每个元素形状 `(500, 6)`。
- **新版格式（ndarray）**：返回一个 ndarray，形状 `(500, 6, 2)`。第 3 维是类别。

### 1.7 处理 SHAP 值的两种格式

```python
if isinstance(sv_full, list):
    sv = sv_full[1]
else:
    sv = sv_full
    if sv.ndim == 3:
        sv = sv[:, :, 1]
```

这段代码兼容两种格式，提取**正类（VIVO）的 SHAP 值**：

#### 情况 1：list 格式

```python
if isinstance(sv_full, list):
    sv = sv_full[1]
```

- `isinstance(sv_full, list)`：检查 `sv_full` 是否是 list。
- `sv = sv_full[1]`：取第 1 个元素（正类 VIVO 的 SHAP 值），形状 `(500, 6)`。

#### 情况 2：ndarray 格式

```python
else:
    sv = sv_full
    if sv.ndim == 3:
        sv = sv[:, :, 1]
```

- `sv = sv_full`：先假设是 2D 数组（形状 `(500, 6)`），直接用。
- `if sv.ndim == 3:`：如果是 3D 数组（形状 `(500, 6, 2)`），需要进一步处理。
- `sv = sv[:, :, 1]`：取第 3 维的第 1 个元素（正类 VIVO），得到形状 `(500, 6)`。

> 💡 **为什么需要兼容两种格式？**
>
> sklearn 和 shap 的版本更新会改变 `shap_values()` 的返回格式。旧版 sklearn（< 1.0）的 RF 返回 list，新版（>= 1.0）可能返回 ndarray。这段兼容代码保证教程在不同环境下都能运行。
>
> **关键概念**：`sv` 最终的形状是 `(500, 6)`——500 个样本，6 个特征。`sv[i, j]` 表示第 i 个样本的第 j 个特征的 SHAP 值。

### 1.8 计算特征重要性

```python
shap_importance = np.abs(sv).mean(0)
```

- `np.abs(sv)`：对 SHAP 值取绝对值，形状 `(500, 6)`。因为 SHAP 值可正可负，取绝对值后才能比较"贡献大小"。
- `.mean(0)`：沿第 0 轴（样本轴）求均值，得到形状 `(6,)` 的数组。每个元素是该特征的**平均绝对 SHAP 值**（mean |SHAP|）。

> 💡 **mean |SHAP| 的含义**
>
> mean |SHAP| 是 SHAP 库默认的特征重要性度量。它回答"整体上，这个特征平均贡献多大？"。
>
> - mean |SHAP| 大 → 该特征对很多样本都有较大贡献 → 重要。
> - mean |SHAP| 小 → 该特征对大多数样本贡献小 → 不重要。
>
> 注意：mean |SHAP| 是**绝对值**的均值，不区分正负方向。要看方向，需要看依赖图。

### 1.9 特征排序

```python
feature_order = np.argsort(shap_importance)[::-1]
```

- `np.argsort(shap_importance)`：返回排序后的索引（**升序**）。例如 `shap_importance = [0.03, 0.27, 0.04]`，`argsort` 返回 `[1, 2, 0]`（最小的 0.03 在索引 0，最大的 0.27 在索引 1）。
- `[::-1]`：反转数组，变成**降序**。`[1, 2, 0]` 变成 `[0, 2, 1]`。

所以 `feature_order[0]` 是最重要的特征的索引，`feature_order[-1]` 是最不重要的。

### 1.10 打印特征排名

```python
print(f"    特征排名: {[feature_names[i] for i in feature_order]}")
print(f"    Mean |SHAP|: {[f'{shap_importance[i]:.4f}' for i in feature_order]}")
```

- `[feature_names[i] for i in feature_order]`：用列表推导式，按重要性从高到低取出特征名。
- `[f'{shap_importance[i]:.4f}' for i in feature_order]`：按重要性从高到低取出 mean |SHAP|，保留 4 位小数。

实际运行输出：

```
    特征排名: ['year', 'Diagnostic.means', 'Code.Profession', 'Age', 'Raca.Color', 'Extension']
    Mean |SHAP|: ['0.2690', '0.0611', '0.0373', '0.0312', '0.0235', '0.0110']
```

### 1.11 特征排名解读

| 排名 | 特征 | Mean \|SHAP\| | 占比 | 解读 |
|------|------|--------------|------|------|
| 1 | **year** | 0.2690 | 62.5% | 诊断年份，最重要的特征，贡献远超其他 |
| 2 | Diagnostic.means | 0.0611 | 14.2% | 诊断方式，第二重要 |
| 3 | Code.Profession | 0.0373 | 8.7% | 职业代码 |
| 4 | Age | 0.0312 | 7.2% | 患者年龄 |
| 5 | Raca.Color | 0.0235 | 5.5% | 种族/肤色 |
| 6 | Extension | 0.0110 | 2.6% | 肿瘤扩展程度，最不重要 |

> 💡 **year 占 62.5% 的总贡献**
>
> year 的 mean |SHAP| = 0.2690，是第二名 Diagnostic.means（0.0611）的 4.4 倍。这说明 year 是模型的"主导特征"——它一个特征就贡献了 62.5% 的总 SHAP 重要性。
>
> 这种"一超多强"的格局在医学数据中常见——通常有一个时间相关的特征（诊断年份、随访年份）主导预测，因为医疗技术和生存率随时间变化。

### 1.12 选择 Top 6 特征

```python
top_n = 6
top6_idx = feature_order[:top_n]
```

- `top_n = 6`：选前 6 个特征。
- `top6_idx = feature_order[:top_n]`：取 `feature_order` 的前 6 个元素，得到 Top 6 特征的索引数组。

> 💡 **为什么选 6 个？**
>
> 因为后续的依赖图是 2×3 网格（2 行 3 列 = 6 个子图）。6 个特征正好填满网格。本教程只有 6 个特征，所以 Top 6 就是全部特征。如果特征更多（如 20 个），Top 6 会筛选出最重要的 6 个。

---

## 二、实际运行结果

### 2.1 控制台输出

```
[1] 训练 Random Forest + SHAP...
    RF AUC = 0.8800
    特征排名: ['year', 'Diagnostic.means', 'Code.Profession', 'Age', 'Raca.Color', 'Extension']
    Mean |SHAP|: ['0.2690', '0.0611', '0.0373', '0.0312', '0.0235', '0.0110']
```

### 2.2 关键数据

| 指标 | 值 | 含义 |
|------|-----|------|
| RF AUC | 0.8800 | 模型在测试集上的分类性能（优秀） |
| SHAP 样本数 | 500 | 用于计算 SHAP 值的样本数 |
| 特征数 | 6 | 总特征数 |
| Top 1 特征 | year | 最重要的特征（mean \|SHAP\| = 0.2690） |
| Top 6 特征 | Extension | 最不重要的特征（mean \|SHAP\| = 0.0110） |

### 2.3 与前序教程的一致性

本模块的模型训练和 SHAP 计算与案例 12、12b **完全一致**：

| 教程 | RF AUC | SHAP 样本 | Top 1 特征 | mean \|SHAP\| |
|------|--------|----------|-----------|--------------|
| 案例 12 | 0.8800 | 500 | year | 0.2690 |
| 案例 12b | 0.8800 | 500 | year | 0.2690 |
| **案例 14（本教程）** | **0.8800** | **500** | **year** | **0.2690** |

这保证了后续的"密度 Ratio"指标可以与前序的"R²"指标组合，形成完整的特征评估框架。

---

## 小贴士

### 💡 小贴士 1：SHAP 值的符号含义

SHAP 值可正可负：
- **正值** → 推高 VIVO 概率（让模型更倾向于预测"存活"）。
- **负值** → 推低 VIVO 概率（让模型更倾向于预测"死亡"）。
- **0** → 该特征对这个样本的预测没有影响。

在依赖图中，y 轴是 SHAP 值：
- y > 0 的点 → 该特征值推高 VIVO 概率。
- y < 0 的点 → 该特征值推低 VIVO 概率。

### 💡 小贴士 2：mean |SHAP| vs 传统 Feature Importance

| 指标 | 计算方式 | 优点 | 缺点 |
|------|---------|------|------|
| 传统 Feature Importance（Gini） | 基于不纯度下降 | 计算快 | 偏向高基数特征，无法看方向 |
| **mean \|SHAP\|** | 基于博弈论 Shapley Value | 公平、一致、可加 | 计算慢 |

mean |SHAP| 是更可靠的特征重要性度量，因为它基于博弈论的 Shapley Value，满足"一致性"公理（如果模型改变使某特征贡献增大，其 SHAP 重要性不会减小）。

### 💡 小贴士 3：`feature_order` 的复用

`feature_order` 是一个 NumPy 数组，存储了按重要性降序排列的特征索引。后续模块会反复使用它：

```python
# 模块 2 中
for rank, feat_idx in enumerate(top6_idx):  # top6_idx = feature_order[:6]
    ...
```

这种"一次排序，多次复用"的设计避免了重复计算，也让代码更清晰。

### 💡 小贴士 4：`sv` 的形状

`sv` 的形状是 `(500, 6)`：
- 第 0 轴（500）：样本。
- 第 1 轴（6）：特征。

`sv[i, j]` 表示第 i 个样本的第 j 个特征的 SHAP 值。例如 `sv[0, 1]` 是第 0 个样本的 year（索引 1）的 SHAP 值。

---

## 常见问题

### Q1: 为什么 RF AUC = 0.8800？这个数字怎么来的？

**A**: AUC 是 ROC 曲线下面积。ROC 曲线的 x 轴是假阳性率（FPR），y 轴是真阳性率（TPR）。AUC = 0.8800 意味着：随机选一个 VIVO 样本和一个 MORTO 样本，模型给 VIVO 样本的预测概率高于 MORTO 样本的概率是 0.88。

AUC 的优点：
- 不受阈值影响（评估的是概率排序，不是分类结果）。
- 不受类别不平衡影响（即使 VIVO 只有 1%，AUC 仍然有效）。

### Q2: 为什么 `class_weight='balanced'` 对接近平衡的数据也有用？

**A**: 本数据集 VIVO 49.30%，MORTO 50.70%，接近平衡。`class_weight='balanced'` 仍然有用，因为：

1. **保险作用**：如果数据分布有微小变化（如采样导致 VIVO 降到 45%），`balanced` 会自动调整权重。
2. **分裂准则**：RF 在分裂时考虑类别权重，`balanced` 让分裂更关注少数类（即使少数类只少一点点）。
3. **预测概率**：`balanced` 会让 RF 的预测概率更接近真实概率（而不是偏向多数类）。

### Q3: `shap.TreeExplainer` 和 `shap.Explainer` 有什么区别？

**A**:
- `shap.TreeExplainer(model)`：专门针对树模型，利用树结构高效计算 SHAP 值。复杂度 O(T × L × D)。
- `shap.Explainer(model, ...)`：通用解释器，自动选择合适的算法（Tree、Deep、Kernel、Linear）。更方便但可能不如专用解释器高效。

本教程用 `TreeExplainer` 是为了明确性和效率。在生产环境中，可以用 `shap.Explainer(model)` 让 SHAP 自动选择。

### Q4: `sv_full` 为什么可能是 list 也可能是 ndarray？

**A**: 这是 sklearn 和 shap 版本更新的结果：

- **旧版 sklearn（< 1.0）**：RF 的 `predict_proba` 返回 list，`shap_values` 也返回 list。
- **新版 sklearn（>= 1.0）**：RF 的 `predict_proba` 可能返回 ndarray，`shap_values` 也返回 ndarray。

兼容代码 `if isinstance(sv_full, list): ... else: if sv.ndim == 3: ...` 保证两种格式都能正确处理。

### Q5: 为什么 `np.argsort(shap_importance)[::-1]` 而不是 `np.argsort(-shap_importance)`？

**A**: 两种写法都能实现降序排序，但有细微差别：

- `np.argsort(shap_importance)[::-1]`：先升序排序，再反转。对于相等的值，保持原始顺序（稳定排序）。
- `np.argsort(-shap_importance)`：对负值升序排序。对于相等的值，可能不保持原始顺序（不稳定排序）。

本教程用 `[::-1]` 是为了稳定性——如果有两个特征的重要性相等，它们的相对顺序保持不变。

### Q6: 如果特征数超过 6 个，Top 6 怎么选？

**A**: `top6_idx = feature_order[:top_n]` 取 `feature_order` 的前 6 个元素。如果特征数 > 6，这会选出最重要的 6 个。如果特征数 = 6（本教程的情况），这会选出全部特征（按重要性排序）。

### Q7: SHAP 值的单位和范围是什么？

**A**: SHAP 值的单位与模型输出的单位一致。对于 RF 分类器，SHAP 值是**对数几率（log odds）**的贡献，不是直接的概率贡献。

- SHAP 值 = +0.1 → 该特征让 log odds 增加 0.1（即 VIVO/MORTO 的几率乘以 e^0.1 ≈ 1.105）。
- SHAP 值 = -0.1 → 该特征让 log odds 减少 0.1。

本教程的 SHAP 值范围大约在 [-0.5, +0.5] 之间（从依赖图可以看出）。

### Q8: 为什么 year 这么重要？

**A**: year 是诊断年份。在医学数据中，诊断年份通常反映：
1. **医疗技术进步**：近年诊断的患者有更好的治疗手段。
2. **数据偏倚**：早年诊断的患者随访时间长，可能更多被记录为 MORTO。
3. **生存率变化**：某些癌症的生存率随时间提高。

所以 year 主导预测是合理的，但也要警惕"时间偏倚"——模型可能学到了"早年 = 死亡"的虚假关联。

---

## 本模块小结

本模块完成了**模型训练、SHAP 计算与特征排名**：

1. **训练随机森林**：
   - 参数：`n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1`。
   - 在训练集（7000 样本）上拟合，在测试集（3000 样本）上评估。
   - **RF AUC = 0.8800**（优秀）。

2. **选择 SHAP 样本**：
   - 取测试集前 500 个样本（`X_shap`、`y_shap`）。
   - 500 个样本足够稳定估计 SHAP 分布，计算成本可控。

3. **计算 SHAP 值**：
   - 用 `shap.TreeExplainer(model)` 创建树模型专用解释器。
   - `explainer.shap_values(X_shap)` 计算 SHAP 值。
   - 兼容 list 和 ndarray 两种格式，提取正类（VIVO）的 SHAP 值 `sv`，形状 `(500, 6)`。

4. **特征重要性排名**：
   - `shap_importance = np.abs(sv).mean(0)`：计算 mean |SHAP|。
   - `feature_order = np.argsort(shap_importance)[::-1]`：降序排序。

5. **实际运行结果**：

| 排名 | 特征 | Mean \|SHAP\| |
|------|------|--------------|
| 1 | year | 0.2690 |
| 2 | Diagnostic.means | 0.0611 |
| 3 | Code.Profession | 0.0373 |
| 4 | Age | 0.0312 |
| 5 | Raca.Color | 0.0235 |
| 6 | Extension | 0.0110 |

6. **选择 Top 6 特征**：
   - `top6_idx = feature_order[:6]`：取前 6 个特征的索引。
   - 这 6 个特征将作为模块 2 双轴依赖图的主角。

**核心发现**：
1. RF AUC = 0.8800，模型性能优秀。
2. year 是最重要的特征（mean |SHAP| = 0.2690），占总贡献的 62.5%。
3. 特征排名与前序教程（案例 12、12b）完全一致，保证可比性。
4. SHAP 值 `sv` 的形状是 `(500, 6)`，后续模块会反复使用。

**下一步**：在模块 2 中，我们将用这 6 个特征的 SHAP 值绘制双轴依赖图——左轴是特征值频数分布（直方图），右轴是 SHAP 值（散点图），并叠加二次拟合曲线和密度标注框。这是本教程的核心可视化。
