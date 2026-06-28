# 模块 2：稳定性统计量计算 — 均值、标准差、CV、置信区间、排名稳定性

> 本模块是案例教程 17「SHAP 稳定性 Bootstrap 分析」的**分析模块**。本模块把模块 1 得到的 30 组特征重要性（形状 `(30, 6)` 的数组）转换成 5 个稳定性指标：均值、标准差、变异系数（CV）、95% 置信区间、Spearman 排名相关性。这些指标共同回答"特征重要性是否可靠"的核心问题。 
>
> 本模块最核心的知识点有三个：**一是变异系数（CV）的定义和判断标准**——CV = std/mean，是无量纲的波动度量，CV < 0.1 表示高度稳定；**二是 95% 置信区间的百分位法计算**——用 P2.5 和 P97.5 作为区间端点，不假设正态分布；**三是 Spearman 排名相关性的计算逻辑**——比较"单次排名"与"平均排名"的一致性，量化整体排序的稳定性。

***

## 学习目标

学完本模块后，你将能够：

1. **理解 5 个稳定性指标的公式和含义**：Mean SHAP、Std SHAP、CV、CI Width、Rank ρ。
2. **掌握变异系数（CV）的计算和判断标准**：知道 CV = std/mean，CV < 0.1 高度稳定，0.1–0.3 中等，> 0.3 不稳定。
3. **理解 95% 置信区间的百分位法**：知道用 `np.percentile(arr, 2.5)` 和 `np.percentile(arr, 97.5)` 计算区间端点。
4. **掌握** **`np.argsort`** **的排序索引用法**：知道 `np.argsort(arr)[::-1]` 如何得到降序排列的索引。
5. **理解 Spearman 排名相关性的计算逻辑**：知道如何比较"单次 Bootstrap 排名"与"平均排名"的一致性。
6. **能够解读实际运行结果**：知道 year 的 CV=0.0637（高度稳定）、Code.Profession 的 CV=0.3948（不稳定）意味着什么。
7. **理解** **`1e-8`** **的数值保护作用**：知道为什么 CV 计算中要加 `1e-8` 防止除零。
8. **掌握稳定性判断的"重要性-CV"二维框架**：知道"高重要性+低 CV"和"低重要性+高 CV"的不同解读。

***

## 一、统计量计算概述

```python
# ============================================================================
# 2. 计算统计量
# ============================================================================
print(f"\n[2] 计算稳定性统计量...")

mean_imp = bootstrap_shap_importance.mean(axis=0)
std_imp = bootstrap_shap_importance.std(axis=0)
cv_imp = std_imp / (mean_imp + 1e-8)
ci_lower = np.percentile(bootstrap_shap_importance, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_shap_importance, 97.5, axis=0)
ci_width = ci_upper - ci_lower

sorted_idx = np.argsort(mean_imp)[::-1]

# 排名稳定性
base_ranking = np.argsort(mean_imp)[::-1]  # 平均排名
rank_correlations = []
for boot_imp in bootstrap_shap_importance:
    boot_ranking = np.argsort(boot_imp)[::-1]
    corr, _ = spearmanr(base_ranking, boot_ranking)
    rank_correlations.append(corr)
```

这段代码把 `(30, 6)` 的重要性数组转换成 5 个稳定性指标。让我们逐行拆解。

***

## 二、均值与标准差

```python
mean_imp = bootstrap_shap_importance.mean(axis=0)
std_imp = bootstrap_shap_importance.std(axis=0)
```

### 2.1 `mean_imp = bootstrap_shap_importance.mean(axis=0)`

计算每个特征在 30 次迭代中的**平均重要性**。

- **`bootstrap_shap_importance`**：形状 `(30, 6)`，30 次迭代 × 6 个特征。
- **`.mean(axis=0)`**：沿第 0 轴（迭代轴）求平均，得到形状 `(6,)` 的数组。
- **结果**：`mean_imp[i]` 是第 i 个特征在 30 次迭代中的平均 SHAP 重要性。

> 💡 **重点概念：Mean SHAP 的含义**
>
> Mean SHAP = E\[importance] = 30 次 Bootstrap 的平均重要性。
>
> **解读**：
>
> - 越高 → 特征越重要（平均而言）。
> - 这是"点估计"——单次报告的"特征重要性"。
> - 但点估计无法反映"波动"，需要配合 Std 和 CV 一起看。

### 2.2 `std_imp = bootstrap_shap_importance.std(axis=0)`

计算每个特征在 30 次迭代中的**标准差**。

- **`.std(axis=0)`**：沿第 0 轴求标准差，得到形状 `(6,)` 的数组。
- **结果**：`std_imp[i]` 是第 i 个特征在 30 次迭代中重要性的波动程度。

> 💡 **重点概念：Std SHAP 的含义**
>
> Std SHAP = σ\[importance] = 30 次 Bootstrap 重要性的标准差。
>
> **解读**：
>
> - 越低 → 特征重要性越稳定。
> - 越高 → 特征重要性波动越大。
> - **局限**：Std 是有量纲的（与 Mean 同单位），不便于跨特征比较。比如 Mean=0.22 的特征 Std=0.01，和 Mean=0.02 的特征 Std=0.01，前者的相对波动小得多。所以需要 CV。

### 2.3 实际运行结果

根据 `results/24_shap_stability_bootstrap.csv`：

| 特征               | Mean SHAP | Std SHAP |
| ---------------- | --------- | -------- |
| year             | 0.224881  | 0.014320 |
| Extension        | 0.132820  | 0.010982 |
| Raca.Color       | 0.072443  | 0.013179 |
| Diagnostic.means | 0.050581  | 0.006417 |
| Age              | 0.035635  | 0.005813 |
| Code.Profession  | 0.024023  | 0.009485 |

**观察**：

- `year` 的 Mean 最高（0.225），Std 适中（0.014）。
- `Code.Profession` 的 Mean 最低（0.024），Std 看起来不大（0.009），但**相对于 Mean**来说很大（0.009/0.024 ≈ 0.40）。这就是为什么需要 CV。

***

## 三、变异系数（CV）— 核心稳定性指标

```python
cv_imp = std_imp / (mean_imp + 1e-8)
```

### 3.1 公式

**CV = std / mean**，即标准差除以均值。

- **`std_imp`**：形状 `(6,)`，每个特征的标准差。
- **`mean_imp + 1e-8`**：每个特征的均值加一个极小值 `1e-8`（10⁻⁸）。
- **结果**：`cv_imp[i]` 是第 i 个特征的变异系数，形状 `(6,)`。

### 3.2 为什么除以 mean？

CV 是**无量纲**的相对波动度量：

- Std 有量纲（与 Mean 同单位），不便于跨特征比较。
- CV = Std/Mean 消除了量纲，可以比较"不同量级"的特征。

**举例**：

- `year`：Mean=0.225, Std=0.014 → CV = 0.014/0.225 = 0.0637（6.37%）
- `Code.Profession`：Mean=0.024, Std=0.009 → CV = 0.009/0.024 = 0.3948（39.48%）

虽然 `year` 的 Std（0.014）比 `Code.Profession` 的 Std（0.009）大，但 `year` 的 CV（0.064）远小于 `Code.Profession` 的 CV（0.395）。**CV 才是正确的稳定性比较指标**。

### 3.3 为什么加 `1e-8`？

```python
cv_imp = std_imp / (mean_imp + 1e-8)
```

`1e-8` 是**数值保护**，防止除零错误。

- 如果某个特征的 Mean 恰好为 0（理论上可能，虽然实践中极少），`std/0` 会得到 `inf`（无穷大）或 `NaN`。
- 加一个极小值 `1e-8` 后，`std/(0 + 1e-8) = std/1e-8`，是一个很大的有限数，不会报错。
- `1e-8` 足够小，对正常情况（Mean >> 1e-8）的计算结果几乎无影响。

> 💡 **小贴士：数值保护的常见做法**
>
> 在机器学习中，除法操作的数值保护很常见：
>
> - `x / (y + 1e-8)`：防止 y=0 导致除零。
> - `np.log(x + 1e-10)`：防止 x=0 导致 log(0)=-inf。
> - `x / np.sqrt(y + 1e-8)`：防止 y=0 导致除零。
>
> `1e-8` 是一个常用的"epsilon"值，足够小不影响精度，又足够大防止数值问题。

### 3.4 CV 的判断标准

> 💡 **重点概念：CV 的判断标准**
>
> | CV 范围     | 稳定性等级    | 含义               |
> | --------- | -------- | ---------------- |
> | **< 0.1** | **高度稳定** | 特征重要性波动很小，结论非常可信 |
> | 0.1 – 0.3 | 中等稳定     | 有一定波动，但整体趋势可信    |
> | **> 0.3** | **不稳定**  | 波动很大，结论需谨慎       |
>
> **本教程的判断标准来自代码**（第 320–330 行）：
>
> ```python
> f.write(f"  CV < 0.1: 高度稳定 → ")
> f.write(f"  0.1 ≤ CV < 0.3: 中等稳定 → ")
> f.write(f"  CV ≥ 0.3: 不稳定 → ")
> ```

### 3.5 实际运行结果

根据 `results/24_shap_stability_bootstrap.csv`：

| 特征                  | Mean   | Std    | **CV**     | 稳定性判断      |
| ------------------- | ------ | ------ | ---------- | ---------- |
| **year**            | 0.2249 | 0.0143 | **0.0637** | ✅ **高度稳定** |
| Extension           | 0.1328 | 0.0110 | **0.0827** | ✅ **高度稳定** |
| Raca.Color          | 0.0724 | 0.0132 | 0.1819     | ✅ 中等稳定     |
| Diagnostic.means    | 0.0506 | 0.0064 | 0.1269     | ✅ 中等稳定     |
| Age                 | 0.0356 | 0.0058 | 0.1631     | ✅ 中等稳定     |
| **Code.Profession** | 0.0240 | 0.0095 | **0.3948** | ⚠️ **不稳定** |

**关键观察**：

1. **`year`** **是最稳定的特征**（CV=0.0637），其重要性在 30 次 Bootstrap 中几乎不变。
2. **`Code.Profession`** **是最不稳定的特征**（CV=0.3948），是 `year` 的 6 倍！
3. **2 个高度稳定 + 3 个中等稳定 + 1 个不稳定**，整体稳定性良好。

> 💡 **小贴士：为什么 year 最稳定？**
>
> `year` 的 CV=0.0637，是第二名（Extension, CV=0.0827）的 77%。原因：
>
> 1. **year 是连续变量**，分布良好（近似正态）。
> 2. **year 的重要性极高**（0.225，是第二名的 1.7 倍），真实信号在数据扰动下"幸存"。
> 3. **year 与目标的相关性稳定**——无论 Bootstrap 怎么抽样，year 与 VIVO 的关联始终存在。
>
> **教学启示**：越重要的特征，通常也越稳定。因为真实信号在数据扰动下"幸存"，而噪声特征在不同样本中"忽高忽低"。

> 💡 **小贴士：为什么 Code.Profession 不稳定？**
>
> `Code.Profession` 的 CV=0.3948，是 `year` 的 6 倍。原因：
>
> 1. **该特征值大量为 0**，分布稀疏。
> 2. **Bootstrap 抽样中**，非零值的出现频率在不同样本中差异大。
> 3. **少部分非零值患者被抽中/漏掉**，大幅改变该特征的重要性。
>
> **教学启示**：不稳定不意味着"不重要"——它可能意味着"不稳定重要"，即该特征对某些患者很重要，但不是对所有患者。

***

## 四、95% 置信区间

```python
ci_lower = np.percentile(bootstrap_shap_importance, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_shap_importance, 97.5, axis=0)
ci_width = ci_upper - ci_lower
```

### 4.1 `np.percentile` 的用法

`np.percentile(arr, q, axis=...)` 计算数组的第 q 百分位数。

- **`arr`**：输入数组，这里是 `bootstrap_shap_importance`，形状 `(30, 6)`。
- **`q`**：百分位数，范围 \[0, 100]。
  - `q=2.5`：第 2.5 百分位数（即下界）。
  - `q=97.5`：第 97.5 百分位数（即上界）。
- **`axis=0`**：沿第 0 轴（迭代轴）计算，得到形状 `(6,)` 的数组。

### 4.2 百分位法的原理

```python
ci_lower = np.percentile(bootstrap_shap_importance, 2.5, axis=0)
ci_upper = np.percentile(bootstrap_shap_importance, 97.5, axis=0)
```

**95% 置信区间** = \[P2.5, P97.5]，即第 2.5 百分位数到第 97.5 百分位数。

- 把 30 次迭代的重要性排序。
- P2.5 是"第 30 × 0.025 = 0.75 个"位置的值（用线性插值）。
- P97.5 是"第 30 × 0.975 = 29.25 个"位置的值。
- 区间 \[P2.5, P97.5] 覆盖了 95% 的 Bootstrap 估计值。

> 💡 **重点概念：百分位法 vs 正态法**
>
> 计算 95% CI 有两种方法：
>
> **方法 1：正态法（参数化）**
>
> ```
> CI = mean ± 1.96 × std
> ```
>
> 假设重要性服从正态分布。优点：简单。缺点：如果分布偏斜（如 Code.Profession 的右偏分布），正态法不准确。
>
> **方法 2：百分位法（非参数化）— 本教程采用**
>
> ```
> CI = [P2.5, P97.5]
> ```
>
> 不假设分布形状，直接用经验分布的百分位。优点：对任意分布都适用。缺点：需要足够多的 Bootstrap 迭代（≥ 30）。
>
> 本教程用百分位法，更稳健。

### 4.3 `ci_width = ci_upper - ci_lower`

计算置信区间的**宽度**。

- **`ci_width[i]`** = 第 i 个特征的 95% CI 宽度 = P97.5 - P2.5。
- **解读**：越窄 → 重要性估计越精确；越宽 → 不确定性越大。

### 4.4 实际运行结果

根据 `results/24_shap_stability_bootstrap.csv`：

| 特征               | CI\_Lower | CI\_Upper | CI\_Width |
| ---------------- | --------- | --------- | --------- |
| year             | 0.199852  | 0.246152  | 0.046300  |
| Extension        | 0.113037  | 0.153675  | 0.040638  |
| Raca.Color       | 0.046228  | 0.098210  | 0.051983  |
| Diagnostic.means | 0.039798  | 0.060872  | 0.021074  |
| Age              | 0.027106  | 0.047151  | 0.020045  |
| Code.Profession  | 0.011533  | 0.044311  | 0.032779  |

**关键观察**：

1. **`year`** **的 CI = \[0.200, 0.246]**：很窄，说明"year 重要性 ≈ 0.22"这个结论很可信。
2. **`Code.Profession`** **的 CI = \[0.012, 0.044]**：相对 Mean（0.024）来说很宽（宽度 0.033 > Mean 0.024），说明"Code.Profession 重要性 ≈ 0.024"这个结论不可信。
3. **`Raca.Color`** **的 CI 宽度最大（0.052）**：虽然 Mean 不低（0.072），但波动大。

> 💡 **小贴士：如何用 CI 判断"重要性是否显著"？**
>
> 一个简单的规则：**如果 CI 不包含 0，则该特征"显著重要"**。
>
> 本教程中所有特征的 CI 下界都 > 0，所以都是"显著重要"的。但"显著"不等于"稳定"——Code.Profession 显著（CI 不含 0），但不稳定（CV=0.395）。
>
> 更严格的判断：**如果两个特征的 CI 不重叠，则它们的重要性"显著不同"**。比如 `year` 的 CI \[0.200, 0.246] 与 `Extension` 的 CI \[0.113, 0.154] 不重叠，说明 `year` 显著比 `Extension` 重要。

***

## 五、排序索引

```python
sorted_idx = np.argsort(mean_imp)[::-1]
```

### 5.1 `np.argsort(mean_imp)`

`np.argsort` 返回**排序后的索引**，而不是排序后的值。

- **`mean_imp`**：形状 `(6,)`，6 个特征的平均重要性。
- **`np.argsort(mean_imp)`**：返回索引数组，使得 `mean_imp[np.argsort(mean_imp)]` 是升序排列。

**举例**：如果 `mean_imp = [0.2, 0.1, 0.3]`，则 `np.argsort(mean_imp) = [1, 0, 2]`（因为 0.1 最小，0.2 次之，0.3 最大）。

### 5.2 `[::-1]` 反转

`[::-1]` 是 Python 切片语法，表示"从头到尾，步长 -1"，即**反转数组**。

- `np.argsort(mean_imp)` 是升序索引。
- `np.argsort(mean_imp)[::-1]` 是降序索引。

**结果**：`sorted_idx[0]` 是最重要特征的索引，`sorted_idx[1]` 是第二重要的，依此类推。

### 5.3 实际运行结果

根据 Mean SHAP 的降序排列：

```python
sorted_idx = [1, 5, 4, 3, 0, 2]  # 对应 ['year', 'Extension', 'Raca.Color', 'Diagnostic.means', 'Age', 'Code.Profession']
```

（注：`feature_cols = ['Age', 'year', 'Code.Profession', 'Diagnostic.means', 'Extension', 'Raca.Color']`，所以索引 1 是 year，索引 5 是 Extension，等等。）

这个 `sorted_idx` 会在模块 3 的可视化中大量使用，用于按重要性排序绘制条形图。

***

## 六、排名稳定性 — Spearman 相关系数

```python
# 排名稳定性
base_ranking = np.argsort(mean_imp)[::-1]  # 平均排名
rank_correlations = []
for boot_imp in bootstrap_shap_importance:
    boot_ranking = np.argsort(boot_imp)[::-1]
    corr, _ = spearmanr(base_ranking, boot_ranking)
    rank_correlations.append(corr)
```

这段代码计算**排名稳定性**——每次 Bootstrap 的特征排名与平均排名的一致性。

### 6.1 `base_ranking = np.argsort(mean_imp)[::-1]`

计算"基准排名"——按平均重要性降序排列的索引。

- 这与前面的 `sorted_idx` 完全相同。
- `base_ranking[0]` = 最重要特征的索引（year）。
- `base_ranking[1]` = 第二重要特征的索引（Extension）。
- ...

### 6.2 遍历每次 Bootstrap

```python
for boot_imp in bootstrap_shap_importance:
    boot_ranking = np.argsort(boot_imp)[::-1]
    corr, _ = spearmanr(base_ranking, boot_ranking)
    rank_correlations.append(corr)
```

对每次 Bootstrap 迭代：

1. **`boot_imp`**：该次迭代的 6 个特征重要性，形状 `(6,)`。
2. **`boot_ranking = np.argsort(boot_imp)[::-1]`**：该次迭代的特征排名（降序索引）。
3. **`spearmanr(base_ranking, boot_ranking)`**：计算该次排名与基准排名的 Spearman 相关系数。
4. **`corr, _ = ...`**：`spearmanr` 返回 `(corr, pvalue)` 元组，我们只取 `corr`，忽略 p 值。
5. **`rank_correlations.append(corr)`**：把相关系数存入列表。

### 6.3 `spearmanr` 的工作原理

`scipy.stats.spearmanr(a, b)` 计算 a 和 b 的 Spearman 等级相关系数：

1. 把 a 和 b 分别转成排名（rank）。
2. 计算排名的 Pearson 相关系数。

**举例**：

- `base_ranking = [1, 5, 4, 3, 0, 2]`（year 第一，Extension 第二，...）
- 某次 `boot_ranking = [1, 5, 4, 3, 0, 2]`（完全相同）→ ρ = 1.0
- 某次 `boot_ranking = [1, 4, 5, 3, 0, 2]`（Extension 和 Raca.Color 交换）→ ρ < 1.0
- 某次 `boot_ranking = [2, 0, 4, 3, 5, 1]`（完全乱序）→ ρ 接近 0 或负

> 💡 **重点概念：Spearman ρ 的判断标准**
>
> | ρ 范围      | 排名稳定性   | 含义            |
> | --------- | ------- | ------------- |
> | **> 0.9** | **稳定**  | 单次排名与平均排名高度一致 |
> | 0.7 – 0.9 | 中等      | 有一定偏差，但整体趋势一致 |
> | **< 0.7** | **不稳定** | 排名大幅偏离平均      |
>
> **注意**：Spearman ρ 衡量的是**排名**的一致性，不是**数值**的一致性。即使每次的 SHAP 值波动很大，只要排名顺序不变，ρ 就接近 1。

### 6.4 实际运行结果

根据 `results/24_shap_stability_bootstrap.txt`：

```
排名稳定性:
  平均 Spearman ρ = 0.9067
  中位数 Spearman ρ = 1.0000
  最小 Spearman ρ = 0.4857
  最大 Spearman ρ = 1.0000
  ρ > 0.9 的比例 = 70.0%
```

**关键观察**：

1. **平均 ρ = 0.9067**：整体排名稳定（> 0.9）。
2. **中位数 ρ = 1.0000**：超过一半的迭代排名与平均排名完全一致！
3. **最小 ρ = 0.4857**：有一次迭代排名大幅偏离（< 0.5）。
4. **最大 ρ = 1.0000**：有迭代排名完全一致。
5. **70% 的迭代 ρ > 0.9**：大部分迭代排名稳定。

> 💡 **小贴士：如何解读"最小 ρ = 0.4857"？**
>
> 虽然平均 ρ = 0.9067 看起来很好，但最小值 0.4857 说明**有一次迭代**的排名与平均排名相差甚远。
>
> **这意味着**：如果只训练一次模型，得到的特征排名可能是一个"偏离平均"的偶然结果。这正是 Bootstrap 稳定性分析的价值——它揭示了"单次训练的风险"。
>
> **解决方案**：报告平均排名 + 95% 置信区间，而不是单次排名。这正是本教程的核心输出。

> ⚠️ **注意：中位数 ρ = 1.0000 的含义**
>
> 中位数 ρ = 1.0 意味着 30 次迭代中，**至少有 15 次**的排名与平均排名完全一致（ρ=1.0）。
>
> 这说明大部分迭代中，6 个特征的排名顺序是稳定的（year > Extension > Raca.Color > Diagnostic.means > Age > Code.Profession）。只有少数迭代出现排名交换（主要是中间几个特征的顺序变化）。

***

## 七、输出表格

```python
# 输出表格
print(f"\n{'Feature':<22} {'Mean':>8} {'Std':>8} {'CV':>8} {'CI_W':>8} {'Rank_Stab':>10}")
print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
for idx in sorted_idx:
    fn = feature_names[idx]
    rs = np.mean([np.argmax(boot_imp == mean_imp.max()) == np.argmax(boot_imp == mean_imp.max())
                  for boot_imp in bootstrap_shap_importance])
    print(f"  {fn:<20} {mean_imp[idx]:>8.4f} {std_imp[idx]:>8.4f} "
          f"{cv_imp[idx]:>8.4f} {ci_width[idx]:>8.4f} {'-':>10}")

print(f"\n  平均排名相关系数: {np.mean(rank_correlations):.4f} (1.0=完全稳定)")
print(f"  排名相关系数范围: [{np.min(rank_correlations):.4f}, {np.max(rank_correlations):.4f}]")
```

### 7.1 表头打印

```python
print(f"\n{'Feature':<22} {'Mean':>8} {'Std':>8} {'CV':>8} {'CI_W':>8} {'Rank_Stab':>10}")
print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
```

- **`<22`**：左对齐，宽度 22 字符。
- **`>8`**：右对齐，宽度 8 字符。
- **`'-'*22`**：22 个 `-` 字符，作为分隔线。

### 7.2 遍历排序后的特征

```python
for idx in sorted_idx:
    fn = feature_names[idx]
    ...
```

按重要性降序遍历特征。`sorted_idx` 是降序索引数组，`idx` 是当前特征的索引，`fn` 是特征名。

### 7.3 `rs` 变量（代码中的一个小问题）

```python
rs = np.mean([np.argmax(boot_imp == mean_imp.max()) == np.argmax(boot_imp == mean_imp.max())
              for boot_imp in bootstrap_shap_importance])
```

> ⚠️ **注意：这行代码有逻辑问题**
>
> 仔细看这个列表推导式：`np.argmax(boot_imp == mean_imp.max()) == np.argmax(boot_imp == mean_imp.max())`。两边完全相同，所以结果**永远是 True**，`rs` 永远是 1.0。
>
> 这看起来是一个**未完成的代码**——作者可能想计算"该特征在多少次迭代中排名第一"，但写错了。
>
> 不过这个 `rs` 变量**没有被打印**（打印时用了 `'-'`），所以不影响最终结果。真正的排名稳定性是用 `rank_correlations` 计算的。

### 7.4 打印每个特征的统计量

```python
print(f"  {fn:<20} {mean_imp[idx]:>8.4f} {std_imp[idx]:>8.4f} "
      f"{cv_imp[idx]:>8.4f} {ci_width[idx]:>8.4f} {'-':>10}")
```

- `{fn:<20}`：特征名左对齐，宽度 20。
- `{mean_imp[idx]:>8.4f}`：Mean 右对齐，宽度 8，保留 4 位小数。
- `{std_imp[idx]:>8.4f}`：Std 同上。
- `{cv_imp[idx]:>8.4f}`：CV 同上。
- `{ci_width[idx]:>8.4f}`：CI\_Width 同上。
- `{'-':>10}`：Rank\_Stab 列打印 `-`（因为 `rs` 计算有问题，用占位符）。

### 7.5 打印排名稳定性汇总

```python
print(f"\n  平均排名相关系数: {np.mean(rank_correlations):.4f} (1.0=完全稳定)")
print(f"  排名相关系数范围: [{np.min(rank_correlations):.4f}, {np.max(rank_correlations):.4f}]")
```

- `np.mean(rank_correlations)`：30 次 Spearman ρ 的平均值。
- `np.min(rank_correlations)`：最小值（最差的排名一致性）。
- `np.max(rank_correlations)`：最大值（最好的排名一致性）。

### 7.6 实际运行输出

```
[2] 计算稳定性统计量...

Feature                    Mean      Std       CV      CI_W   Rank_Stab
---------------------- -------- -------- -------- -------- ----------
  year                   0.2249   0.0143   0.0637   0.0463          -
  Extension              0.1328   0.0110   0.0827   0.0406          -
  Raca.Color             0.0724   0.0132   0.1819   0.0520          -
  Diagnostic.means       0.0506   0.0064   0.1269   0.0211          -
  Age                    0.0356   0.0058   0.1631   0.0200          -
  Code.Profession        0.0240   0.0095   0.3948   0.0328          -

  平均排名相关系数: 0.9067 (1.0=完全稳定)
  排名相关系数范围: [0.4857, 1.0000]
```

***

## 八、5 个稳定性指标汇总

本模块计算了 5 个稳定性指标，下表汇总了它们的公式、含义和判断标准：

| 指标            | 公式                            | 含义                    | 判断标准                               |
| ------------- | ----------------------------- | --------------------- | ---------------------------------- |
| **Mean SHAP** | `E[importance]`               | 30 次 Bootstrap 的平均重要性 | 越高则特征越重要                           |
| **Std SHAP**  | `σ[importance]`               | 重要性的波动程度（有量纲）         | 越低越稳定                              |
| **CV**        | `σ / μ`                       | 相对波动量（无量纲）            | <0.1: 高度稳定, 0.1–0.3: 中等, >0.3: 不稳定 |
| **CI Width**  | `P97.5 - P2.5`                | 95% 置信区间的宽度           | 越窄越可信                              |
| **Rank ρ**    | `Spearman(rank_i, rank_mean)` | 单次排名与平均排名的一致性         | >0.9: 稳定, 0.7–0.9: 中等, <0.7: 不稳定   |

### 8.1 完整稳定性表格

 

| 排名 | 特征                  | Mean SHAP | Std    | **CV**     | CI Width | **稳定性判断**  |
| -- | ------------------- | --------- | ------ | ---------- | -------- | ---------- |
| 1  | **year**            | 0.2249    | 0.0143 | **0.0637** | 0.0463   | ✅ **高度稳定** |
| 2  | Extension           | 0.1328    | 0.0110 | **0.0827** | 0.0406   | ✅ **高度稳定** |
| 3  | Raca.Color          | 0.0724    | 0.0132 | 0.1819     | 0.0520   | ✅ 中等稳定     |
| 4  | Diagnostic.means    | 0.0506    | 0.0064 | 0.1269     | 0.0211   | ✅ 中等稳定     |
| 5  | Age                 | 0.0356    | 0.0058 | 0.1631     | 0.0200   | ✅ 中等稳定     |
| 6  | **Code.Profession** | 0.0240    | 0.0095 | **0.3948** | 0.0328   | ⚠️ **不稳定** |

### 8.2 稳定性评估

 

```
稳定性评估:

  CV < 0.1: 高度稳定 → ['year', 'Extension']
  0.1 ≤ CV < 0.3: 中等稳定 → ['Raca.Color', 'Diagnostic.means', 'Age']
  CV ≥ 0.3: 不稳定 → ['Code.Profession']
```

***

## 九、"重要性-CV"二维解读框架

> 💡 **重点概念：高 CV 但低重要性的解读陷阱**
>
> `Code.Profession`: Mean=0.024, CV=0.3948
>
> 两种可能的解读：
>
> **解读 A（错误）**：
>
> > "Code.Profession 不稳定 → 不可信 → 放弃它"
>
> **解读 B（正确）**：
>
> > "Code.Profession 的重要性很低（0.024）且不稳定（0.395）。低重要性 + 高 CV 意味着它的'重要性'主要由抽样噪声决定。应谨慎将其作为'重要特征'报告。"
>
> **判断规则**：
>
> | 组合          | 解读                |
> | ----------- | ----------------- |
> | 高重要性 + 低 CV | 该特征确实重要，结论可信      |
> | 低重要性 + 高 CV | 该特征可能不重要，结论不可信    |
> | 高重要性 + 高 CV | 该特征重要但波动大，需结合领域知识 |
> | 低重要性 + 低 CV | 该特征确实不重要，结论可信     |

本教程的 6 个特征分布在这个二维框架中：

| 特征               | 重要性      | CV       | 象限                        |
| ---------------- | -------- | -------- | ------------------------- |
| year             | 高（0.225） | 低（0.064） | **高重要性 + 低 CV** → 确实重要，可信 |
| Extension        | 高（0.133） | 低（0.083） | **高重要性 + 低 CV** → 确实重要，可信 |
| Raca.Color       | 中（0.072） | 中（0.182） | 中重要性 + 中 CV → 谨慎          |
| Diagnostic.means | 中（0.051） | 中（0.127） | 中重要性 + 中 CV → 谨慎          |
| Age              | 低（0.036） | 中（0.163） | 低重要性 + 中 CV → 可能不重要       |
| Code.Profession  | 低（0.024） | 高（0.395） | **低重要性 + 高 CV** → 不重要，不可信 |

***

## 十、本模块代码完整执行流程

把本模块的代码串起来，执行流程是：

1. **计算均值**：`bootstrap_shap_importance.mean(axis=0)` → `mean_imp`，形状 `(6,)`。
2. **计算标准差**：`bootstrap_shap_importance.std(axis=0)` → `std_imp`，形状 `(6,)`。
3. **计算 CV**：`std_imp / (mean_imp + 1e-8)` → `cv_imp`，形状 `(6,)`。
4. **计算 CI 下界**：`np.percentile(..., 2.5, axis=0)` → `ci_lower`，形状 `(6,)`。
5. **计算 CI 上界**：`np.percentile(..., 97.5, axis=0)` → `ci_upper`，形状 `(6,)`。
6. **计算 CI 宽度**：`ci_upper - ci_lower` → `ci_width`，形状 `(6,)`。
7. **计算排序索引**：`np.argsort(mean_imp)[::-1]` → `sorted_idx`，形状 `(6,)`。
8. **计算基准排名**：`np.argsort(mean_imp)[::-1]` → `base_ranking`。
9. **遍历每次 Bootstrap**：计算 `boot_ranking` 和 `spearmanr(base_ranking, boot_ranking)` → `rank_correlations`，长度 30。
10. **打印表格**：按重要性降序打印每个特征的 Mean、Std、CV、CI\_Width。
11. **打印排名稳定性**：平均 ρ、ρ 范围。

执行完毕后，得到 5 个核心统计量数组（`mean_imp`、`std_imp`、`cv_imp`、`ci_lower/upper/width`）和 1 个排名相关性列表（`rank_correlations`），为模块 3 的可视化提供数据。

***

## 小贴士

> 💡 **小贴士 1：CV 的"分母陷阱"**
>
> CV = std/mean 的一个陷阱是：**当 mean 很小时，CV 会自动变大**，即使 std 本身不大。
>
> 比如 `Code.Profession`：Mean=0.024, Std=0.009。Std 看起来很小（不到 0.01），但因为 Mean 也很小（0.024），CV = 0.009/0.024 = 0.395，被"放大"了。
>
> **解读建议**：**同时看 Mean 和 CV**，不要只看 CV。如果 Mean 很低（如 < 0.05），高 CV 可能只是"分母效应"，不一定意味着"真的不稳定"。

> 💡 **小贴士 2：百分位法 vs 正态法的选择**
>
> 本教程用百分位法计算 CI。什么时候用正态法？
>
> - **正态法**：当 Bootstrap 估计的分布近似正态时（如样本量大、重要性不是极端值），正态法更高效（用 mean ± 1.96×std）。
> - **百分位法**：当分布偏斜或有离群点时，百分位法更稳健。
>
> SHAP 重要性通常右偏（有少数高值），所以百分位法更合适。如果你不确定，**两种都算**，比较结果。如果两者接近，说明分布近似正态；如果差异大，用百分位法。

> 💡 **小贴士 3：Spearman ρ 的"标度问题"**
>
> 本教程只有 6 个特征，排名空间很小（6! = 720 种可能）。在这种小标度下，Spearman ρ 可能"不够敏感"：
>
> - 交换两个相邻特征（如第 3 和第 4），ρ 下降很少。
> - 交换首尾特征（第 1 和第 6），ρ 大幅下降。
>
> 对于更多特征（如 20 个），Spearman ρ 更敏感。本教程的 6 个特征场景下，**同时报告完整排名分布**比只看 ρ 更有信息量。

***

## 常见问题

> ❓ **Q1：为什么 CV 的判断标准是 0.1 和 0.3？**
>
> **A**：这是经验法则，没有严格的理论依据，但在实践中广泛使用：
>
> - **CV < 0.1**：波动幅度小于均值的 10%，通常认为"很小"。
> - **CV > 0.3**：波动幅度超过均值的 30%，通常认为"很大"。
> - **0.1–0.3**：中间地带，需要结合领域知识判断。
>
> 不同领域可能有不同标准。在生物医学中，CV < 0.15 有时也被认为"可接受"。本教程用 0.1/0.3 是相对严格的标准。

> ❓ **Q2：`np.percentile`** **的线性插值是怎么做的？**
>
> **A**：对于 30 个数据点和 P2.5：
>
> 1. 排序后，位置 = (30-1) × 2.5/100 = 0.725（第 0.725 个，介于第 0 和第 1 个之间）。
> 2. 用线性插值：`value = arr[0] + 0.725 × (arr[1] - arr[0])`。
>
> 对于 P97.5：
>
> 1. 位置 = (30-1) × 97.5/100 = 28.275（第 28.275 个，介于第 28 和第 29 个之间）。
> 2. 用线性插值：`value = arr[28] + 0.275 × (arr[29] - arr[28])`。
>
> NumPy 默认用线性插值，可以通过 `interpolation` 参数改为其他方法（如 `'lower'`、`'higher'`、`'nearest'`）。

> ❓ **Q3：Spearman ρ = 1.0 是不是太完美了？**
>
> **A**：不是"太完美"，而是"排名完全一致"。本教程中，30 次迭代中有 15+ 次的排名与平均排名完全相同（ρ=1.0），这是因为：
>
> 1. **6 个特征的重要性梯度明显**：year (0.225) >> Extension (0.133) >> Raca.Color (0.072) > ...，相邻特征差距大。
> 2. **小扰动不足以改变排名**：Bootstrap 引入的波动（Std ≈ 0.01）远小于相邻特征的差距（如 year 和 Extension 差 0.09）。
>
> 只有当波动大到能"翻转"相邻特征时，ρ 才会下降。最小 ρ=0.4857 的那次迭代，可能是多个中间特征的排名同时翻转。

> ❓ **Q4：`1e-8`** **会不会影响 CV 的计算精度？**
>
> **A**：不会。本教程中 Mean 的最小值是 0.024（Code.Profession），远大于 1e-8。所以 `mean + 1e-8 ≈ mean`，对 CV 的影响可以忽略。
>
> 只有当 Mean 真的接近 0（如 1e-10）时，`1e-8` 才会显著改变结果。但那种情况下，特征本身就没有重要性，CV 的意义也不大。

> ❓ **Q5：为什么代码中** **`rs`** **变量计算有问题但不影响结果？**
>
> **A**：`rs` 变量的本意可能是"该特征在多少次迭代中排名第一"，但代码写错了（两边表达式相同，永远是 True）。不过 `rs` **没有被打印**（打印时用了 `'-'` 占位），所以不影响最终输出。
>
> 真正的排名稳定性是用 `rank_correlations`（Spearman ρ 列表）计算的，这部分代码是正确的。`rs` 是一个"遗留代码"，可以忽略。

> ❓ **Q6：如果我想计算"每个特征在 30 次迭代中的排名序列"，怎么做？**
>
> **A**：可以用以下代码：
>
> ```python
> # 每次迭代中，每个特征的排名（1=最重要，6=最不重要）
> rank_sequences = []
> for boot_imp in bootstrap_shap_importance:
>     # argsort 两次得到排名
>     ranks = np.argsort(np.argsort(boot_imp)[::-1]) + 1  # 1-based
>     rank_sequences.append(ranks)
> rank_sequences = np.array(rank_sequences)  # 形状 (30, 6)
>
> # 每个特征的排名分布
> for i, fn in enumerate(feature_names):
>     ranks = rank_sequences[:, i]
>     print(f"{fn}: 排名分布 = {np.bincount(ranks)} (1-6 名的次数)")
> ```
>
> 这样可以看到每个特征在 30 次迭代中"排第几名"的分布，比单一的 ρ 更详细。

***

## 本模块小结

本模块完成了 SHAP 稳定性 Bootstrap 分析的**统计量化**：

1. **计算了 5 个稳定性指标**：
   - Mean SHAP：平均重要性（点估计）。
   - Std SHAP：波动程度（有量纲）。
   - **CV**：相对波动量（无量纲），核心稳定性指标。
   - CI Width：95% 置信区间宽度（百分位法）。
   - Rank ρ：Spearman 排名相关性（整体排名稳定性）。
2. **掌握了 CV 的判断标准**：<0.1 高度稳定，0.1–0.3 中等，>0.3 不稳定。
3. **理解了百分位法 CI**：用 P2.5 和 P97.5 作为区间端点，不假设正态分布。
4. **理解了 Spearman 排名相关性**：比较单次排名与平均排名的一致性。
5. **得到了实际运行结果**：
   - `year` 和 `Extension` 高度稳定（CV < 0.1）。
   - `Raca.Color`、`Diagnostic.means`、`Age` 中等稳定（0.1 ≤ CV < 0.3）。
   - `Code.Profession` 不稳定（CV = 0.395）。
   - 平均 Spearman ρ = 0.9067，整体排名稳定。
6. **掌握了解读框架**："重要性-CV"二维框架，避免"高 CV 但低重要性"的解读陷阱。

** **

***

**   **
