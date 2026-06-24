# 模块 1：Pipeline 构建与 CV 评估框架

> 本模块是案例教程 9「机器学习建模 — 7 模型对比」的第二部分，承接模块 0（数据加载与预处理）。在定义 7 个模型之前，我们必须先建立两个核心函数：**`build_pipeline`** 用于统一所有模型的预处理流程（防泄漏），**`evaluate_model_cv`** 用于用 Stratified 5-Fold CV 稳健评估模型性能。
>
> 本模块最核心的知识点有三个：**一是 Pipeline 如何防止数据泄漏**——所有预处理（插补、标准化）都在每折训练集上 fit、在验证集上 transform，杜绝测试集信息泄露；**二是 Stratified 5-Fold CV 的实现细节**——如何用 `cv.split(X, y)` 遍历每一折，如何在每折上记录 AUC/Recall/Brier/Time；**三是不同模型的概率输出差异**——SVM 的 `LinearSVC` 没有 `predict_proba`，需要用 `decision_function` + Sigmoid 转换得到概率。

***

## 学习目标

学完本模块后，你将能够：

1. **理解** **`build_pipeline`** **函数的设计**：为什么用 Pipeline 而不是手动预处理，`scale_needed` 参数如何控制是否标准化。
2. **掌握 Pipeline 防泄漏的原理**：明白为什么"在每折训练集上 fit、在验证集上 transform"能杜绝数据泄漏，对比"先预处理再划分"的错误做法。
3. **理解** **`evaluate_model_cv`** **函数的完整流程**：从 `StratifiedKFold` 初始化到 5 折循环，从训练计时到指标计算。
4. **掌握 SVM 概率输出的特殊处理**：`LinearSVC` 没有 `predict_proba`，需要用 `decision_function` + Sigmoid 转换，理解为什么这样转换。
5. **理解 AUC、Recall、Brier Score 三个指标的计算方式**：分别用 `roc_auc_score`、`recall_score`、`brier_score_loss`，以及它们的参数含义。
6. **理解** **`pos_label=1`** **的含义**：为什么 Recall 要指定正类为 1（VIVO）。
7. **掌握评估结果字典的结构**：理解返回的字典包含哪些字段，为什么同时保存 `auc_mean` 和 `auc_std`。
8. **理解** **`time.time()`** **计时的方式**：为什么只计 `pipe.fit` 的时间，不包括预测时间。

***

## 一、`build_pipeline` 函数：统一预处理流程

```python
# ============================================================================
# 1. 定义模型 & 评估函数
# ============================================================================
def build_pipeline(model, scale_needed=True):
    """统一的 Pipeline 构建"""
    steps = []
    steps.append(('imputer', SimpleImputer(strategy='median')))
    if scale_needed:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)
```

### 1.1 函数签名与参数

```python
def build_pipeline(model, scale_needed=True):
```

- `model`：一个 sklearn 分类器实例（如 `LogisticRegression()`、`RandomForestClassifier()`）。
- `scale_needed`：是否需要标准化，默认 `True`。树模型（DT/RF/XGBoost/LightGBM）传 `False`，因为它们对特征尺度不敏感。

### 1.2 `steps = []` — 初始化步骤列表

Pipeline 由一系列步骤（step）组成，每个步骤是一个 `(名称, 转换器/模型)` 的元组。`steps` 列表按顺序存储这些元组。

### 1.3 `steps.append(('imputer', SimpleImputer(strategy='median')))` — 中位数插补

- `'imputer'`：步骤名称，自定义字符串，后续可以用 `pipe.named_steps['imputer']` 访问。
- `SimpleImputer(strategy='median')`：中位数插补器，用该列的中位数填充缺失值。

**为什么用中位数而不是均值？**

> 💡 **重点概念：中位数 vs 均值插补**
>
> - **均值插补**（`strategy='mean'`）：用该列的均值填充缺失值。均值对异常值敏感——一个极端值就能把均值拉偏。
> - **中位数插补**（`strategy='median'`）：用该列的中位数填充缺失值。中位数是鲁棒统计量，不受少数极端值影响。
>
> 本教程的 `Code.Profession` 和 `Code.of.Morphology` 取值范围高达 0–9999，可能存在极端值。用中位数插补更稳健。
>
> 对于分类特征（编码后是小整数），中位数和均值差别不大；对于连续特征（Age），中位数更抗异常值（如 200 岁的录入错误）。

### 1.4 `if scale_needed: steps.append(('scaler', StandardScaler()))` — 条件标准化

- `'scaler'`：步骤名称。
- `StandardScaler()`：标准化器，把特征缩放成均值 0、标准差 1。
- `if scale_needed`：只有需要标准化的模型才添加这一步。

**哪些模型需要标准化？**

| 模型                  | `scale_needed`     | 原因                           |
| ------------------- | ------------------ | ---------------------------- |
| Logistic Regression | `True`             | 梯度下降对量纲敏感                    |
| SVM (Linear)        | `True`             | 间隔最大化对量纲敏感                   |
| KNN                 | `True`             | 距离计算对量纲敏感                    |
| Decision Tree       | `False`            | 树分裂基于阈值，与量纲无关                |
| Random Forest       | `False`            | 同上                           |
| XGBoost             | `True`（本教程设为 True） | 实际上 XGBoost 对尺度不敏感，但标准化不影响性能 |
| LightGBM            | `True`（本教程设为 True） | 同上                           |

> 💡 **小贴士**：XGBoost 和 LightGBM 实际上对特征尺度不敏感（树分裂基于阈值）。本教程把它们设为 `scale_needed=True` 是为了统一流程，标准化不会影响它们的性能。代码注释也说明了这一点："XGBoost 对特征尺度不敏感, 标准化不影响性能"。

### 1.5 `steps.append(('model', model))` — 添加模型

- `'model'`：步骤名称。
- `model`：传入的分类器实例。

**注意**：模型必须是 Pipeline 的最后一个步骤，因为只有最后一步可以是 estimator（分类器/回归器），前面的步骤必须是 transformer（转换器）。

### 1.6 `return Pipeline(steps)` — 构造 Pipeline

`Pipeline(steps)` 把步骤列表串成一条流水线。调用 `pipe.fit(X, y)` 时，数据会依次经过 imputer → scaler → model，每一步都 fit\_transform（最后一步只 fit）。

### 1.7 Pipeline 防泄漏的原理（重点）

> 💡💡💡 **重点概念：Pipeline 如何防止数据泄漏？**
>
> **错误做法（会泄漏）**：
>
> ```python
> # 先在全量数据上预处理，再划分
> X_imputed = SimpleImputer().fit_transform(X)  # 用了全量数据的统计量
> X_scaled = StandardScaler().fit_transform(X_imputed)  # 用了全量数据的统计量
> X_train, X_test = train_test_split(X_scaled)  # 测试集的统计量已经泄露到训练集
> ```
>
> **正确做法（Pipeline，不泄漏）**：
>
> ```python
> pipe = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler()), ('model', LR())])
> for tr_idx, te_idx in cv.split(X, y):
>     pipe.fit(X[tr_idx], y[tr_idx])  # 只在训练折上 fit
>     # 内部：imputer 在 X[tr_idx] 上 fit，scaler 在 imputer.transform(X[tr_idx]) 上 fit
>     y_pred = pipe.predict(X[te_idx])  # 在验证折上 transform + predict
>     # 验证折的统计量没有参与 fit，无泄漏
> ```
>
> Pipeline 的关键在于：**fit 时只在训练数据上学习参数，transform 时应用到验证数据上**。这样验证集的统计量（均值、中位数、标准差）不会泄露到训练集。
>
> 这是案例教程 7（数据泄漏）的核心教训，本教程严格遵守。

***

## 二、`evaluate_model_cv` 函数：CV 评估框架

```python
def evaluate_model_cv(X, y, model_name, model, scale_needed=True, n_splits=5):
    """用 Stratified K-Fold 评估模型性能与稳定性"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(model, scale_needed)

    aucs = []
    recalls = []
    briers = []
    times = []

    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        start_t = time.time()
        pipe.fit(X_tr, y_tr)
        elapsed = time.time() - start_t

        if scale_needed and hasattr(pipe.named_steps.get('model'), 'decision_function'):
            # SVM: use decision_function for AUC
            try:
                y_score = pipe.decision_function(X_te)
                y_prob = 1 / (1 + np.exp(-y_score))
            except:
                y_pred = pipe.predict(X_te)
                y_prob = np.clip(y_pred.astype(float), 0, 1)
        else:
            try:
                y_prob = pipe.predict_proba(X_te)[:, 1]
            except:
                y_pred = pipe.predict(X_te)
                y_prob = np.clip(y_pred.astype(float), 0, 1)

        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_te, y_prob)
        rec = recall_score(y_te, y_pred, pos_label=1)
        brier = brier_score_loss(y_te, y_prob)

        aucs.append(auc)
        recalls.append(rec)
        briers.append(brier)
        times.append(elapsed)

    return {
        'name': model_name,
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'brier_mean': np.mean(briers),
        'brier_std': np.std(briers),
        'time_mean': np.mean(times),
        'time_total': np.sum(times),
        'aucs': aucs,
        'recalls': recalls,
        'briers': briers,
    }
```

### 2.1 函数签名与参数

```python
def evaluate_model_cv(X, y, model_name, model, scale_needed=True, n_splits=5):
```

- `X`：特征矩阵（numpy 数组）。
- `y`：标签数组（numpy 数组）。
- `model_name`：模型名称字符串（如 `'Logistic Regression'`），用于结果展示。
- `model`：分类器实例。
- `scale_needed`：是否需要标准化，传给 `build_pipeline`。
- `n_splits`：CV 折数，默认 5。

### 2.2 `cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)` — 初始化分层 CV

- `n_splits=5`：5 折交叉验证。
- `shuffle=True`：划分前打乱数据顺序。如果不打乱，且数据按标签排序，会导致某些折全是正类、某些折全是负类。
- `random_state=RANDOM_STATE`：固定随机种子，确保划分可复现。

**为什么用 StratifiedKFold 而不是 KFold？**

> 💡 **重点概念：StratifiedKFold vs KFold**
>
> - **KFold**：随机划分成 K 折，不保证每折的类别比例一致。在不平衡数据上，可能导致某些折的少数类极少，模型无法学习。
> - **StratifiedKFold**：分层划分，保证每折的类别比例与整体一致。本数据集 VIVO 41.15%，每折也是约 41.15% VIVO。
>
> 案例教程 8 已验证，Stratified 5-Fold 是医学评估的标准方法。

### 2.3 `pipe = build_pipeline(model, scale_needed)` — 构建 Pipeline

调用前面定义的 `build_pipeline` 函数，构建包含插补、（标准化）、模型的 Pipeline。

**注意**：Pipeline 在循环外构建，但在循环内 `fit`。每次 `fit` 会重置 Pipeline 的内部参数，所以可以复用同一个 Pipeline 实例。

### 2.4 初始化指标列表

```python
aucs = []
recalls = []
briers = []
times = []
```

这四个列表分别记录每折的 AUC、Recall、Brier Score 和训练耗时。5 折后每个列表有 5 个元素。

### 2.5 `for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):` — 5 折循环

- `cv.split(X, y)`：生成交叉验证的索引，每次迭代返回 `(训练索引, 验证索引)`。
- `enumerate`：同时获取折序号 `fold_idx`（0–4）和索引。
- `tr_idx`：训练折的样本索引（约 16,000 个）。
- `te_idx`：验证折的样本索引（约 4,000 个）。

### 2.6 `X_tr, X_te = X[tr_idx], X[te_idx]` — 按索引取子集

- `X_tr`：训练折特征（约 16,000 × 9）。
- `X_te`：验证折特征（约 4,000 × 9）。
- `y_tr`、`y_te`：对应的标签。

### 2.7 训练计时

```python
start_t = time.time()
pipe.fit(X_tr, y_tr)
elapsed = time.time() - start_t
```

- `time.time()`：返回当前时间戳（秒）。
- `pipe.fit(X_tr, y_tr)`：在训练折上拟合 Pipeline（插补 + 标准化 + 模型）。
- `elapsed`：训练耗时（秒）。

**为什么只计** **`fit`** **时间，不包括** **`predict`** **时间？**

> 💡 **小贴士**：训练时间（fit）通常远大于预测时间（predict），是模型计算成本的主要部分。本教程关注"训练成本"，因为训练是离线的一次性开销，而预测是在线的高频开销。如果要评估部署成本，应该单独计 predict 时间。
>
> 实际上，KNN 的 predict 时间可能比 fit 长（因为 KNN 是"懒学习"，predict 时才计算距离），但本教程统一只计 fit 时间，保持公平。

### 2.8 概率输出的特殊处理（重点）

```python
if scale_needed and hasattr(pipe.named_steps.get('model'), 'decision_function'):
    # SVM: use decision_function for AUC
    try:
        y_score = pipe.decision_function(X_te)
        y_prob = 1 / (1 + np.exp(-y_score))
    except:
        y_pred = pipe.predict(X_te)
        y_prob = np.clip(y_pred.astype(float), 0, 1)
else:
    try:
        y_prob = pipe.predict_proba(X_te)[:, 1]
    except:
        y_pred = pipe.predict(X_te)
        y_prob = np.clip(y_pred.astype(float), 0, 1)
```

这段代码处理不同模型的概率输出差异，是本函数最复杂的部分。

#### 2.8.1 为什么需要特殊处理？

不同模型提供不同的概率输出方法：

| 模型                  | `predict_proba` | `decision_function` | 本教程处理方式                       |
| ------------------- | --------------- | ------------------- | ----------------------------- |
| Logistic Regression | ✅               | ✅                   | `predict_proba`               |
| SVM (LinearSVC)     | ❌               | ✅                   | `decision_function` + Sigmoid |
| KNN                 | ✅               | ❌                   | `predict_proba`               |
| Decision Tree       | ✅               | ❌                   | `predict_proba`               |
| Random Forest       | ✅               | ❌                   | `predict_proba`               |
| XGBoost             | ✅               | ✅                   | `predict_proba`               |
| LightGBM            | ✅               | ✅                   | `predict_proba`               |

**关键问题**：`LinearSVC` 没有 `predict_proba` 方法（只有 `SVC` 才有，但 `SVC` 在大样本上太慢）。我们需要用 `decision_function` + Sigmoid 转换得到概率。

#### 2.8.2 `hasattr(pipe.named_steps.get('model'), 'decision_function')` — 检查是否有 decision\_function

- `pipe.named_steps`：Pipeline 中所有步骤的字典，键是步骤名称。
- `pipe.named_steps.get('model')`：获取 `'model'` 步骤（即分类器实例）。
- `hasattr(obj, 'decision_function')`：检查对象是否有 `decision_function` 方法。

**为什么用** **`scale_needed`** **作为前置条件？**

代码用 `if scale_needed and hasattr(...)` 作为判断。这是一个启发式简化：本教程中只有 SVM 同时满足"需要标准化"和"有 decision\_function"。其他需要标准化的模型（LR/KNN）虽然有 `decision_function`（LR 有，KNN 没有），但它们也有 `predict_proba`，走 `else` 分支用 `predict_proba` 更准确。

> ⚠️ **注意**：这个判断逻辑有点取巧。更严谨的写法应该是：
>
> ```python
> if hasattr(pipe, 'predict_proba'):
>     y_prob = pipe.predict_proba(X_te)[:, 1]
> elif hasattr(pipe, 'decision_function'):
>     y_score = pipe.decision_function(X_te)
>     y_prob = 1 / (1 + np.exp(-y_score))
> else:
>     y_pred = pipe.predict(X_te)
>     y_prob = np.clip(y_pred.astype(float), 0, 1)
> ```
>
> 但本教程的写法在实际运行中也能得到正确结果，因为只有 SVM 走第一个分支。

#### 2.8.3 `y_score = pipe.decision_function(X_te)` — SVM 的决策分数

`decision_function` 返回样本到决策超平面的有符号距离：

- 正值：预测为正类（VIVO）。
- 负值：预测为负类（MORTO）。
- 绝对值越大：置信度越高。

#### 2.8.4 `y_prob = 1 / (1 + np.exp(-y_score))` — Sigmoid 转换

把决策分数转换成 \[0, 1] 的概率：

$$p = \frac{1}{1 + e^{-s}}$$

其中 $s$ 是决策分数，$p$ 是概率。

**为什么用 Sigmoid？**

> 💡 **重点概念：Sigmoid 转换的原理**
>
> Sigmoid 函数把任意实数压缩到 (0, 1) 区间：
>
> - $s = 0$ → $p = 0.5$（决策边界）
> - $s \to +\infty$ → $p \to 1$（高置信度正类）
> - $s \to -\infty$ → $p \to 0$（高置信度负类）
>
> 逻辑回归本身就是用 Sigmoid 把线性组合 $w^T x + b$ 转成概率。SVM 的 `decision_function` 也是线性的（$w^T x + b$），用 Sigmoid 转换在数学上是合理的。
>
> **注意**：这种转换不是"校准"的概率——SVM 的 decision\_function 没有概率语义，Sigmoid 转换只是把分数压到 \[0, 1]。如果需要校准的概率，应该用 `CalibratedClassifierCV`（案例教程 14 会讨论）。但对于 AUC 计算，这种转换足够了，因为 AUC 只关心排序。

#### 2.8.5 `else` 分支：用 predict\_proba

```python
y_prob = pipe.predict_proba(X_te)[:, 1]
```

- `pipe.predict_proba(X_te)`：返回形状为 (n\_samples, 2) 的概率矩阵，第一列是负类概率，第二列是正类概率。
- `[:, 1]`：取第二列（正类 VIVO 的概率）。

#### 2.8.6 `except` 兜底

```python
except:
    y_pred = pipe.predict(X_te)
    y_prob = np.clip(y_pred.astype(float), 0, 1)
```

如果 `predict_proba` 或 `decision_function` 都失败（极端情况），用 `predict` 得到 0/1 预测，转成 float 后 clip 到 \[0, 1]。

- `np.clip(y_pred.astype(float), 0, 1)`：把 0/1 预测当作概率（0 或 1）。
- 这种兜底的概率是"硬预测"，AUC 会退化，但至少不报错。

### 2.9 `y_pred = (y_prob >= 0.5).astype(int)` — 阈值化得到类别预测

- `y_prob >= 0.5`：概率 ≥ 0.5 判为正类（VIVO），返回布尔数组。
- `.astype(int)`：转成 0/1 整数。

**为什么用 0.5 作为阈值？**

0.5 是默认阈值，对应"概率最大的类"。在平衡数据上这是合理的默认值。在不平衡数据上，可能需要调整阈值（如降低到 0.3 来提高 Recall），但本教程统一用 0.5，保持对比公平。

### 2.10 计算三个指标

```python
auc = roc_auc_score(y_te, y_prob)
rec = recall_score(y_te, y_pred, pos_label=1)
brier = brier_score_loss(y_te, y_prob)
```

#### 2.10.1 `roc_auc_score(y_te, y_prob)` — AUC

- `y_te`：真实标签（0/1）。
- `y_prob`：预测概率（\[0, 1] 的浮点数）。
- 返回 AUC 值（0–1）。

**AUC 的含义**：AUC 是 ROC 曲线下面积，衡量模型区分正负类的能力。AUC=1 完美，AUC=0.5 等同随机。AUC 只关心**排序**——把正类排在负类前面的能力，不关心绝对概率值。

**为什么 AUC 用** **`y_prob`** **而不是** **`y_pred`？**

AUC 需要连续的概率值来计算不同阈值下的 TPR/FPR。如果用 0/1 预测，AUC 会退化成只反映一个阈值的性能。

#### 2.10.2 `recall_score(y_te, y_pred, pos_label=1)` — Recall

- `y_te`：真实标签。
- `y_pred`：0/1 预测（不是概率）。
- `pos_label=1`：指定正类为 1（VIVO）。

**Recall 的含义**：Recall = TP / (TP + FN)，即"实际正类中被正确预测为正类的比例"。本教程中，Recall = "存活患者中被正确识别为存活的比例"。

**为什么 Recall 用** **`y_pred`** **而不是** **`y_prob`？**

Recall 是基于 0/1 预测的指标，需要先阈值化。如果用概率，需要指定阈值，而 `recall_score` 不接受概率。

**为什么** **`pos_label=1`？**

> 💡 **重点概念：`pos_label`** **的临床意义**
>
> sklearn 默认 `pos_label=1`，但显式写出更清晰。本教程把 VIVO（存活）映射为 1，所以 `pos_label=1` 表示"计算存活患者的 Recall"。
>
> 在医学场景中，"存活患者的检出率"（不漏诊存活患者）是临床关键指标——漏诊一个可能存活的患者，意味着放弃治疗机会。所以本教程重点关注 VIVO 的 Recall。

#### 2.10.3 `brier_score_loss(y_te, y_prob)` — Brier Score

- `y_te`：真实标签（0/1）。
- `y_prob`：预测概率。
- 返回 Brier Score（0–1，越低越好）。

**Brier Score 的含义**：Brier Score = $\frac{1}{N}\sum\_{i=1}^{N}(p\_i - y\_i)^2$，即预测概率与真实标签的均方误差。Brier=0 完美，Brier=1 最差。

**为什么 Brier Score 重要？**

> 💡 **重点概念：Brier Score 衡量概率校准**
>
> AUC 只关心排序，不关心绝对概率。但临床决策需要准确的概率——"这个患者存活概率是 80%"和"存活概率是 99%"对决策影响不同。
>
> Brier Score 衡量概率的准确性：
>
> - Brier 低 → 预测概率接近真实标签（校准好）
> - Brier 高 → 预测概率偏离真实标签（校准差）
>
> 本教程的实验结果显示，LightGBM 的 Brier=0.0974（最低），说明它的概率估计最准确。

### 2.11 记录每折指标

```python
aucs.append(auc)
recalls.append(rec)
briers.append(brier)
times.append(elapsed)
```

把每折的指标追加到列表。5 折后每个列表有 5 个元素。

### 2.12 返回结果字典

```python
return {
    'name': model_name,
    'auc_mean': np.mean(aucs),
    'auc_std': np.std(aucs),
    'recall_mean': np.mean(recalls),
    'recall_std': np.std(recalls),
    'brier_mean': np.mean(briers),
    'brier_std': np.std(briers),
    'time_mean': np.mean(times),
    'time_total': np.sum(times),
    'aucs': aucs,
    'recalls': recalls,
    'briers': briers,
}
```

返回的字典包含以下字段：

| 字段            | 含义            | 用途            |
| ------------- | ------------- | ------------- |
| `name`        | 模型名称          | 展示和绘图标签       |
| `auc_mean`    | AUC 均值        | 性能排名          |
| `auc_std`     | AUC 标准差       | 稳定性排名         |
| `recall_mean` | Recall 均值     | 少数类识别能力       |
| `recall_std`  | Recall 标准差    | Recall 稳定性    |
| `brier_mean`  | Brier 均值      | 概率校准质量        |
| `brier_std`   | Brier 标准差     | Brier 稳定性     |
| `time_mean`   | 平均每折耗时        | 单次训练成本        |
| `time_total`  | 5 折总耗时        | 总训练成本（用于速度排名） |
| `aucs`        | 5 折 AUC 列表    | 绘制 5 折波动图     |
| `recalls`     | 5 折 Recall 列表 | 扩展用           |
| `briers`      | 5 折 Brier 列表  | 扩展用           |

**为什么同时保存** **`mean`** **和** **`std`？**

- `mean`：性能的**点估计**，反映模型在该数据集上的平均表现。
- `std`：性能的**不确定性**，反映模型跨折的稳定性。

> 💡 **重点概念：均值 ± 标准差的解读**
>
> 实验结果通常表示为 `AUC = 0.9423 ± 0.0016`，含义是：
>
> - 5 折 AUC 的均值是 0.9423
> - 5 折 AUC 的标准差是 0.0016
> - 大约 68% 的折的 AUC 落在 \[0.9407, 0.9439] 区间
>
> 标准差越小，模型越稳定。本教程中 KNN 的 σ=0.0012（最稳定），LR/SVM 的 σ=0.0033（最不稳定）。

**为什么保存原始列表** **`aucs`/`recalls`/`briers`？**

原始列表用于绘制 5 折波动图（12f\_fold\_variance.png），展示每折的 AUC 分布。这种可视化能直观看到模型的稳定性。

***

## 三、评估框架的五个维度

本教程的评估框架包含五个维度，每个维度回答一个不同的问题：

| 维度    | 指标     | 回答的问题         | 本教程结果范围       |
| ----- | ------ | ------------- | ------------- |
| 性能    | AUC    | 模型区分正负类的能力如何？ | 0.8936–0.9423 |
| 少数类识别 | Recall | 存活患者的检出率如何？   | 0.8335–0.8984 |
| 概率校准  | Brier  | 预测概率准确吗？      | 0.0974–0.1347 |
| 计算成本  | Time   | 训练要多久？        | 0.06s–64.64s  |
| 稳定性   | σ(AUC) | 跨折表现稳定吗？      | 0.0012–0.0033 |

> 💡💡💡 **重点概念：为什么需要五个维度？**
>
> 单一指标会误导模型选择：
>
> - 只看 AUC：会选 LightGBM（0.9423），但忽略它训练慢（64.64s）。
> - 只看 Recall：会选 SVM/RF（0.8984），但 SVM 的 AUC 只有 0.8946。
> - 只看 Brier：会选 LightGBM（0.0974），但和 XGBoost（0.0982）差距很小。
> - 只看速度：会选 LR/KNN（0.06s），但它们的 AUC 最低。
> - 只看稳定性：会选 KNN（σ=0.0012），但它的 Recall 最低。
>
> **五个维度联合解读**才能做出合理的模型选择。这是本教程的核心教学目标——**没有"最好"的模型，只有"最适合目标"的模型**。

***

## 小贴士

1. **Pipeline 是防泄漏的关键**：所有预处理都在每折训练集上 fit、在验证集上 transform，杜绝测试集信息泄露。
2. **`scale_needed`** **参数控制是否标准化**：树模型传 `False`，线性模型和距离模型传 `True`。
3. **中位数插补比均值插补更鲁棒**：不受极端值影响，适合有异常值的数据。
4. **`StratifiedKFold`** **是不平衡数据的必备**：保证每折的类别比例与整体一致。
5. **SVM 的** **`LinearSVC`** **没有** **`predict_proba`**：需要用 `decision_function` + Sigmoid 转换得到概率。
6. **Sigmoid 转换不是校准**：只是把分数压到 \[0, 1]，AUC 够用但概率不一定准确。
7. **AUC 用概率，Recall 用 0/1 预测**：AUC 关心排序（需要连续概率），Recall 关心分类（需要阈值化）。
8. **`pos_label=1`** **指定正类**：本教程把 VIVO 映射为 1，Recall 衡量"存活患者的检出率"。
9. **均值 ± 标准差是标准报告格式**：均值反映性能，标准差反映稳定性。
10. **五个维度联合解读**：AUC + Recall + Brier + Time + σ，缺一不可。

***

## 常见问题

**Q1：为什么 Pipeline 在循环外构建，但在循环内 fit？**

A：Pipeline 实例可以复用，每次 `fit` 会重置内部参数（imputer 的中位数、scaler 的均值/标准差、模型的权重）。在循环外构建可以避免重复创建对象的开销，代码也更简洁。但要注意：不能在循环外 `fit`，否则会用全量数据学习预处理参数，导致泄漏。

**Q2：`StratifiedKFold`** **的** **`shuffle=True`** **和** **`random_state=42`** **有什么作用？**

A：`shuffle=True` 在划分前打乱数据顺序，避免数据按标签排序时某些折全是同一类。`random_state=42` 固定打乱的随机种子，确保划分可复现——任何人运行这段代码都得到相同的 5 折划分。

**Q3：为什么 SVM 用** **`decision_function`** **+ Sigmoid，而不是用** **`SVC(probability=True)`？**

A：`SVC(probability=True)` 内部用 Platt scaling（5 折内部 CV 拟合一个逻辑回归来校准概率），计算开销大，且 `SVC` 在 20,000 样本上比 `LinearSVC` 慢得多。本教程用 `LinearSVC` + Sigmoid 是性能和速度的折中——Sigmoid 转换虽然不是严格校准，但对 AUC 计算足够。

**Q4：`y_pred = (y_prob >= 0.5).astype(int)`** **中的 0.5 阈值合理吗？**

A：在平衡数据上 0.5 是合理的默认值。在不平衡数据上，可能需要调整阈值。但本教程统一用 0.5，保持 7 个模型的对比公平。如果要优化某个模型的 Recall，可以单独调整阈值（如降到 0.3），但这超出了本教程的范围。

**Q5：为什么** **`time.time()`** **只计** **`fit`** **时间，不包括** **`predict`** **时间？**

A：训练时间（fit）通常远大于预测时间（predict），是模型计算成本的主要部分。本教程关注"训练成本"。实际上 KNN 的 predict 时间可能比 fit 长（懒学习），但统一只计 fit 时间保持公平。

**Q6：`np.std(aucs)`** **计算的是样本标准差还是总体标准差？**

A：numpy 默认 `ddof=0`，计算的是**总体标准差**（除以 N）。如果要样本标准差（除以 N-1），需要 `np.std(aucs, ddof=1)`。本教程用默认的总体标准差，5 折时差别很小（除以 5 vs 除以 4）。

**Q7：`hasattr(pipe.named_steps.get('model'), 'decision_function')`** **这个判断为什么用** **`scale_needed`** **作为前置条件？**

A：这是一个启发式简化。本教程中只有 SVM 同时满足"需要标准化"和"有 decision\_function 但没有 predict\_proba"。其他需要标准化的模型（LR/KNN）虽然有 decision\_function（LR 有），但它们也有 predict\_proba，走 `else` 分支用 predict\_proba 更准确。这个写法有点取巧，但实际运行正确。

**Q8：如果某个模型在某一折训练失败（如不收敛），会怎样？**

A：本教程用 `try/except` 兜底，如果 `predict_proba` 或 `decision_function` 失败，用 `predict` 得到 0/1 预测当作概率。这样即使模型不收敛，也不会中断整个评估流程。但 AUC 会退化（因为 0/1 预测的 AUC 只反映一个阈值）。

**Q9：为什么返回字典里同时有** **`time_mean`** **和** **`time_total`？**

A：`time_mean` 是平均每折耗时，反映单次训练成本；`time_total` 是 5 折总耗时，反映 CV 评估的总成本。速度排名用 `time_total`（因为 5 折是固定的评估框架），单次成本比较用 `time_mean`。

**Q10：Brier Score 越低越好，那 0.0974（LightGBM）算好还是差？**

A：Brier Score 的绝对值没有统一标准，取决于数据集的难度。一般来说：

- Brier < 0.1：校准较好
- Brier 0.1–0.2：校准一般
- Brier > 0.2：校准较差

本教程中 LightGBM 的 0.0974 是最好的，LR 的 0.1310 是最差的之一。但所有模型的 Brier 都在 0.1–0.13 之间，差距不大，说明校准质量整体可以接受。

***

## 本模块小结

### 核心知识点回顾

1. **`build_pipeline`** **函数**：
   - 统一所有模型的预处理流程：`SimpleImputer(median) → [StandardScaler] → model`
   - `scale_needed` 参数控制是否标准化（树模型传 `False`）
   - Pipeline 防泄漏：每折独立 fit\_transform
2. **`evaluate_model_cv`** **函数**：
   - 用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 做 5 折 CV
   - 每折记录 AUC、Recall、Brier、Time
   - 返回包含 mean/std/原始列表的字典
3. **Pipeline 防泄漏的原理**：
   - 错误做法：先在全量数据上预处理，再划分（测试集统计量泄露）
   - 正确做法：Pipeline 在每折训练集上 fit，在验证集上 transform
4. **SVM 概率输出的特殊处理**：
   - `LinearSVC` 没有 `predict_proba`
   - 用 `decision_function` + Sigmoid 转换：`p = 1 / (1 + exp(-s))`
   - Sigmoid 转换不是严格校准，但 AUC 计算够用
5. **三个指标的计算**：
   - AUC：`roc_auc_score(y_te, y_prob)`，用概率，衡量排序能力
   - Recall：`recall_score(y_te, y_pred, pos_label=1)`，用 0/1 预测，衡量存活患者检出率
   - Brier：`brier_score_loss(y_te, y_prob)`，用概率，衡量校准质量
6. **五个评估维度**：
   - 性能（AUC）、少数类识别（Recall）、概率校准（Brier）、计算成本（Time）、稳定性（σ）
   - 五个维度联合解读，缺一不可

### 评估框架结构

```
evaluate_model_cv(X, y, model_name, model, scale_needed, n_splits=5)
    │
    ├── cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ├── pipe = build_pipeline(model, scale_needed)
    │
    └── for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            ├── X_tr, X_te = X[tr_idx], X[te_idx]
            ├── start_t = time.time()
            ├── pipe.fit(X_tr, y_tr)          # 训练（含预处理）
            ├── elapsed = time.time() - start_t
            │
            ├── 获取概率:
            │   ├── SVM: decision_function + Sigmoid
            │   └── 其他: predict_proba[:, 1]
            │
            ├── y_pred = (y_prob >= 0.5).astype(int)
            ├── auc = roc_auc_score(y_te, y_prob)
            ├── rec = recall_score(y_te, y_pred, pos_label=1)
            ├── brier = brier_score_loss(y_te, y_prob)
            │
            └── 追加到 aucs/recalls/briers/times 列表
    │
    └── 返回 {name, auc_mean, auc_std, recall_mean, ..., aucs, recalls, briers}
```

### 下一部分预告

本模块完成了 Pipeline 构建和 CV 评估框架，得到了一个通用的评估函数 `evaluate_model_cv`。接下来的模块 2 将定义 7 个模型：

**模块 2：七个模型定义详解**

- 逐个解释 7 个模型的参数和原理
- 包括 LR、SVM、KNN、DT、RF、XGBoost、LightGBM
- 讨论 `class_weight='balanced'` 和 `scale_pos_weight` 的区别
- 解释 `n_estimators=200`、`max_depth=10`、`C=1.0` 等参数的含义

让我们带着"如何统一评估 7 个不同模型"的框架，进入模块 2 的模型定义！

***

