# 模块 3：构造前后对比实验 — 特征构造的效果评估

> 本模块是案例教程 4「特征工程」的最后一个核心模块。在模块 1 中我们对比了三种标准化方法，得出"标准化不改变性能但加速收敛"的结论；在模块 2 中我们基于医学领域知识构造了 7 个新特征（`Age_Group`、`Age_Sq`、`Year_Decade`、`Year_From_2000`、`Gender_x_AgeGroup`、`Is_Child`、`Age_Centered`）。本模块将回答整个案例最关键的问题：**这些新特征到底有没有用？** 我们将通过严格的对比实验，把"基础 6 特征"和"工程化 13 特征"放进同一个模型、同一份数据划分、同一套评估指标里，看看特征构造带来的真实收益与代价。

---

## 📚 学习目标

完成本模块后，你将能够：

1. **理解对比实验的公平性原则**：掌握如何通过相同的 `random_state` 和 `stratify` 保证两个特征集的样本划分完全一致，从而隔离"特征数变化"这一唯一变量。
2. **掌握对多个特征集分别做预处理**：理解为什么对 `X_base_raw` 和 `X_all_raw` 要分别调用 `imputer.fit_transform` 和 `scaler.fit_transform`，而不是共享一套预处理参数。
3. **理解 Python 字典 + 元组的循环模式**：能用 `fe_sets = {'名称': (X_tr, X_te, n_feat)}` 这种结构优雅地组织多组实验。
4. **掌握 `_` 占位符的语义**：理解 `X_all_train, X_all_test, _, _ = train_test_split(...)` 中下划线变量表示"我故意丢弃这个返回值"。
5. **回顾并巩固 ROC 曲线与 PR 曲线**：理解 `roc_curve` 与 `precision_recall_curve` 的返回值结构，以及它们在不平衡数据下的差异。
6. **解读多指标对比柱状图**：能从 AUC、Recall、Brier、PR-AUC 四个子图中读出"哪些指标变好、哪些变差"。
7. **建立"多指标评估"思维**：理解为什么不能只看 AUC，特别是当 Recall 出现"反直觉下降"时如何分析根因。
8. **理解特征工程的权衡本质**：认识到"加特征"不是免费午餐——AUC 和 Brier 改善的代价可能是 Recall 下降。
9. **完成整个案例教程 4 的知识闭环**：从标准化（模块 1）到特征构造（模块 2）再到效果评估（模块 3），形成"特征质量决定模型上限"的完整认知。

> **前置知识**：模块 1（标准化比较）、模块 2（7 个新特征构造）、案例教程 3（逻辑回归与评估指标）、Python 字典与元组解包。

> **本模块对应代码**：`src/04_feature_engineering.py` 第 370–555 行。

---

## 一、代码总览

本模块的实验逻辑可以概括为以下流程：

```
┌──────────────────────────────────────────────────────────────────┐
│  1. 创建 SimpleImputer，对两个特征集分别做均值插补                  │
│     - X_base_raw  ← imputer.fit_transform(df_eng[base_features])  │
│     - X_all_raw   ← imputer.fit_transform(df_eng[all_feat_names]) │
│                                                                  │
│  2. 两次 train_test_split（相同 random_state + stratify）          │
│     - 保证 X_base 和 X_all 的样本划分完全一致                      │
│                                                                  │
│  3. 对两个特征集分别做 StandardScaler                             │
│     - X_base_train_s / X_base_test_s                             │
│     - X_all_train_s  / X_all_test_s                              │
│                                                                  │
│  4. 用 fe_sets 字典组织实验：                                     │
│     - 'Base (6 features)':       (X_base_train_s, X_base_test_s, 6) │
│     - 'Engineered (13 features)': (X_all_train_s,  X_all_test_s, 13) │
│                                                                  │
│  5. for 循环遍历 fe_sets：                                        │
│     a. 训练 LogisticRegression(class_weight='balanced')           │
│     b. 计算 AUC / Recall / Brier / PR-AUC                         │
│     c. 绘制 ROC 曲线、PR 曲线                                     │
│                                                                  │
│  6. 绘制 3 张图：                                                 │
│     - 图07e: ROC 曲线对比                                         │
│     - 图07f: PR 曲线对比                                          │
│     - 图07g: 性能对比柱状图（4 指标）                              │
│                                                                  │
│  7. 保存结果到 CSV 和 TXT                                         │
└──────────────────────────────────────────────────────────────────┘
```

下面我们逐步拆解每一段代码。

---

## 二、对两个特征集分别做均值插补

### 2.1 完整代码

```python
# 处理缺失
imputer_feat = SimpleImputer(strategy='mean')

# 基础特征集
X_base_raw = imputer_feat.fit_transform(df_eng[base_features])

# 全部特征集 (基础 + 构造)
all_feat_names = base_features + engineered_features
X_all_raw = imputer_feat.fit_transform(df_eng[all_feat_names])
```

### 2.2 为什么要对两个特征集分别插补？

回顾模块 2，我们构造了 7 个新特征并存入 `df_eng`。现在 `df_eng` 里同时包含：
- **6 个基础特征**：`['Age', 'year', 'Gender', 'Code.Profession', 'Diagnostic.means', 'Raca.Color']`
- **7 个构造特征**：`['Age_Group', 'Age_Sq', 'Year_Decade', 'Year_From_2000', 'Gender_x_AgeGroup', 'Is_Child', 'Age_Centered']`

为了对比"加特征前 vs 加特征后"，我们需要准备两个不同的特征矩阵：

| 变量名 | 来自列 | 形状 | 含义 |
|--------|--------|------|------|
| `X_base_raw` | `df_eng[base_features]` | `(N, 6)` | 仅含 6 个原始特征 |
| `X_all_raw` | `df_eng[all_feat_names]` | `(N, 13)` | 6 个原始 + 7 个构造 |

### 2.3 逐行解析

#### `imputer_feat = SimpleImputer(strategy='mean')`

创建一个均值插补器实例。`strategy='mean'` 表示用该列非缺失值的均值填充缺失值。

> 💡 **小贴士**：这里新建了一个 `imputer_feat` 而不是复用模块 1 的 `imputer`，是为了避免状态污染。每个 `SimpleImputer` 实例内部会保存它"见过"的均值，重复 `fit` 会覆盖旧值。新建实例是更安全的做法。

#### `X_base_raw = imputer_feat.fit_transform(df_eng[base_features])`

这一行做了两件事：
1. **`fit`**：计算 `df_eng[base_features]` 每一列的均值，存入 `imputer_feat.statistics_`。
2. **`transform`**：用这些均值填充缺失值，返回 NumPy 数组 `X_base_raw`。

此时 `imputer_feat.statistics_` 是一个长度为 6 的数组，对应 6 个基础特征的均值。

#### `all_feat_names = base_features + engineered_features`

Python 列表拼接。`base_features` 是 6 个原始特征，`engineered_features` 是 7 个构造特征，拼接后 `all_feat_names` 是长度为 13 的列表。

#### `X_all_raw = imputer_feat.fit_transform(df_eng[all_feat_names])`

**注意**：这里再次调用了 `fit_transform`！这会**覆盖**之前算的 6 个均值，重新计算 13 个特征的均值。

> ⚠️ **常见问题**：为什么不对 `X_base_raw` 和 `X_all_raw` 用同一套插补参数？
>
> 因为两个特征集的列数不同（6 vs 13），插补器内部存的 `statistics_` 数组长度必须与列数匹配。如果用 6 列的插补器去 transform 13 列的数据，会直接报维度错误。所以**必须分别 `fit`**。
>
> 更深层的原因：构造特征（如 `Age_Sq = Age ** 2`）的均值与原始 `Age` 的均值不同，需要独立计算。

### 2.4 插补后的数据形态

假设原始数据有 80,000 个样本：

```
X_base_raw.shape = (80000, 6)    # 6 个基础特征
X_all_raw.shape  = (80000, 13)   # 13 个特征（6 基础 + 7 构造）
```

两个矩阵的**行数完全相同**（同一批样本），只是列数不同。这为后续的"公平对比"奠定了基础。

---

## 三、两次 train_test_split 的一致性（**重点**）

### 3.1 完整代码

```python
# 划分
y_eng = df_eng['target'].values

X_base_train, X_base_test, y_eng_train, y_eng_test = train_test_split(
    X_base_raw, y_eng, test_size=0.3, random_state=RANDOM_STATE, stratify=y_eng
)
X_all_train, X_all_test, _, _ = train_test_split(
    X_all_raw, y_eng, test_size=0.3, random_state=RANDOM_STATE, stratify=y_eng
)
```

### 3.2 为什么这是本模块最关键的细节？

对比实验的核心原则是**控制变量**：我们只想观察"特征数从 6 变成 13"这一变量对模型性能的影响，必须排除其他干扰因素。其中最重要的干扰因素就是**样本划分不一致**——如果 Base 模型的测试集是样本 A、B、C，而 Engineered 模型的测试集是样本 D、E、F，那性能差异就说不清是特征带来的还是样本带来的。

### 3.3 保证一致性的两个关键参数

#### `random_state=RANDOM_STATE`

`RANDOM_STATE = 42`（在文件开头定义）。`train_test_split` 内部使用随机数生成器打乱样本顺序，`random_state` 是随机种子。**相同的种子 + 相同的输入长度 = 相同的打乱顺序**。

由于 `X_base_raw` 和 `X_all_raw` 的行数都是 80,000，传入相同的 `y_eng`，所以两次划分的**打乱索引完全相同**。

#### `stratify=y_eng`

分层抽样：按 `y_eng` 的类别比例划分，保证训练集和测试集中 VIVO/MORTO 的比例与原始数据一致。

两次调用都传入**同一个 `y_eng`**，所以分层逻辑也完全一致。

### 3.4 两次划分的样本对应关系

经过上述两次 `train_test_split`，我们得到：

| 变量 | 形状 | 含义 |
|------|------|------|
| `X_base_train` | `(56000, 6)` | Base 训练集 |
| `X_base_test` | `(24000, 6)` | Base 测试集 |
| `X_all_train` | `(56000, 13)` | Engineered 训练集 |
| `X_all_test` | `(24000, 13)` | Engineered 测试集 |
| `y_eng_train` | `(56000,)` | 训练标签 |
| `y_eng_test` | `(24000,)` | 测试标签 |

**关键性质**：`X_base_train[i]` 和 `X_all_train[i]` 对应**同一个样本**——前者只取了该样本的 6 个基础特征，后者取了全部 13 个特征。测试集同理。

这样，两个模型在测试集上面对的是**完全相同的样本**，只是特征数不同。性能差异只能归因于特征构造。

### 3.5 `_` 占位符的语义

```python
X_all_train, X_all_test, _, _ = train_test_split(
    X_all_raw, y_eng, test_size=0.3, random_state=RANDOM_STATE, stratify=y_eng
)
```

`train_test_split` 返回 4 个值：`X_train, X_test, y_train, y_test`。在第二次调用时，我们只需要 `X_all_train` 和 `X_all_test`，不需要 `y_train` 和 `y_test`——因为它们和第一次调用返回的 `y_eng_train`、`y_eng_test` 完全相同（同样的 `y_eng`、同样的 `random_state`、同样的 `stratify`）。

Python 约定用 `_` 表示"我故意丢弃这个返回值"。这是一种**语义信号**：告诉读代码的人"这个值不重要，不要在这里找 bug"。

> 💡 **小贴士**：`_` 在 Python 交互环境中还有"上一个表达式结果"的特殊含义，但在赋值语句中它就是一个普通变量名。约定俗成地用它接收不需要的值，是 Pythonic 的写法。

### 3.6 如果不用相同的 random_state 会怎样？

假设第二次调用写成 `random_state=123`，那么：
- 打乱顺序不同 → 测试集样本不同
- 即使 `stratify=y_eng` 保证了类别比例，具体哪些样本进测试集也不同
- 两个模型的性能差异会被"样本差异"污染，无法归因于特征构造

这就是为什么对比实验必须严格控制 `random_state`。

---

## 四、对两个特征集分别做 StandardScaler

### 4.1 完整代码

```python
# 标准化
scaler_std = StandardScaler()
X_base_train_s = scaler_std.fit_transform(X_base_train)
X_base_test_s = scaler_std.transform(X_base_test)
X_all_train_s = scaler_std.fit_transform(X_all_train)
X_all_test_s = scaler_std.transform(X_all_test)
```

### 4.2 为什么要分别标准化？

回顾模块 1 的结论：**标准化不改变模型性能，但能加速收敛并让系数可比**。本模块为了与模块 1 保持一致（且让 13 个特征的系数可比），对两个特征集都做 StandardScaler。

### 4.3 为什么不能共享一套标准化参数？

和插补器一样，`StandardScaler` 内部存了 `mean_` 和 `scale_`（标准差），它们的长度必须等于特征数。

- `X_base_train` 是 6 列 → `scaler.mean_` 长度为 6
- `X_all_train` 是 13 列 → `scaler.mean_` 长度为 13

如果用 6 列的 scaler 去 `transform` 13 列的数据，会报维度错误。所以**必须分别 `fit`**。

### 4.4 fit_transform 与 transform 的区别（回顾模块 1）

```python
X_base_train_s = scaler_std.fit_transform(X_base_train)   # 训练集：fit + transform
X_base_test_s  = scaler_std.transform(X_base_test)        # 测试集：只 transform
```

- **`fit_transform(X_train)`**：从训练集计算均值和标准差，然后标准化训练集。
- **`transform(X_test)`**：用训练集的均值和标准差标准化测试集。

> ⚠️ **常见问题**：为什么测试集不能 `fit_transform`？
>
> 因为这会导致**数据泄露（Data Leakage）**：测试集的统计信息（均值、标准差）泄露到模型中，使模型在测试集上表现"虚高"。正确做法是测试集永远只用 `transform`，且参数来自训练集。

### 4.5 标准化后的数据

```
X_base_train_s.shape = (56000, 6)     # 均值≈0, 标准差≈1
X_base_test_s.shape  = (24000, 6)
X_all_train_s.shape  = (56000, 13)
X_all_test_s.shape   = (24000, 13)
```

至此，我们准备好了 4 个标准化后的特征矩阵，可以开始训练模型了。

---

## 五、fe_sets 字典：组织对比实验（**重点**）

### 5.1 完整代码

```python
# 训练
fe_sets = {
    'Base (6 features)': (X_base_train_s, X_base_test_s, 6),
    'Engineered (13 features)': (X_all_train_s, X_all_test_s, len(all_feat_names)),
}

fe_results = []

fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
fig_pr, ax_pr = plt.subplots(figsize=(9, 7))

colors_fe = {'Base (6 features)': '#3498db', 'Engineered (13 features)': '#e74c3c'}
```

### 5.2 字典 + 元组的设计模式

`fe_sets` 是一个字典，键是特征集名称（字符串），值是一个**三元组**：

```python
(训练集, 测试集, 特征数)
```

| 键 | 训练集 | 测试集 | 特征数 |
|----|--------|--------|--------|
| `'Base (6 features)'` | `X_base_train_s` | `X_base_test_s` | `6` |
| `'Engineered (13 features)'` | `X_all_train_s` | `X_all_test_s` | `13`（`len(all_feat_names)`） |

### 5.3 为什么用这种结构？

这种"字典 + 元组"的模式有几个优点：

1. **可扩展**：如果以后想加第三个特征集（比如"只用构造特征"），只需在字典里加一行，循环代码完全不用改。
2. **可读性强**：`for set_name, (X_tr, X_te, n_feat) in fe_sets.items()` 一眼就能看出循环变量是什么。
3. **集中管理**：所有实验配置在一个字典里，修改方便。

### 5.4 `len(all_feat_names)` vs 硬编码 13

注意第二个特征数写成了 `len(all_feat_names)` 而不是直接写 `13`。这是**防御性编程**：如果以后增减构造特征，`all_feat_names` 的长度会自动更新，不需要手动改这里的数字。

而第一个写成了硬编码 `6`，是因为 `base_features` 在文件开头就固定了，且 Base 集不会变。两种写法都可以，关键是保持一致性。

### 5.5 其他准备工作

```python
fe_results = []   # 用于收集每次实验的结果（字典）
```

空列表，循环中会往里 `append` 结果字典。

```python
fig_roc, ax_roc = plt.subplots(figsize=(9, 7))
fig_pr, ax_pr = plt.subplots(figsize=(9, 7))
```

预先创建两张画布（figure）和坐标轴（axes），循环中往同一张画布上画两条曲线（Base 一条、Engineered 一条），实现"对比图"效果。

```python
colors_fe = {'Base (6 features)': '#3498db', 'Engineered (13 features)': '#e74c3c'}
```

颜色字典：Base 用蓝色（`#3498db`），Engineered 用红色（`#e74c3c`）。后续画图时通过 `colors_fe[set_name]` 取色，保证两次循环的颜色一致。

> 💡 **小贴士**：`#3498db` 是扁平化设计常用的蓝色，`#e74c3c` 是对应的红色。这种"蓝 vs 红"的对比色在学术图表中很常见，色盲友好度也不错。

---

## 六、循环中的模型训练与评估

### 6.1 完整代码

```python
for set_name, (X_tr, X_te, n_feat) in fe_sets.items():
    print(f"\n     --- {set_name} ({n_feat} 个特征) ---")

    lr_fe = LogisticRegression(
        class_weight='balanced',
        max_iter=5000,
        random_state=RANDOM_STATE,
        solver='lbfgs'
    )
    start = time.time()
    lr_fe.fit(X_tr, y_eng_train)
    elapsed = time.time() - start

    y_prob_fe = lr_fe.predict_proba(X_te)[:, 1]
    y_pred_fe = lr_fe.predict(X_te)

    auc_fe = roc_auc_score(y_eng_test, y_prob_fe)
    recall_fe = recall_score(y_eng_test, y_pred_fe, pos_label=1)
    brier_fe = brier_score_loss(y_eng_test, y_prob_fe)
    ap_fe = average_precision_score(y_eng_test, y_prob_fe)

    fe_results.append({
        'Feature_Set': set_name,
        'N_Features': n_feat,
        'AUC': auc_fe,
        'Recall': recall_fe,
        'Brier': brier_fe,
        'PR_AUC': ap_fe,
        'Time': elapsed
    })

    print(f"      AUC={auc_fe:.4f}  Recall={recall_fe:.4f}  Brier={brier_fe:.4f}  PR-AUC={ap_fe:.4f}")

    # ROC
    fpr, tpr, _ = roc_curve(y_eng_test, y_prob_fe)
    ax_roc.plot(fpr, tpr, color=colors_fe[set_name], linewidth=2.5,
                label=f'{set_name} (AUC={auc_fe:.4f})')

    # PR
    prec, rec, _ = precision_recall_curve(y_eng_test, y_prob_fe)
    ax_pr.plot(rec, prec, color=colors_fe[set_name], linewidth=2.5,
               label=f'{set_name} (AP={ap_fe:.4f})')
```

### 6.2 循环变量解包

```python
for set_name, (X_tr, X_te, n_feat) in fe_sets.items():
```

`fe_sets.items()` 每次迭代返回 `(键, 值)`，例如 `('Base (6 features)', (X_base_train_s, X_base_test_s, 6))`。

外层 `set_name, (...)` 解包键和值；内层 `(X_tr, X_te, n_feat)` 进一步解包元组的三个元素。这种"嵌套解包"是 Python 的优雅特性。

### 6.3 模型创建与训练（回顾案例 3 / 模块 1）

```python
lr_fe = LogisticRegression(
    class_weight='balanced',
    max_iter=5000,
    random_state=RANDOM_STATE,
    solver='lbfgs'
)
```

参数与模块 1 完全一致（除了没有 `tol=1e-5`，因为这里不关注收敛迭代数的细微差异）：

- **`class_weight='balanced'`**：自动平衡少数类 VIVO 和多数类 MORTO 的权重。
- **`max_iter=5000`**：最大迭代次数，留足收敛余量。
- **`random_state=RANDOM_STATE`**：保证可复现。
- **`solver='lbfgs'`**：拟牛顿法优化器，适合中小型数据集。

```python
start = time.time()
lr_fe.fit(X_tr, y_eng_train)
elapsed = time.time() - start
```

用 `time.time()` 计时训练耗时，单位是秒。

### 6.4 两种预测（回顾案例 3）

```python
y_prob_fe = lr_fe.predict_proba(X_te)[:, 1]   # 预测概率（属于正类 VIVO 的概率）
y_pred_fe = lr_fe.predict(X_te)               # 预测类别（0 或 1）
```

- **`predict_proba`** 返回形状 `(n_samples, 2)` 的数组，第 0 列是 MORTO 概率，第 1 列是 VIVO 概率。`[:, 1]` 取第 1 列。
- **`predict`** 返回形状 `(n_samples,)` 的类别数组，内部对 `predict_proba` 的结果以 0.5 为阈值做硬判断。

> 💡 **小贴士**：`predict` 用的默认阈值是 0.5。在不平衡数据中，0.5 往往不是最优阈值——这正是后面 Recall 下降的一个潜在原因。

### 6.5 四个评估指标（回顾案例 3）

```python
auc_fe = roc_auc_score(y_eng_test, y_prob_fe)
recall_fe = recall_score(y_eng_test, y_pred_fe, pos_label=1)
brier_fe = brier_score_loss(y_eng_test, y_prob_fe)
ap_fe = average_precision_score(y_eng_test, y_prob_fe)
```

| 指标 | 输入 | 含义 | 方向 |
|------|------|------|------|
| AUC | `y_prob_fe`（概率） | ROC 曲线下面积，区分能力 | 越大越好 |
| Recall | `y_pred_fe`（类别） | 召回率 = TP / (TP + FN)，少数类捕捉能力 | 越大越好 |
| Brier | `y_prob_fe`（概率） | 预测概率与真实标签的均方误差，校准度 | 越小越好 |
| PR-AUC (AP) | `y_prob_fe`（概率） | PR 曲线下面积，不平衡数据下的综合指标 | 越大越好 |

**注意**：AUC、Brier、PR-AUC 都用**概率** `y_prob_fe` 计算，只有 Recall 用**类别** `y_pred_fe` 计算。这意味着前三个指标不受阈值影响，而 Recall 严重依赖 0.5 阈值。

### 6.6 收集结果

```python
fe_results.append({
    'Feature_Set': set_name,
    'N_Features': n_feat,
    'AUC': auc_fe,
    'Recall': recall_fe,
    'Brier': brier_fe,
    'PR_AUC': ap_fe,
    'Time': elapsed
})
```

每次循环把结果存成一个字典，`append` 到 `fe_results` 列表。循环结束后 `fe_results` 是一个长度为 2 的列表，可以直接转成 DataFrame。

### 6.7 绘制 ROC 和 PR 曲线（循环内）

```python
fpr, tpr, _ = roc_curve(y_eng_test, y_prob_fe)
ax_roc.plot(fpr, tpr, color=colors_fe[set_name], linewidth=2.5,
            label=f'{set_name} (AUC={auc_fe:.4f})')

prec, rec, _ = precision_recall_curve(y_eng_test, y_prob_fe)
ax_pr.plot(rec, prec, color=colors_fe[set_name], linewidth=2.5,
           label=f'{set_name} (AP={ap_fe:.4f})')
```

每次循环往 `ax_roc` 和 `ax_pr` 上画一条曲线。两次循环后，两张图各有两条曲线（蓝色 Base + 红色 Engineered）。具体解读见后续章节。

---

## 七、图07e：ROC 曲线对比

### 7.1 完整代码

```python
# ROC
fpr, tpr, _ = roc_curve(y_eng_test, y_prob_fe)
ax_roc.plot(fpr, tpr, color=colors_fe[set_name], linewidth=2.5,
            label=f'{set_name} (AUC={auc_fe:.4f})')

# 循环结束后：
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curve: Base vs Engineered Features', fontsize=14, fontweight='bold')
ax_roc.legend(fontsize=10)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07e_fe_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
```

### 7.2 `roc_curve` 返回值详解

```python
fpr, tpr, _ = roc_curve(y_eng_test, y_prob_fe)
```

`roc_curve` 返回三个 NumPy 数组：

| 返回值 | 含义 | 范围 |
|--------|------|------|
| `fpr` | False Positive Rate = FP / (FP + TN) | [0, 1] |
| `tpr` | True Positive Rate = TP / (TP + FN)（即 Recall） | [0, 1] |
| `thresholds` | 对应的决策阈值 | [0, 1]，降序 |

`_` 表示丢弃 thresholds（我们只画曲线，不标注阈值点）。

### 7.3 `ax_roc.plot(fpr, tpr, ...)` 绘制曲线

```python
ax_roc.plot(fpr, tpr, color=colors_fe[set_name], linewidth=2.5,
            label=f'{set_name} (AUC={auc_fe:.4f})')
```

- **`fpr`** 作为 x 轴，**`tpr`** 作为 y 轴。
- **`color`**：Base 用蓝色，Engineered 用红色。
- **`linewidth=2.5`**：线宽 2.5，比默认稍粗，便于对比。
- **`label`**：图例文本，包含特征集名称和 AUC 值（保留 4 位小数）。

### 7.4 对角线参考线

```python
ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.4)
```

- **`[0, 1], [0, 1]`**：从 (0,0) 到 (1,1) 的直线。
- **`'k--'`**：`k` 表示黑色（black），`--` 表示虚线。
- **`alpha=0.4`**：透明度 40%，避免抢戏。

这条对角线代表**随机猜测**的 ROC 曲线（AUC=0.5）。模型曲线越远离对角线（越靠左上角），性能越好。

### 7.5 其他装饰代码

```python
ax_roc.set_xlabel('False Positive Rate', fontsize=12)
ax_roc.set_ylabel('True Positive Rate', fontsize=12)
ax_roc.set_title('ROC Curve: Base vs Engineered Features', fontsize=14, fontweight='bold')
ax_roc.legend(fontsize=10)
ax_roc.spines['top'].set_visible(False)
ax_roc.spines['right'].set_visible(False)
```

- 设置 x/y 轴标签、标题、图例。
- `spines['top'].set_visible(False)` 和 `spines['right'].set_visible(False)`：隐藏上边框和右边框，让图更简洁（这是学术绘图的常见风格）。

```python
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07e_fe_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
```

- `tight_layout()`：自动调整子图间距，避免标签重叠。
- `savefig(..., dpi=150, bbox_inches='tight')`：以 150 DPI 保存，`bbox_inches='tight'` 自动裁剪空白边缘。
- `close()`：关闭画布，释放内存（重要！否则循环多了会内存泄漏）。

### 7.6 图像解读

![ROC曲线对比](../img/07e_fe_roc_curve.png)

**实际运行结果**：

| 特征集 | AUC |
|--------|-----|
| Base (6 features) | 0.8698 |
| Engineered (13 features) | 0.8737 |

**重点解读**：

- 两条曲线非常接近，肉眼难以区分——这说明特征构造对 ROC 曲线的整体形状影响很小。
- Engineered 的 AUC=0.8737 **略高于** Base 的 0.8698，差距仅 0.0039（约 0.45%）。
- 在 ROC 空间中，Engineered 曲线在大部分阈值范围内都**略微**更靠左上角，但优势很小。
- 两条曲线都远高于对角线（AUC=0.5），说明两个模型都有较强的区分能力。

> 💡 **小贴士**：AUC 差距 < 0.01 时，单看 ROC 曲线很难判断"哪个更好"。这时候需要看其他指标（Brier、PR-AUC、Recall）来综合判断。

---

## 八、图07f：PR 曲线对比

### 8.1 完整代码

```python
# PR
prec, rec, _ = precision_recall_curve(y_eng_test, y_prob_fe)
ax_pr.plot(rec, prec, color=colors_fe[set_name], linewidth=2.5,
           label=f'{set_name} (AP={ap_fe:.4f})')

# 循环结束后：
ax_pr.set_xlabel('Recall', fontsize=12)
ax_pr.set_ylabel('Precision', fontsize=12)
ax_pr.set_title('PR Curve: Base vs Engineered Features', fontsize=14, fontweight='bold')
ax_pr.legend(fontsize=10)
ax_pr.spines['top'].set_visible(False)
ax_pr.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07f_fe_pr_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
```

### 8.2 `precision_recall_curve` 返回值详解

```python
prec, rec, _ = precision_recall_curve(y_eng_test, y_prob_fe)
```

`precision_recall_curve` 返回三个 NumPy 数组：

| 返回值 | 含义 | 范围 |
|--------|------|------|
| `prec` | Precision = TP / (TP + FP) | [0, 1] |
| `rec` | Recall = TP / (TP + FN) | [0, 1] |
| `thresholds` | 对应的决策阈值 | [0, 1]，长度比 prec/rec 少 1 |

**注意**：`precision_recall_curve` 的返回值顺序是 `(precision, recall, thresholds)`，而 `roc_curve` 是 `(fpr, tpr, thresholds)`。两个函数的顺序不同，容易记混！

### 8.3 PR 曲线 vs ROC 曲线（回顾案例 3）

| 维度 | ROC 曲线 | PR 曲线 |
|------|----------|---------|
| 横轴 | FPR = FP / (FP + TN) | Recall = TP / (TP + FN) |
| 纵轴 | TPR = TP / (TP + FN) | Precision = TP / (TP + FP) |
| 基线 | 对角线（AUC=0.5） | 水平线 y=正类比例（AP=0.41） |
| 是否受 TN 影响 | 是 | 否 |
| 不平衡数据敏感度 | 较低 | 较高 |

**关键区别**：ROC 曲线的 FPR 分母包含 TN（真阴性），当负类远多于正类时，FPR 会被"稀释"得很小，导致 ROC 曲线看起来很漂亮。PR 曲线不含 TN，对不平衡数据更敏感。

在我们的数据中，VIVO（正类）约 41%，MORTO（负类）约 59%，不平衡程度中等。PR 曲线比 ROC 曲线更能反映少数类的预测质量。

### 8.4 `ax_pr.plot(rec, prec, ...)` 绘制曲线

```python
ax_pr.plot(rec, prec, color=colors_fe[set_name], linewidth=2.5,
           label=f'{set_name} (AP={ap_fe:.4f})')
```

- **`rec`** 作为 x 轴，**`prec`** 作为 y 轴。
- 注意顺序：`plot(rec, prec)`，先 x 后 y。这与 ROC 的 `plot(fpr, tpr)` 一致。
- `AP`（Average Precision）是 PR 曲线下面积的近似，作为图例显示。

### 8.5 为什么 PR 曲线没有对角线？

回顾 ROC 图，我们画了 `[0,1], [0,1]` 的对角线作为随机基线。但 PR 图没有这条线，因为 PR 曲线的"随机基线"是一条**水平线** y = 正类比例（约 0.41），而不是对角线。

代码里没有画这条水平基线，但你可以脑补一条 y=0.41 的虚线——两条 PR 曲线都明显高于这条线，说明模型确实学到了东西。

### 8.6 图像解读

![PR曲线对比](../img/07f_fe_pr_curve.png)

**实际运行结果**：

| 特征集 | AP (PR-AUC) |
|--------|-------------|
| Base (6 features) | 0.7412 |
| Engineered (13 features) | 0.7530 |

**重点解读**：

- Engineered 的 AP=0.7530 **高于** Base 的 0.7412，差距 0.0118（约 1.6%）。
- 这个差距比 AUC 的差距（0.0039）更明显——印证了"PR 曲线对不平衡数据更敏感"的结论。
- 在高 Recall 区域（Recall > 0.8），两条曲线的 Precision 都开始下降，但 Engineered 的下降幅度略小。
- 整体来看，特征构造对 PR 曲线的改善比对 ROC 曲线更显著。

> 💡 **小贴士**：当比较两个模型在不平衡数据上的表现时，**优先看 PR-AUC**，再看 AUC。PR-AUC 的差异更能反映少数类预测质量的真实变化。

---

## 九、图07g：性能对比柱状图

### 9.1 完整代码

```python
# 性能对比图 (特征构造)
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
for i, metric in enumerate(['AUC', 'Recall', 'Brier', 'PR_AUC']):
    ax = axes[i]
    names = [r['Feature_Set'] for r in fe_results]
    vals = [r[metric] for r in fe_results]
    colors_b = ['#3498db', '#e74c3c']
    bars = ax.bar(names, vals, color=colors_b, edgecolor='white', width=0.4)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Feature Engineering: Base vs Engineered Features',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07g_fe_performance.png"), dpi=150, bbox_inches='tight')
plt.close()
```

### 9.2 子图布局

```python
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
```

- **`1, 4`**：1 行 4 列，共 4 个子图并排。
- **`figsize=(16, 5)`**：画布宽 16 英寸、高 5 英寸，每个子图约 4×5 英寸。
- 4 个子图分别画 AUC、Recall、Brier、PR_AUC。

### 9.3 循环绘制 4 个子图

```python
for i, metric in enumerate(['AUC', 'Recall', 'Brier', 'PR_AUC']):
    ax = axes[i]
    names = [r['Feature_Set'] for r in fe_results]
    vals = [r[metric] for r in fe_results]
```

- `enumerate(['AUC', 'Recall', 'Brier', 'PR_AUC'])`：遍历 4 个指标名，`i` 是索引（0-3），`metric` 是指标名。
- `ax = axes[i]`：取第 i 个子图。
- `names`：从 `fe_results` 提取两个特征集名称 → `['Base (6 features)', 'Engineered (13 features)']`。
- `vals`：从 `fe_results` 提取该指标的两个值，例如 AUC → `[0.8698, 0.8737]`。

### 9.4 绘制柱状图

```python
colors_b = ['#3498db', '#e74c3c']
bars = ax.bar(names, vals, color=colors_b, edgecolor='white', width=0.4)
```

- `ax.bar(names, vals, ...)`：以 `names` 为 x 轴标签，`vals` 为柱高。
- `color=colors_b`：第一根柱蓝色（Base），第二根柱红色（Engineered）。
- `edgecolor='white'`：柱子边缘白色，视觉上更清晰。
- `width=0.4`：柱宽 0.4（默认 0.8），让柱子更细，留白更多。

### 9.5 在柱子上标注数值

```python
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
```

- `zip(bars, vals)`：把柱子对象和数值配对。
- `bar.get_x() + bar.get_width()/2`：柱子的水平中心点。
- `bar.get_height()`：柱子顶部 y 坐标。
- `ax.text(x, y, text, ha, va, ...)`：在 (x, y) 处写文字。
  - `ha='center'`：水平居中对齐。
  - `va='bottom'`：垂直底部对齐（文字在柱子上方）。
  - `fontsize=10, fontweight='bold'`：字号 10，加粗。

### 9.6 子图标题与装饰

```python
ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

每个子图以指标名为标题，隐藏上/右边框。

```python
plt.suptitle('Feature Engineering: Base vs Engineered Features',
             fontsize=14, fontweight='bold')
plt.tight_layout()
```

- `suptitle`：整个画布的总标题（super title），位于所有子图上方。
- `tight_layout()`：自动调整间距。

### 9.7 图像解读

![性能对比](../img/07g_fe_performance.png)

**实际运行结果**：

| 特征集 | AUC | Recall | Brier | PR-AUC |
|--------|-----|--------|-------|--------|
| Base (6 features) | 0.8698 | 0.9026 | 0.1437 | 0.7412 |
| Engineered (13 features) | 0.8737 | 0.8886 | 0.1409 | 0.7530 |

**变化量**：

| 指标 | Base | Engineered | 变化 | 方向 |
|------|------|------------|------|------|
| AUC | 0.8698 | 0.8737 | **+0.0039** | ↑ 变好 |
| Recall | 0.9026 | 0.8886 | **-0.0140** | ↓ 变差 |
| Brier | 0.1437 | 0.1409 | **-0.0028** | ↓ 变好（Brier 越小越好） |
| PR-AUC | 0.7412 | 0.7530 | **+0.0118** | ↑ 变好 |

**重点解读**：

- **AUC 柱状图**：Engineered 略高于 Base，但差距很小，肉眼几乎看不出。
- **Recall 柱状图**：**Engineered 反而低于 Base！** 这是本案例最反直觉的结果。
- **Brier 柱状图**：Engineered 略低于 Base（更好），差距也很小。
- **PR-AUC 柱状图**：Engineered 高于 Base，差距比 AUC 明显。

> 🚨 **核心观察**：4 个指标中，3 个变好（AUC↑、Brier↓、PR-AUC↑），1 个变差（Recall↓）。这说明特征构造不是"全面胜利"，而是有得有失的**权衡**。

---

## 十、核心讨论：结果综合解读（本模块最重要的部分）

### 10.1 实际运行结果汇总

| 特征集 | 特征数 | AUC | Recall | Brier | PR-AUC | 耗时 |
|--------|--------|-----|--------|-------|--------|------|
| Base (6 features) | 6 | 0.8698 | 0.9026 | 0.1437 | 0.7412 | 0.021s |
| Engineered (13 features) | 13 | 0.8737 | 0.8886 | 0.1409 | 0.7530 | 0.056s |

**变化量**：

| 指标 | 变化 | 相对变化 | 解读 |
|------|------|----------|------|
| AUC | +0.0039 | +0.45% | 区分能力小幅提升 |
| Recall | -0.0140 | -1.55% | 少数类捕捉能力下降 |
| Brier | -0.0028 | -1.95% | 校准度改善 |
| PR-AUC | +0.0118 | +1.59% | 不平衡数据处理能力提升 |
| 耗时 | +0.035s | +167% | 训练时间增加（特征数翻倍） |

### 10.2 三个"变好"的指标

#### AUC 上升 +0.0039：区分能力提升

AUC 衡量模型"区分正负类"的整体能力，与阈值无关。Engineered 的 AUC 从 0.8698 升到 0.8737，说明 7 个新特征确实为模型提供了**额外的区分信息**。

但提升幅度很小（0.45%），可能原因：
- 基础 6 特征已经包含了大部分区分信息（特别是 `Age` 和 `year`）。
- 构造特征（如 `Age_Group`、`Age_Sq`）本质上是 `Age` 的非线性变换，与 `Age` 高度相关，新增信息有限。
- 逻辑回归是线性模型，无法充分利用非线性特征（如 `Age_Sq`）的交互效应。

#### Brier 下降 -0.0028：校准度改善

Brier Score = 预测概率与真实标签的均方误差，衡量**预测概率的准确性**。Engineered 的 Brier 从 0.1437 降到 0.1409，说明预测概率更接近真实标签。

这意味着：如果医生用模型的预测概率做决策（如"概率 > 0.7 才建议进一步检查"），Engineered 的概率更可靠。

#### PR-AUC 上升 +0.0118：不平衡数据处理能力提升

PR-AUC（AP）对不平衡数据更敏感。Engineered 的 PR-AUC 从 0.7412 升到 0.7530，提升 1.59%——比 AUC 的提升（0.45%）更显著。

这说明特征构造对**少数类（VIVO）的预测质量**有实质性改善，尤其是在高 Recall 区域。

### 10.3 一个"变差"的指标：Recall 下降 -0.0140（**反直觉结果！**）

> 🚨 **反直觉结果**：AUC 上升了，为什么 Recall 反而下降？
>
> 这是本案例最重要的教学点。AUC 和 Recall 看似都衡量"模型好不好"，但它们衡量的是**不同的东西**。AUC 上升不意味着所有阈值下的 Recall 都上升。

#### 原因 1：决策边界复杂度增加

Base 模型在 6 维特征空间中画一条线性决策边界；Engineered 模型在 13 维空间中画一条线性决策边界。维度增加后，决策边界的"形状"更复杂，对少数类（VIVO）的拟合可能反而更困难——尤其是当新增特征与少数类的相关性不强时。

具体来说，`Age_Sq`、`Year_From_2000` 等特征主要捕捉**整体趋势**，对少数类的边界贡献有限，反而可能引入噪声。

#### 原因 2：逻辑回归是线性模型

逻辑回归只能学到**线性决策边界**。我们构造的 `Age_Sq`（年龄平方）本意是捕捉非线性效应（如 U 型关系），但逻辑回归把它当作一个**独立的线性特征**处理，无法真正学到非线性。

更糟糕的是，`Age` 和 `Age_Sq` 高度相关（相关系数接近 1），这种**多重共线性**会让逻辑回归的系数估计不稳定，可能扭曲决策边界。

> 💡 **小贴士**：如果想真正利用非线性特征，应该用非线性模型（如随机森林、梯度提升树、神经网络）。逻辑回归 + 非线性特征是一个"半吊子"组合。

#### 原因 3：Brier 改善的代价

Brier Score 衡量**所有样本**的预测概率准确性。Engineered 模型为了让多数类（MORTO）的概率更准确，可能**牺牲**了少数类（VIVO）的边界精度。

具体来说：在 0.5 阈值附近，Engineered 模型对一些"模糊"的 VIVO 样本给出了更低的概率（更接近 0.5 但低于 0.5），导致它们被预测为 MORTO，从而 Recall 下降。但这些样本的概率预测本身可能更"准确"（更符合真实风险），所以 Brier 改善。

这是一种**整体校准 vs 局部召回**的权衡。

#### 原因 4：阈值效应

Recall 严重依赖 0.5 这个默认阈值。在不平衡数据中，0.5 往往不是最优阈值。

考虑这个例子：
- Base 模型对某 VIVO 样本预测概率 0.51 → 预测为 VIVO → 计入 TP
- Engineered 模型对同一样本预测概率 0.49 → 预测为 MORTO → 计入 FN

两个预测概率只差 0.02，但 Recall 一个算对、一个算错。如果 Engineered 模型整体上把概率"压低"了一点（更保守），就会有一批原本刚好过 0.5 的 VIVO 样本掉到 0.5 以下，Recall 下降。

**解决方案**：如果业务关心 Recall，应该**调整阈值**（如改为 0.4 或 0.3），而不是死守 0.5。

### 10.4 教学要点：多指标评估的重要性

> 📌 **核心教学点**：不要只看单一指标，需要多维度评估。
>
> 如果只看 AUC，我们会得出"特征构造有效"的结论。
> 如果只看 Recall，我们会得出"特征构造有害"的结论。
> 如果只看 Brier，我们会得出"特征构造改善校准"的结论。
>
> 真实情况是：特征构造在**区分能力（AUC）**、**校准度（Brier）**、**不平衡处理（PR-AUC）**三个维度上有小幅改善，但在**默认阈值下的少数类召回（Recall）**上有损失。
>
> 这种"多维权衡"才是特征工程的真实面貌。

### 10.5 实际业务建议

如果这是一个真实的临床决策支持系统，应该怎么选？

| 业务场景 | 优先指标 | 推荐特征集 | 理由 |
|----------|----------|------------|------|
| 筛查（漏诊代价高） | Recall | Base 或调阈值 | Recall 优先，可接受更多误诊 |
| 风险分层（概率要准） | Brier | Engineered | 概率校准好，适合分层 |
| 整体区分（谁高谁低） | AUC | Engineered | AUC 略高 |
| 不平衡数据评估 | PR-AUC | Engineered | PR-AUC 明显更高 |

**关键洞察**：没有"绝对更好"的特征集，只有"更适合业务目标"的特征集。

---

## 十一、整个案例教程 4 的总结

### 11.1 模块 1：标准化比较的核心结论

| 维度 | Raw（不标准化） | StandardScaler | RobustScaler |
|------|-----------------|----------------|--------------|
| AUC | 与标准化后相同 | 与 Raw 相同 | 与 Raw 相同 |
| 收敛迭代数 | 174 次 | 10 次 | 10 次 |
| 系数可比性 | 不可比 | 可比 | 可比 |
| 受离群值影响 | — | 受影响 | 不受影响 |

**核心结论**：
- 标准化**不改变模型性能**（AUC、Recall、Brier 都不变），但**极大加速收敛**（174→10 次迭代）。
- 标准化让系数变得**可比**，可以用来判断特征重要性。
- `StandardScaler` 适合无极端离群值的数据；`RobustScaler` 适合含离群值的数据。

### 11.2 模块 2：特征构造的核心结论

我们基于医学领域知识构造了 7 个新特征：

| 特征 | 类型 | 医学意义 |
|------|------|----------|
| `Age_Group` | 分箱 | 不同年龄段癌症类型和预后差异 |
| `Age_Sq` | 非线性 | 年龄与死亡率的 U 型关系 |
| `Year_Decade` | 分箱 | 医疗技术进步的年代效应 |
| `Year_From_2000` | 趋势 | 存活率随时间的线性改善 |
| `Gender_x_AgeGroup` | 交互 | 性别效应的年龄依赖性 |
| `Is_Child` | 指示 | 儿童癌症的特殊性 |
| `Age_Centered` | 中心化 | 提高截距可解释性 |

**核心结论**：特征工程 = 将领域知识编码为模型可理解的形式。

### 11.3 模块 3：对比实验的核心结论

| 指标 | Base | Engineered | 变化 | 评价 |
|------|------|------------|------|------|
| AUC | 0.8698 | 0.8737 | +0.0039 | 小幅提升 |
| Recall | 0.9026 | 0.8886 | -0.0140 | 下降（反直觉） |
| Brier | 0.1437 | 0.1409 | -0.0028 | 小幅改善 |
| PR-AUC | 0.7412 | 0.7530 | +0.0118 | 明显提升 |

**核心结论**：特征构造带来 AUC、Brier、PR-AUC 的小幅改善，但 Recall 下降——这是**权衡**，不是全面胜利。

### 11.4 跨模块的核心认知

#### 认知 1：标准化不改变性能，但改变效率

这是模块 1 的核心反直觉结论。很多人以为"标准化能让模型更准"，实验证明并非如此——标准化改变的是**优化路径**，不是**最优解**。

#### 认知 2：特征构造改善部分指标，但不是全部

这是模块 3 的核心反直觉结论。很多人以为"加特征一定更好"，实验证明并非如此——特征构造是**权衡**，可能改善 AUC 但损害 Recall。

#### 认知 3：特征质量决定模型上限

这是整个案例教程 4 的核心认知。所谓"上限"，是指无论你怎么调参、怎么换模型，性能都不会超过特征本身所蕴含的信息量。

- 垃圾特征 + 精心调参 → 性能上限低
- 好特征 + 默认参数 → 性能上限高

但"好特征"不等于"多特征"。本案例中，从 6 个特征加到 13 个，性能提升有限——因为基础 6 特征已经包含了大部分信息，构造特征主要是**重新编码**已有信息（如 `Age` → `Age_Group` + `Age_Sq` + `Age_Centered`）。

#### 认知 4："好特征"需要结合领域知识、数据形态、模型特性

- **领域知识**：医学告诉我们"儿童癌症是不同疾病"，所以构造 `Is_Child`。
- **数据形态**：观察 `Age` 分布发现两端稀疏，所以构造 `Age_Group` 分箱。
- **模型特性**：逻辑回归是线性模型，构造 `Age_Sq` 本意是引入非线性，但效果有限——因为线性模型无法真正利用非线性特征。

> 📌 **从统计到建模的桥梁**：
>
> 回顾案例教程 2（特征分析），我们对每个特征做了效应量排序，发现 `Age` 和 `year` 是最强的预测特征。这为案例教程 4 的特征构造指明了方向：
>
> - 案例 2 的效应量排序 → 案例 4 的特征构造方向
> - `Age` 效应最强 → 构造 `Age_Group`、`Age_Sq`、`Age_Centered`、`Is_Child`
> - `year` 效应次强 → 构造 `Year_Decade`、`Year_From_2000`
> - `Gender` 有效应 → 构造 `Gender_x_AgeGroup` 交互
>
> 这就是"从统计到建模的桥梁"：统计分析告诉我们**哪些特征重要**，特征工程把这些重要性**编码成模型可利用的形式**。

---

## 十二、小贴士与常见问题

### 12.1 小贴士

> 💡 **小贴士 1**：对比实验一定要控制变量。本模块通过相同的 `random_state` 和 `stratify` 保证两个特征集的样本划分完全一致，这是公平对比的前提。

> 💡 **小贴士 2**：`_` 占位符是 Python 的约定，表示"我故意丢弃这个返回值"。看到 `_` 就知道这个值不重要，不要在这里找 bug。

> 💡 **小贴士 3**：当 AUC 差距 < 0.01 时，单看 ROC 曲线很难判断优劣。这时候要结合 PR-AUC、Brier、Recall 等指标综合判断。

> 💡 **小贴士 4**：PR 曲线对不平衡数据更敏感，比较少数类预测质量时优先看 PR-AUC，而不是 AUC。

> 💡 **小贴士 5**：Recall 严重依赖 0.5 阈值。如果业务关心 Recall，应该调整阈值（如改为 0.4 或 0.3），而不是死守默认值。

> 💡 **小贴士 6**：逻辑回归 + 非线性特征（如 `Age_Sq`）是"半吊子"组合。想真正利用非线性，应该用随机森林、梯度提升树等非线性模型。

### 12.2 常见问题

> ❓ **Q1：为什么对 `X_base_raw` 和 `X_all_raw` 要分别 `fit_transform`？不能用同一个 imputer 吗？**
>
> A：因为两个特征集的列数不同（6 vs 13），imputer 内部的 `statistics_` 数组长度必须与列数匹配。用 6 列的 imputer 去 transform 13 列的数据会报维度错误。所以必须分别 `fit`。更深层的原因是：构造特征（如 `Age_Sq`）的均值与原始 `Age` 不同，需要独立计算。

> ❓ **Q2：两次 `train_test_split` 为什么能保证样本划分一致？**
>
> A：因为三个条件同时满足：(1) 传入相同的 `y_eng`；(2) 相同的 `random_state=42`；(3) 相同的 `stratify=y_eng`。`train_test_split` 内部用随机数生成器打乱样本顺序，相同的种子 + 相同的输入长度 = 相同的打乱索引。

> ❓ **Q3：为什么第二次 `train_test_split` 用 `_` 接收 y？**
>
> A：因为第二次返回的 `y_train` 和 `y_test` 与第一次完全相同（同样的 `y_eng`、同样的 `random_state`），没必要再存一份。用 `_` 表示"我故意丢弃"，是 Python 的约定。

> ❓ **Q4：AUC 上升了，为什么 Recall 反而下降？**
>
> A：这是本案例的核心反直觉点。原因有四：(1) 决策边界复杂度增加，对少数类拟合更困难；(2) 逻辑回归是线性模型，无法真正利用非线性特征；(3) Brier 改善的代价是多数类概率更准，少数类 Recall 可能为此付出代价；(4) 阈值效应——0.5 阈值对不平衡数据不是最优。详细分析见第 10.3 节。

> ❓ **Q5：Recall 下降了，是不是说明特征构造失败了？**
>
> A：不能这么说。Recall 只是 4 个指标中的一个。特征构造在 AUC、Brier、PR-AUC 三个指标上都有改善，只是 Recall 下降。这是一种**权衡**，不是失败。如果业务关心 Recall，可以调整阈值；如果业务关心概率校准，Engineered 反而更好。

> ❓ **Q6：为什么 `precision_recall_curve` 返回的 thresholds 比 prec 和 rec 少一个？**
>
> A：这是 scikit-learn 的设计。PR 曲线的最后一个点是 (recall=1, precision=正类比例)，对应"阈值=0，全部预测为正"的极端情况，没有对应的阈值。所以 thresholds 数组长度 = len(prec) - 1。

> ❓ **Q7：训练耗时从 0.021s 增加到 0.056s，正常吗？**
>
> A：正常。特征数从 6 增加到 13（翻倍多），逻辑回归的优化复杂度与特征数大致呈线性或略超线性关系，耗时增加 167% 在预期范围内。但绝对值仍然很小（< 0.1s），对实际应用没有影响。

> ❓ **Q8：如果我想进一步提升 Recall，应该怎么做？**
>
> A：几个方向：(1) 调整决策阈值（如 0.5 → 0.3），直接提升 Recall；(2) 换用非线性模型（如随机森林、XGBoost），更好地利用非线性特征；(3) 构造更多少数类相关的特征；(4) 调整 `class_weight`（如 `{0:1, 1:2}`），给少数类更大权重。

---

## 十三、本模块小结

### 13.1 我们做了什么

1. **准备两个特征集**：对 6 个基础特征和 13 个全部特征分别做均值插补。
2. **公平划分**：用相同的 `random_state` 和 `stratify` 保证两个特征集的样本划分完全一致。
3. **分别标准化**：对两个特征集分别做 StandardScaler，回顾模块 1。
4. **循环训练评估**：用 `fe_sets` 字典组织实验，循环训练两个逻辑回归模型，计算 4 个指标。
5. **绘制 3 张图**：ROC 曲线对比、PR 曲线对比、性能对比柱状图。
6. **综合解读**：发现 AUC、Brier、PR-AUC 改善，但 Recall 下降——这是权衡，不是全面胜利。

### 13.2 我们学到了什么

1. **对比实验的公平性原则**：控制变量，保证样本划分一致。
2. **`_` 占位符的语义**：表示故意丢弃返回值。
3. **字典 + 元组的循环模式**：优雅地组织多组实验。
4. **ROC 曲线与 PR 曲线的区别**：PR 曲线对不平衡数据更敏感。
5. **多指标评估的重要性**：单一指标会误导决策。
6. **特征构造的权衡本质**：AUC 和 Brier 改善的代价可能是 Recall 下降。

### 13.3 核心一句话

> **特征构造不是免费午餐——它在改善部分指标的同时，可能损害其他指标。理解这种权衡，是特征工程成熟度的标志。**

---

## 十四、整个案例教程 4 的总结

### 14.1 三个模块的知识闭环

```
┌─────────────────────────────────────────────────────────────────┐
│  模块 1: 标准化比较                                              │
│  - Raw vs StandardScaler vs RobustScaler                        │
│  - 结论: 标准化不改变性能，但加速收敛（174→10 次），让系数可比   │
│  - 认知: 标准化改变优化路径，不改变最优解                         │
│                                                                 │
│  模块 2: 特征构造                                                │
│  - 基于医学领域知识构造 7 个新特征                               │
│  - 特征工程 = 将领域知识编码为模型可理解的形式                    │
│  - 认知: 好特征需要结合领域知识、数据形态、模型特性               │
│                                                                 │
│  模块 3: 对比实验（本模块）                                      │
│  - Base (6 特征) vs Engineered (13 特征)                        │
│  - AUC↑ +0.0039, Brier↓ -0.0028, PR-AUC↑ +0.0118               │
│  - Recall↓ -0.0140（反直觉！）                                   │
│  - 认知: 特征构造是权衡，不是全面胜利                             │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 三个核心认知

#### 认知 1：标准化不改变性能，但改变效率

模块 1 的反直觉结论：标准化让 AUC、Recall、Brier 都不变，但收敛迭代数从 174 降到 10。这说明标准化改变的是**优化路径**（梯度下降走得更顺），不是**最优解**（最优系数等价）。

#### 认知 2：特征构造改善部分指标，但不是全部

模块 3 的反直觉结论：加特征后 AUC、Brier、PR-AUC 都改善，但 Recall 下降。这说明特征构造是**权衡**，不是免费午餐。理解权衡的本质，是特征工程成熟度的标志。

#### 认知 3：特征质量决定模型上限

整个案例教程 4 的核心认知：无论你怎么调参、怎么换模型，性能都不会超过特征本身所蕴含的信息量。但"好特征"不等于"多特征"——本案例从 6 个加到 13 个，性能提升有限，因为基础特征已经包含了大部分信息。

### 14.3 从统计到建模的桥梁

回顾案例教程 2（特征分析），我们对每个特征做了效应量排序：

| 排名 | 特征 | 效应量 | 在案例 4 中的构造方向 |
|------|------|--------|----------------------|
| 1 | `Age` | 最强 | `Age_Group`、`Age_Sq`、`Age_Centered`、`Is_Child` |
| 2 | `year` | 次强 | `Year_Decade`、`Year_From_2000` |
| 3 | `Gender` | 中等 | `Gender_x_AgeGroup`（交互） |
| 4-6 | 其他 | 较弱 | 未构造（信息量有限） |

这就是"从统计到建模的桥梁"：
- **案例 2** 的效应量排序 → 告诉我们**哪些特征重要**
- **案例 4** 的特征构造 → 把这些重要性**编码成模型可利用的形式**

统计分析是特征工程的**指南针**，没有它，特征构造就是盲目尝试。

### 14.4 给学生的建议

1. **不要盲目加特征**：每加一个特征，都要问"它带来了什么新信息？"
2. **多指标评估**：永远不要只看 AUC，至少看 AUC + Recall + Brier + PR-AUC。
3. **理解模型特性**：逻辑回归是线性的，构造非线性特征前要想想"模型能用上吗？"
4. **结合领域知识**：纯数据驱动的特征构造容易过拟合，领域知识是正则化。
5. **接受权衡**：没有"全面更好"的特征集，只有"更适合业务目标"的特征集。

### 14.5 案例教程 4 的最终一句话

> **特征质量决定模型上限，但"好特征"需要结合领域知识、数据形态、模型特性。特征工程不是盲目堆砌，而是有目的的编码——把人类对问题的理解，翻译成模型能理解的语言。**

---

> 📚 **下一步学习建议**：
>
> 案例教程 4 到此结束。建议接下来：
> 1. 复习案例 2（特征分析）和案例 3（缺失值处理），形成完整的"数据预处理 → 特征工程 → 建模评估"知识链。
> 2. 尝试用非线性模型（如随机森林、XGBoost）重新跑本案例的对比实验，看看 Recall 是否还会下降。
> 3. 尝试调整决策阈值（如 0.3、0.4），观察 Recall 和 Precision 的变化曲线。
> 4. 构造更多基于领域知识的特征（如 `Age × year` 交互、`Gender × Diagnostic.means` 交互），看是否能进一步提升 PR-AUC。
