# 模块 2：四种插补策略详解

## 学习目标

完成本模块后，你将能够：

1. **理解插补策略的整体框架**：知道为什么要把四种方法放进一个字典里统一管理，并能看懂"循环遍历 + 分支处理"的代码结构。
2. **掌握 Complete Case（完整案例分析）**：理解 `isnull().any(axis=1)`、布尔索引 `~mask`、`.copy()` 的用法，并能讨论删除法的优缺点。
3. **掌握 Mean Imputation（均值插补）**：理解 `SimpleImputer` 的四种 `strategy`，能区分 `fit_transform` 与 `transform` 的本质差异，并能解释为什么均值插补会"压缩方差"。
4. **掌握 KNN Imputation（K 近邻插补）**：理解 `KNNImputer` 的 `n_neighbors` 和 `weights` 参数，能描述 KNN 插补的算法流程，并能分析其 O(n²×d) 的计算代价。
5. **掌握 MICE Imputation（多重插补）**：理解 `IterativeImputer` 的迭代式回归原理，知道 `max_iter` 的作用，并能解释为什么 MICE 是医学研究的首选。
6. **理解标准化的细节**：知道为什么只对数值列做 `StandardScaler`，为什么测试集只能 `transform` 不能 `fit`。
7. **建立"防止数据泄露"的工程意识**：能在任何预处理环节中正确区分"训练集拟合"与"测试集变换"。

> **前置知识**：Python 字典与循环、Pandas 布尔索引、Scikit-learn 的 `fit/transform` 范式、本案例教程模块 1（数据划分与编码）的输出。

> **本模块对应代码**：`src/03_preprocessing_imputation.py` 第 156–224 行。

---

## 一、整体框架：用字典组织四种插补策略

### 1.1 代码回顾

```python
methods = {
    'Complete Case': {
        'desc': '完整案例分析 (删除含缺失的行)',
        'imputer': None,
        'color': '#7f8c8d'
    },
    'Mean Imputation': {
        'desc': '均值插补 (数值:均值 / 分类:众数)',
        'imputer': SimpleImputer(strategy='mean'),
        'color': '#3498db'
    },
    'KNN Imputation': {
        'desc': 'KNN 插补 (n_neighbors=5)',
        'imputer': KNNImputer(n_neighbors=5, weights='distance'),
        'color': '#e67e22'
    },
    'MICE Imputation': {
        'desc': 'MICE 多重插补 (IterativeImputer)',
        'imputer': IterativeImputer(max_iter=10, random_state=RANDOM_STATE),
        'color': '#9b59b6'
    }
}
```

### 1.2 字典结构详解

`methods` 是一个**嵌套字典**：外层的键是方法名（字符串），值又是一个字典，包含三个键：

| 键         | 含义                                                                                                                                                | 示例值                                  |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| `'desc'`   | 方法的中文描述，用于打印日志和报告，让读者一眼看懂这个方法在做什么                                                                                  | `'均值插补 (数值:均值 / 分类:众数)'`    |
| `'imputer'`| Scikit-learn 的插补器对象（一个**实例化**好的 estimator）。Complete Case 不需要插补器，所以设为 `None`                                                  | `SimpleImputer(strategy='mean')`        |
| `'color'`  | 该方法在可视化中使用的颜色（十六进制 RGB）。后续画 ROC 曲线、柱状图时会按这个颜色区分方法                                                            | `'#3498db'`（蓝色）                     |

> **小贴士**：颜色编码是数据可视化中常用的"设计模式"。把颜色和方法绑定在一起，可以保证整篇报告里同一个方法始终用同一种颜色，读者看图时不会混淆。

### 1.3 为什么用字典组织？

如果不用字典，我们可能会写出四个独立的代码块，每个块里重复一遍"插补 → 标准化 → 训练 → 评估"的逻辑。这样做的缺点是：

1. **代码重复**：四段几乎一样的代码，维护时容易漏改。
2. **难以扩展**：想加第五种方法（比如中位数插补），要再复制一整段。
3. **结果难以汇总**：要把四种方法的结果放进同一个表格，需要手动收集。

用字典组织后，我们只需要一个 `for method_name, config in methods.items():` 循环，就能统一处理所有方法。**配置（字典）与逻辑（循环体）分离**，是工程化代码的典型范式。

### 1.4 主循环骨架

```python
results = []
models = {}
imputed_datasets = {}

for method_name, config in methods.items():
    start_time = time.time()

    if method_name == 'Complete Case':
        # 分支 1：删除缺失行
        ...
    else:
        # 分支 2：用 imputer 插补
        ...

    imputed_datasets[method_name] = (X_train_imp, X_test_imp, y_train_imp, y_test_imp)

    # 标准化（仅数值列）
    ...
```

- `results`：列表，收集每种方法的评估指标（AUC、Recall、Brier 等）。
- `models`：字典，保存训练好的逻辑回归模型，后续画 ROC 时复用。
- `imputed_datasets`：字典，保存每种方法插补后的训练/测试集，后续画分布对比图时复用。

> **关键理解**：`if method_name == 'Complete Case'` 这个分支是必要的，因为 Complete Case 不走 `imputer` 流程，而是直接删行。其他三种方法都走统一的 `fit_transform / transform` 流程。

---

## 二、Complete Case（完整案例分析）

### 2.1 代码

```python
if method_name == 'Complete Case':
    # 删除任何含缺失的行
    train_mask = X_train.isnull().any(axis=1)
    test_mask = X_test.isnull().any(axis=1)

    X_train_imp = X_train[~train_mask].copy()
    y_train_imp = y_train[~train_mask]
    X_test_imp = X_test[~test_mask].copy()
    y_test_imp = y_test[~test_mask]

    n_dropped_train = train_mask.sum()
    n_dropped_test = test_mask.sum()
```

### 2.2 逐行解析

#### `X_train.isnull().any(axis=1)` —— 按行检查是否有任何缺失值

这一行要分两步理解：

1. **`X_train.isnull()`**：返回一个与 `X_train` 形状相同的布尔 DataFrame，缺失值位置为 `True`，否则为 `False`。
2. **`.any(axis=1)`**：`any` 表示"是否有任何一个为 True"，`axis=1` 表示**沿列方向**遍历（也就是对每一行内部的所有列做判断）。

最终返回一个**一维布尔 Series**，长度等于行数。如果某一行**至少有一个**缺失值，对应位置就是 `True`；只有当一行所有列都不缺失时，才是 `False`。

> **关键对比**：
> - `df.isnull().any(axis=0)`（默认）→ 按列判断，返回"每一列是否有缺失"，长度 = 列数。
> - `df.isnull().any(axis=1)` → 按行判断，返回"每一行是否有缺失"，长度 = 行数。
>
> 记忆口诀：**`axis=1` 是"跨列"，结果保留行**。

举例：假设 `X_train` 是

| Age  | year | Gender |
| ---- | ---- | ------ |
| 50   | 2020 | 1      |
| NaN  | 2021 | 0      |
| 60   | NaN  | 1      |
| 55   | 2019 | 0      |

那么 `X_train.isnull().any(axis=1)` 返回：

| 行索引 | 结果  |
| ------ | ----- |
| 0      | False |
| 1      | True  |
| 2      | True  |
| 3      | False |

#### `~train_mask` —— 取反

`~` 是 Pandas/Numpy 中的**按位取反运算符**，把 `True` 变 `False`，`False` 变 `True`。

- `train_mask` 为 `True` 的行 = **有缺失**的行（要删掉）。
- `~train_mask` 为 `True` 的行 = **没有缺失**的行（要保留）。

所以 `~train_mask` 就是"完整行"的掩码。

#### `X_train[~train_mask].copy()` —— 布尔索引 + 深拷贝

- **`X_train[~train_mask]`**：用布尔 Series 对 DataFrame 做索引，只保留 `True` 对应的行。这就是 Pandas 的**布尔索引（boolean indexing）**。
- **`.copy()`**：对结果做深拷贝，得到一个独立的新 DataFrame。后续对 `X_train_imp` 的修改（比如标准化）不会影响原始的 `X_train`。

> **小贴士**：在 Pandas 中，对切片结果做修改时常会触发 `SettingWithCopyWarning`。加 `.copy()` 是最简单的规避方式——明确告诉 Pandas"我要一份副本，可以随便改"。

#### `y_train_imp = y_train[~train_mask]` —— 标签同步删除

注意：**特征和标签必须同步删除**！如果 `X_train` 删了第 5 行，`y_train` 也必须删第 5 行，否则样本和标签会错位。

由于 `y_train` 是 NumPy 数组（在模块 1 中 `y = df_sample['target'].values`），布尔索引对 NumPy 数组同样适用。

#### `n_dropped_train = train_mask.sum()`

`train_mask` 是布尔 Series，`True` 当作 1，`False` 当作 0，`.sum()` 就是"被删掉的行数"。本例中训练集删了 8,730 行。

### 2.3 优缺点讨论

#### 优点

1. **简单**：不需要选择插补方法、不需要调参，一行代码搞定。
2. **不引入插补偏差**：所有保留下来的数据都是**真实观测值**，没有人为构造的成分。
3. **保留真实分布**：因为没填任何"假数据"，特征的均值、方差、相关性都是真实的。

#### 缺点

1. **样本量损失**：本例训练集从 56,000 行降到 47,270 行，**删除了 8,730 行，约 15.6%**。如果缺失率更高（比如 50%），样本量会锐减到不可用。
2. **统计效力下降**：样本越少，标准误越大，置信区间越宽，假设检验越难显著。
3. **如果缺失非 MCAR 会引入偏差**：
   - 如果缺失是 **MCAR（完全随机缺失）**：删除后剩余样本仍是总体的无偏样本，结论可信。
   - 如果缺失是 **MAR（有条件随机缺失）**或 **MNAR（非随机缺失）**：被删掉的样本和保留下来的样本系统性地不同，导致估计偏差。
   - 例如：如果"高龄患者更可能缺失 Raca.Color"，删行后样本会偏向低龄，模型在老年人上的预测就会失准。
4. **测试集也删行**：测试集从 24,000 降到 20,213，意味着模型评估时少了一部分样本，评估结果方差更大。

> **重要提醒**：Complete Case 是一个**基线（baseline）**，不是"推荐做法"。它让我们知道"如果不做任何插补，模型表现如何"，作为其他插补方法的参照。

---

## 三、Mean Imputation（均值插补）

### 3.1 代码

```python
imp = config['imputer']  # SimpleImputer(strategy='mean')
X_train_imp = pd.DataFrame(
    imp.fit_transform(X_train),
    columns=feature_names,
    index=X_train.index
)
X_test_imp = pd.DataFrame(
    imp.transform(X_test),
    columns=feature_names,
    index=X_test.index
)
y_train_imp = y_train
y_test_imp = y_test
```

### 3.2 `SimpleImputer(strategy='mean')` 的 `strategy` 参数

`SimpleImputer` 是 Scikit-learn 提供的简单插补器，`strategy` 参数决定用什么统计量填充缺失值：

| strategy 取值        | 用于什么类型 | 填充值                                  | 示例                                |
| -------------------- | ------------ | --------------------------------------- | ----------------------------------- |
| `'mean'`             | 数值型       | 该列的**均值**                          | Age 缺失 → 填 Age 的均值（如 58.3） |
| `'median'`           | 数值型       | 该列的**中位数**                        | Age 缺失 → 填 Age 的中位数          |
| `'most_frequent'`    | 数值/分类    | 该列的**众数**（出现次数最多的值）      | Gender 缺失 → 填最常见性别          |
| `'constant'`         | 数值/分类    | 一个固定常数（需配合 `fill_value` 参数）| 填 0 或 'Unknown'                   |

> **本例的细节**：代码里写的是 `strategy='mean'`，但 `desc` 里说"数值:均值 / 分类:众数"。这是因为本案例在模块 1 已经把分类特征 LabelEncoder 成了整数（0,1,2,…），所以 `SimpleImputer(strategy='mean')` 对所有列统一计算均值。对分类编码后的整数列，"均值"虽然不是严格意义上的众数，但效果接近——它会填一个接近最常见类别的浮点数。这是工程上的简化处理。

#### `mean` vs `median` 怎么选？

- **`mean`**：受异常值影响大。如果数据里有极端值（比如 Age 里有 200 岁），均值会被拉高。
- **`median`**：对异常值鲁棒。中位数只看中间位置，不受极端值影响。

经验法则：**数据偏态明显或存在异常值时，用 `median`；数据近似对称分布时，用 `mean`**。

### 3.3 `imp.fit_transform(X_train)` —— 拟合并变换训练集

`fit_transform` 是 Scikit-learn 的标准方法，等价于先 `fit` 再 `transform`：

1. **`fit`**：扫描 `X_train` 的每一列，**计算并保存**该列的均值（如果是 `strategy='mean'`）。这些均值存在 `imp.statistics_` 属性里。
2. **`transform`**：把 `X_train` 中的缺失值替换成对应列的均值，返回一个**新的 NumPy 数组**。

> **关键理解**：`fit_transform` 返回的是 NumPy 数组，**丢失了列名和索引**。所以代码用 `pd.DataFrame(..., columns=feature_names, index=X_train.index)` 把它重新包装成 DataFrame，恢复列名和索引。

### 3.4 `imp.transform(X_test)` —— 测试集只变换，不拟合

这一行是**整个模块最关键的一行**，体现了机器学习工程的核心原则。

> **🚨 重点：fit_transform vs transform —— 防止数据泄露！**
>
> - **训练集**用 `fit_transform`：从训练数据中**学习**插补参数（均值），然后应用。
> - **测试集**只能用 `transform`：**直接使用训练集学到的参数**，不能重新计算。
>
> 为什么？如果对测试集 `fit`，相当于让模型"偷看"了测试集的分布信息，这叫**数据泄露（data leakage）**。结果是：
> 1. 评估指标会过于乐观（因为测试集的信息已经"渗透"到了预处理中）。
> 2. 部署到真实场景时，模型遇到全新的数据会表现下降。
>
> **正确做法**：所有"从数据中学习参数"的操作（插补、标准化、PCA、特征选择等），都只能用训练集 `fit`，然后用同一个对象对测试集 `transform`。

举个反例：如果测试集里 Age 的均值是 60，而训练集是 58，对测试集 `fit` 会用 60 填充，但部署时遇到的新数据均值可能是 55——模型在训练时"以为"缺失值会被填 60，实际却填了 55，分布不一致导致性能下降。

### 3.5 `y_train_imp = y_train` —— 标签不删

和 Complete Case 不同，均值插补**不删除任何样本**，所以标签 `y_train` 原封不动。这是均值插补的一个优点：**保留全部样本量**。

### 3.6 重点讨论：均值插补为什么"不好"？

虽然均值插补简单快速，但它在统计学上有三个严重缺陷：

#### 缺陷 1：均值点产生"假峰"

所有缺失值都被替换成**同一个常数**（该列均值）。如果缺失率不低（比如 15%），这 15% 的值全堆在均值这一个点上，直方图上会出现一个**人为的尖峰**，破坏原始分布的形状。

#### 缺陷 2：人为压缩方差

方差度量数据的离散程度。把一部分值全部替换成均值，相当于把这部分数据"拉到中心"，整体方差必然变小。

数学上：假设原变量 X 的方差是 σ²，缺失率是 p。把 p 比例的值替换成均值后，新方差约为 σ² × (1−p)（粗略估计）。本例 Raca.Color 缺失率 15.31%，方差会被压缩约 15%。

> **📌 重点：均值插补压缩方差**
>
> 方差被低估 → 模型会**高估自己对预测的信心** → 校准曲线偏离对角线 → Brier Score 变差。
>
> 这就是为什么本案例中 Mean Imputation 的 Brier Score（0.1466）比 MICE（0.1462）略差——虽然差距小，但方向一致。

#### 缺陷 3：破坏特征间协方差结构

如果 Age 和 year 有一定相关性，把 Age 的缺失值全填成均值，相当于"切断"了这些样本中 Age 和 year 的关系。协方差矩阵被扭曲，影响线性模型（逻辑回归）的系数估计。

> **小贴士**：均值插补适合**缺失率很低**（比如 < 5%）的场景，这时压缩方差的影响可以忽略。本例 Raca.Color 缺失率 15.31%，已经偏高，均值插补的副作用会比较明显。

---

## 四、KNN Imputation（K 近邻插补）

### 4.1 代码

```python
'KNN Imputation': {
    'desc': 'KNN 插补 (n_neighbors=5)',
    'imputer': KNNImputer(n_neighbors=5, weights='distance'),
    'color': '#e67e22'
}
```

### 4.2 `KNNImputer` 参数详解

#### `n_neighbors=5` —— 用 5 个最近邻

对每一个含缺失值的样本，KNN 插补会在**特征空间**中找 5 个**最相似**的样本（即欧氏距离最小的 5 个），然后用这 5 个样本的对应特征值来填充缺失。

- **k 太小**（如 1）：容易受噪声影响，插补值不稳定。
- **k 太大**（如 50）：邻居太远，相似性弱，退化为接近均值插补。
- **k=5** 是一个常用的折中值。

#### `weights='distance'` —— 距离加权

决定如何把 k 个邻居的值组合成最终的插补值：

| weights 取值   | 含义                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------ |
| `'uniform'`    | 所有邻居**等权重**，简单取平均。                                                           |
| `'distance'`   | **距离倒数加权**，距离越近的邻居权重越大。比如距离为 1 的邻居权重是距离为 2 的邻居的 2 倍。|

本例选 `'distance'`，因为医学数据中"相似患者"的信息比"远距离患者"更可靠。

> **小贴士**：KNN 插补要求所有特征都是数值型。本案例在模块 1 已经把分类特征 LabelEncoder 成整数，所以可以直接用 KNNImputer。但要注意：LabelEncoder 后的整数（0,1,2,3,4）之间的欧氏距离未必有临床意义——"人种 0"和"人种 4"的距离不一定比"人种 0"和"人种 1"远。这是 KNN 用于混合类型数据时的固有局限。

### 4.3 KNN 插补原理（算法流程）

对于每一个含缺失值的样本 x：

1. **找邻居**：在所有"非缺失特征"上计算 x 与训练集中其他样本的欧氏距离，找出距离最近的 k 个样本。
2. **加权平均**：对 x 缺失的那个特征，用 k 个邻居在该特征上的值做加权平均（权重由 `weights` 决定）。
3. **填入**：把加权平均的结果填入 x 的缺失位置。

关键点：**只对缺失特征做插补，非缺失特征保持原值**。而且找邻居时**只用非缺失特征**计算距离。

### 4.4 重点讨论：KNN 的计算代价

#### 时间复杂度 O(n²×d)

- 对每一个含缺失值的样本，都要和所有其他样本计算距离 → O(n) 次距离计算。
- 假设有 m 个样本含缺失值，总计算量是 O(m × n × d)，其中 d 是特征数。
- 最坏情况 m ≈ n，复杂度退化为 **O(n²×d)**。

#### 本例的实际耗时

| 方法               | 耗时    | 相对倍数 |
| ------------------ | ------- | -------- |
| Complete Case      | 0.0s    | 1×       |
| Mean Imputation    | 0.0s    | 1×       |
| **KNN Imputation** | **19.5s** | **~195×** |
| MICE Imputation    | 0.1s    | 10×      |

KNN 比其他方法慢了近 200 倍！原因是：

1. 本例训练集 56,000 条，含缺失值的样本约 8,500 条（主要是 Raca.Color 缺失）。
2. 每个缺失样本要和 56,000 条样本计算距离 → 8,500 × 56,000 ≈ 4.76 亿次距离计算。
3. 每次距离计算涉及 5 个特征 → 约 24 亿次浮点运算。

#### 大规模数据不可行

如果数据量到 100 万条，KNN 插补的耗时会是本例的 (1000000/56000)² ≈ 320 倍，即约 6,240 秒（1.7 小时）。在工业级场景下完全不可行。

> **工程建议**：大数据集上如果一定要用 KNN 插补，可以考虑：
> 1. 用 `sklearn.neighbors.NearestNeighbors` 配合 KD-Tree 或 Ball-Tree 加速近邻搜索。
> 2. 先对数据做聚类，在簇内做 KNN 插补。
> 3. 用近似最近邻（ANN）算法，如 FAISS、Annoy。
> 4. 退而求其次，用 MICE，它在保留分布结构的同时计算量小得多。

---

## 五、MICE Imputation（多重插补）

### 5.1 代码

```python
'MICE Imputation': {
    'desc': 'MICE 多重插补 (IterativeImputer)',
    'imputer': IterativeImputer(max_iter=10, random_state=RANDOM_STATE),
    'color': '#9b59b6'
}
```

### 5.2 什么是 MICE？

**MICE** = **Multivariate Imputation by Chained Equations**（链式方程多重插补）。

在 Scikit-learn 中，MICE 由 `IterativeImputer` 实现。注意导入方式特殊：

```python
from sklearn.experimental import enable_iterative_imputer  # 必须先导入这个
from sklearn.impute import IterativeImputer
```

`enable_iterative_imputer` 是一个"实验性功能开关"，因为 Scikit-learn 官方认为这个 API 还可能变动，所以放在 `experimental` 命名空间下。

### 5.3 MICE 原理：迭代式回归插补

MICE 的核心思想是：**把每个有缺失的变量当作回归目标，用其他变量作为特征来预测缺失值**。具体流程：

1. **初始化**：对每个有缺失的列，先用简单方法（如均值/中位数）填一遍，得到一个"完整"的数据集。
2. **迭代**（round-robin）：
   - 选一个有缺失的列作为目标 y，其他列作为特征 X。
   - 用原始有观测值的行训练一个回归模型（默认是 `BayesianRidge`）。
   - 用这个模型预测缺失位置的值，替换掉初始化时的填充值。
   - 对下一个有缺失的列重复上述过程。
   - 所有列都处理一遍，称为"一轮"。
3. **收敛判断**：重复多轮，直到达到 `max_iter` 或参数估计收敛。

> **关键理解**：MICE 是"迭代式"的——每一轮的预测都会更新数据，下一轮基于更新后的数据再预测。随着迭代进行，插补值会逐渐稳定下来。

### 5.4 `IterativeImputer` 参数详解

#### `max_iter=10` —— 最多迭代 10 轮

- 每一轮对所有有缺失的列各做一次回归插补。
- 10 轮通常足够收敛。如果数据复杂，可以增加到 20 或 50，但耗时也会增加。
- 如果在 10 轮内就收敛，会提前停止。

#### `random_state=RANDOM_STATE` —— 固定随机种子

- MICE 在初始化和回归预测时会引入随机性（默认的 `BayesianRidge` 会采样后验分布）。
- 固定 `random_state=42` 保证结果可复现——每次运行代码，插补值都一样。
- 这对教学和科研很重要：别人复现你的实验时能得到完全相同的结果。

> **小贴士**：`RANDOM_STATE = 42` 是机器学习界的"梗"，源自《银河系漫游指南》——42 是"生命、宇宙及一切的终极答案"。

### 5.5 重点讨论：为什么 MICE 是医学研究首选？

在医学统计和流行病学领域，MICE 是处理缺失值的**金标准**之一。原因有四：

#### 1. 保留变量间关系

MICE 用其他变量作为特征预测缺失值，所以**插补值会保留变量间的相关性**。例如，如果 Age 和 Diagnostic.means 相关，MICE 插补 Age 时会考虑 Diagnostic.means 的值，插补结果更符合真实关系。

#### 2. 不低估方差

MICE 默认的 `BayesianRidge` 会从后验分布中**采样**，而不是取均值。这意味着每次插补都引入了适当的随机性，**方差不会被压缩**。这是 MICE 相对均值插补的最大优势。

> **📌 重点：MICE 不低估方差**
>
> 均值插补把所有缺失填成同一个常数 → 方差被压缩。
> MICE 从后验分布采样 → 每个缺失值都不同 → 方差被保留。
>
> 这就是为什么 MICE 的 Brier Score（0.1462）在本案例中是最好的——校准度最优。

#### 3. MAR 假设下无偏

MICE 在 **MAR（Missing At Random，有条件随机缺失）** 假设下能给出**无偏估计**。MAR 是指：缺失的概率只依赖于**观测到的变量**，不依赖于未观测的值本身。

例如："高龄患者更可能缺失 Raca.Color"——如果 Age 是观测到的，那么这种缺失就是 MAR，MICE 能正确处理。

#### 4. 反映插补不确定性

真正的 MICE 会产生**多个**插补数据集（通常 5–20 个），分别建模后用 Rubin 规则合并结果，从而把"插补本身的不确定性"纳入标准误估计。

> **本例的简化**：Scikit-learn 的 `IterativeImputer` 只产生**一个**插补数据集（单次插补），不是严格意义上的"多重"插补。要做真正的多重插补，需要：
> - 多次运行 `IterativeImputer`（每次不同 `random_state`），或
> - 用 `statsmodels` 的 `MICE` 类，或
> - 用 R 语言的 `mice` 包。
>
> 但即便只做单次，`IterativeImputer` 的效果也比均值插补好得多，所以本案例用它作为 MICE 的代表。

---

## 六、标准化部分

### 6.1 代码

```python
scaler = StandardScaler()
if method_name == 'Complete Case':
    num_cols_train = [c for c in numerical_features if c in X_train_imp.columns]
else:
    num_cols_train = numerical_features

if num_cols_train:
    X_train_imp[num_cols_train] = scaler.fit_transform(X_train_imp[num_cols_train])
    X_test_imp[num_cols_train] = scaler.transform(X_test_imp[num_cols_train])
```

### 6.2 `StandardScaler()` —— 标准化（Z-score 变换）

标准化公式：

$$
x_{\text{new}} = \frac{x - \mu}{\sigma}
$$

其中 μ 是均值，σ 是标准差。标准化后：

- 均值 = 0
- 标准差 = 1
- 量纲被消除（单位变成"标准差倍数"）

### 6.3 为什么要标准化？

逻辑回归使用 `lbfgs` 求解器，它基于梯度下降。如果特征量纲差异大（比如 Age 是 0–100，Gender 是 0–1），梯度下降会沿着大数值特征的方向震荡，收敛慢甚至不收敛。标准化后所有特征在同一量纲下，优化更稳定。

### 6.4 `scaler.fit_transform(X_train_imp[num_cols_train])` —— 只对数值列标准化

注意 `num_cols_train` 是 `numerical_features`（即 `['Age', 'year']`），**不包含分类列**。

> **🚨 重点：为什么只标准化数值列，不标准化分类列？**
>
> 分类特征在模块 1 已经被 `LabelEncoder` 转换成整数（0, 1, 2, …）。这些整数**只是类别的标签，没有数值意义**：
> - "人种 0"和"人种 4"的距离不一定比"人种 0"和"人种 1"远。
> - "诊断方式 2"不是"诊断方式 1"的两倍。
>
> 对这些整数做标准化（减均值除标准差）**没有物理意义**，反而会破坏标签的整数性，让结果变成难以解释的浮点数。
>
> 正确做法是：**数值列标准化，分类列保持整数编码不变**。如果想让分类列也"无量纲"，应该用 One-Hot Encoding，而不是标准化。

### 6.5 `scaler.transform(X_test_imp[num_cols_train])` —— 测试集用训练集参数

和插补器一样，标准化器也遵循"训练集 fit，测试集 transform"的原则：

- `scaler.fit_transform(X_train_imp[...])`：从训练集计算 μ 和 σ，存入 `scaler.mean_` 和 `scaler.scale_`，然后变换训练集。
- `scaler.transform(X_test_imp[...])`：**直接用训练集的 μ 和 σ** 变换测试集。

> **🚨 重点：再次强调防止数据泄露！**
>
> 如果对测试集 `fit`，会用测试集自己的均值和标准差变换测试集，导致：
> 1. 训练集和测试集的标准化参数不一致 → 分布错位 → 模型性能下降。
> 2. 测试集信息泄露到预处理中 → 评估指标过于乐观。
>
> **铁律**：任何 `fit` 操作都只能用训练集！

### 6.6 Complete Case 的特殊处理

```python
if method_name == 'Complete Case':
    num_cols_train = [c for c in numerical_features if c in X_train_imp.columns]
else:
    num_cols_train = numerical_features
```

为什么 Complete Case 要做这个判断？因为 Complete Case 删行后，**理论上某些列可能全部缺失被删光**（虽然本例不会发生）。这行代码是一个防御性写法：只标准化"实际存在于 `X_train_imp` 中的数值列"。

> **小贴士**：这种防御性编程在工程上是好习惯。即使当前数据不会触发边界情况，未来数据变了也不会报错。

---

## 七、实际运行结果

### 7.1 四种方法对比表

| 方法               | 训练集样本量 | 测试集样本量 | 耗时    | AUC    | Recall (VIVO) | Brier Score |
| ------------------ | ------------ | ------------ | ------- | ------ | ------------- | ----------- |
| Complete Case      | 47,270       | 20,213       | 0.0s    | 0.8644 | 0.9240        | 0.1493      |
| Mean Imputation    | 56,000       | 24,000       | 0.0s    | 0.8632 | 0.9236        | 0.1466      |
| KNN Imputation     | 56,000       | 24,000       | 19.5s   | 0.8628 | 0.9237        | 0.1467      |
| MICE Imputation    | 56,000       | 24,000       | 0.1s    | 0.8637 | 0.9223        | 0.1462      |

### 7.2 结果解读

#### 样本量

- Complete Case 损失了 8,730 行训练数据（15.6%）和 3,787 行测试数据（15.8%）。
- 其他三种方法都保留了全部 56,000 + 24,000 = 80,000 条样本。

#### 耗时

- Complete Case 和 Mean Imputation 几乎瞬时（0.0s）——前者只是布尔索引，后者只是计算均值。
- KNN 最慢（19.5s），因为要计算样本间距离。
- MICE 较快（0.1s），因为只有 5 个特征，迭代 10 轮的计算量不大。

#### 性能指标

- AUC：四种方法非常接近（0.8628–0.8644），说明在本案例中插补方式对排序能力影响不大。
- Brier Score：MICE 最好（0.1462），Complete Case 最差（0.1493）。这印证了"MICE 保留方差 → 校准更好"的理论。
- Recall：四种方法都在 0.92 左右，差异很小。

> **关键观察**：本案例的结论是"插补方式对模型性能影响有限"，原因是 Raca.Color 这个特征本身对预测的贡献不大（缺失的只有这一个特征）。如果缺失的是 Age 这种强相关特征，不同插补方法的差异会显著放大。

---

## 八、小贴士

1. **先 EDA 再插补**：在插补前一定要先做缺失值分析（见模块 1），搞清楚缺失率、缺失机制（MCAR/MAR/MNAR），再决定插补策略。
2. **小数据用 KNN，大数据用 MICE**：KNN 在小数据上效果好且直观，但 O(n²) 限制了它的扩展性。MICE 在大数据上更快且统计性质更优。
3. **缺失率 > 50% 的列考虑删除**：插补再多也是"编造"，不如直接删掉这一列。
4. **分类变量优先用众数或 MICE**：均值插补对分类变量意义不大，众数（`most_frequent`）或 MICE 更合适。
5. **报告插补方法**：在论文或报告中一定要说明用了什么插补方法，这是可复现性的要求。
6. **固定随机种子**：MICE、KNN 等涉及随机性的方法，务必固定 `random_state`，否则结果不可复现。

---

## 九、常见问题

### Q1：为什么 Complete Case 的 AUC 反而最高？

A：本案例中 Complete Case 的 AUC（0.8644）略高于其他方法，但这**不代表 Complete Case 更好**。原因有二：

1. Complete Case 删掉的是"有缺失的样本"，这些样本可能本身就更难预测（缺失本身携带着信息）。删掉后，剩下的样本"更干净"，模型看起来表现好，但这是**评估偏差**。
2. Complete Case 的测试集也删了行，评估用的样本和真实部署场景不一致。

正确的解读是：**AUC 差异很小（0.0016），在统计噪声范围内**，不能下"Complete Case 更优"的结论。

### Q2：KNN 这么慢，为什么不直接用 Mean？

A：KNN 慢但能**保留局部数据结构**。在 Age 这种连续变量上，KNN 插补的值会随邻居变化，分布更真实；Mean 插补则把所有缺失填成同一个数，破坏分布。如果只看 AUC，差异不大；但如果看分布还原度（见 `06e_distribution_impact.png`），KNN 明显优于 Mean。

### Q3：MICE 的 `max_iter` 设多少合适？

A：默认 10 通常够用。如果发现插补值还在变化（不收敛），可以增加到 20 或 50。但要注意：`max_iter` 越大，耗时越长。本例 5 个特征、10 轮迭代只要 0.1s，如果是 100 个特征，可能要几分钟。

### Q4：为什么不对分类列做 One-Hot 再插补？

A：One-Hot 后每个类别变成一列，KNN/MICE 在高维稀疏矩阵上效果会下降（维度灾难）。本案例选择"LabelEncoder + 整数插补"是工程上的折中。如果分类变量很重要，可以考虑 One-Hot + Mean/Most_Frequent 插补。

### Q5：`fit_transform` 和 `fit` + `transform` 有区别吗？

A：在大多数 Scikit-learn 估计器中，`fit_transform(X)` 等价于 `fit(X).transform(X)`。但某些估计器（如 `IterativeImputer`）的 `fit_transform` 做了优化，比分开调用更快。所以**推荐用 `fit_transform`**。

### Q6：测试集也有缺失值，Complete Case 删了测试集的行，这样评估公平吗？

A：这是一个有争议的问题。删测试集行意味着评估时少了一部分样本，但保留了"真实观测值"。如果用插补方法处理测试集，评估的是"插补后的预测能力"，更接近真实部署场景。本案例为了公平对比，Complete Case 在测试集上也删行，其他方法在测试集上插补。**两种评估方式各有道理，关键是要在报告里说清楚**。

---

## 十、本模块小结

本模块详细讲解了四种缺失值插补策略的代码实现和原理：

1. **Complete Case（完整案例分析）**：删除含缺失的行，简单但损失样本量。适合缺失率低且 MCAR 的场景，本例损失 15.6% 样本。
2. **Mean Imputation（均值插补）**：用列均值填充缺失，简单快速但**压缩方差、破坏分布**。适合缺失率极低的场景。
3. **KNN Imputation（K 近邻插补）**：用相似样本的值填充，保留局部结构但**计算代价 O(n²×d)**，本例耗时 19.5s。
4. **MICE Imputation（多重插补）**：迭代式回归插补，**保留变量间关系、不低估方差**，是医学研究首选，本例耗时仅 0.1s。

核心工程原则：

- **`fit_transform` 训练集，`transform` 测试集**——防止数据泄露，这是机器学习工程的铁律。
- **数值列标准化，分类列保持整数编码**——LabelEncoder 后的整数无量纲意义，标准化会破坏其语义。
- **用字典组织配置**——配置与逻辑分离，便于扩展和维护。

实践建议：

- 缺失率 < 5%：Mean/Median 即可。
- 缺失率 5%–30%：优先 MICE，数据量小可试 KNN。
- 缺失率 > 50%：考虑删除该列。
- 医学/科研场景：MICE 是金标准。
- 工业级大数据：MICE 或近似 KNN（FAISS）。

> **下一步学习**：模块 3 将基于本模块插补后的数据集，训练逻辑回归模型并比较四种插补方法在 AUC、Recall、Brier Score 上的表现，并通过可视化揭示插补方式如何影响模型决策边界。
