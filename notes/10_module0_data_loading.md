# 模块 0：数据加载与目标变量创建

> 本模块是案例教程 2「统计学分析」的起点，承接案例教程 1（EDA）。在开始任何统计检验之前，我们必须先把数据加载进内存、把字符串形式的目标变量（`Status.Vital`）转换成数值标签，并正确处理缺失值。

***

## 学习目标

学完本模块后，你将能够：

1. **理解每个 import 语句的作用**：特别是 `from scipy import stats` 与 `from scipy.stats import mannwhitneyu, chi2_contingency` 这两种导入方式的区别，以及为什么统计学分析需要这两个具体函数。
2. **掌握路径配置与目录创建**：理解 `os.path.join()` 和 `os.makedirs(..., exist_ok=True)` 的作用。
3. **正确读取 CSV 文件**：理解 `pd.read_csv()` 中 `low_memory=False` 和 `encoding='latin-1'` 两个关键参数。
4. **深入理解** **`map`** **vs** **`==`** **的本质区别**：这是本模块最重要的知识点——明白为什么 `map` 比 `==` 更安全，为什么 `==` 会把缺失值误判为 0。
5. **掌握** **`dropna(subset=...)`** **的精确过滤**：理解为什么要在 `subset` 中指定 `target` 而不是 `Status.Vital`。
6. **理解** **`.copy()`** **的作用**：明白为什么要在 `dropna` 后加 `.copy()`，以及它如何避免 `SettingWithCopyWarning`。
7. **掌握用** **`.mean()`** **计算比例的技巧**：理解为什么 `(df['target'] == 1).mean()` 能直接算出存活比例。

***

## 一、导入必要的库

统计学分析需要的数据分析库与案例教程 1 类似，但额外引入了 SciPy 的统计函数。下面是完整的导入代码：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency

warnings.filterwarnings('ignore')
```

### 1.1 逐行解释每个 import

#### `from scipy import stats`

**SciPy**（Scientific Python）是建立在 numpy 之上的科学计算库，`stats` 子模块提供了大量统计函数。

- `from scipy import stats` 这种写法表示**只导入** **`stats`** **子模块**，后续通过 `stats.xxx` 调用。
- 在本教程中用于：
  - `stats.normaltest()`：D'Agostino-Pearson 正态性检验
  - `stats.ttest_ind()`：独立样本 t 检验（正态数值变量）
  - `stats.mannwhitneyu()`：Mann-Whitney U 检验（非正态数值变量）—— 也可以通过这个子模块访问
- 这种导入方式的好处是：**可以统一通过** **`stats.`** **前缀访问所有统计函数**，代码可读性强，且后续如果需要其他统计函数（如 `stats.shapiro`、`stats.pearsonr`）无需修改导入语句。

#### `from scipy.stats import mannwhitneyu, chi2_contingency`

这是另一种导入方式：**直接从** **`scipy.stats`** **中导入两个具体的函数**，这样后续可以直接写 `mannwhitneyu(...)` 和 `chi2_contingency(...)`，不需要加 `stats.` 前缀。

- **`mannwhitneyu`**：Mann-Whitney U 检验（也叫 Wilcoxon 秩和检验），是一种**非参数检验**，用于比较两组独立样本的中位数是否相等。当数值变量不服从正态分布时，用它替代 t 检验。
- **`chi2_contingency`**：卡方独立性检验（Chi-square test of independence），用于检验**两个分类变量**是否独立。在本教程中用于检验每个分类特征与目标变量（存活/死亡）是否相关。

### 1.2 两种导入方式的区别与联系

你可能会疑惑：既然 `from scipy import stats` 已经能访问 `stats.mannwhitneyu` 和 `stats.chi2_contingency`，为什么还要再写一行 `from scipy.stats import mannwhitneyu, chi2_contingency`？

| 导入方式                                   | 调用语法                      | 优点             | 缺点                 |
| -------------------------------------- | ------------------------- | -------------- | ------------------ |
| `from scipy import stats`              | `stats.mannwhitneyu(...)` | 命名空间清晰，避免函数名冲突 | 调用时多写 `stats.` 前缀  |
| `from scipy.stats import mannwhitneyu` | `mannwhitneyu(...)`       | 调用简洁，代码更短      | 函数名直接进入当前命名空间，可能冲突 |

本教程**同时使用两种方式**的原因：

1. `from scipy import stats` 是为了调用 `stats.normaltest()`、`stats.ttest_ind()` 等函数，这些函数没有单独导入，统一通过 `stats.` 前缀访问。
2. `from scipy.stats import mannwhitneyu, chi2_contingency` 是因为这两个函数在代码中会被**频繁、直接**调用（每个数值特征和分类特征都要调用一次），直接导入可以让代码更简洁，减少 `stats.` 前缀的重复书写。

> 💡 **小贴士**：在实际项目中，建议遵循"**常用的、核心的函数直接导入，不常用的通过模块前缀访问**"的原则。这样既保持代码简洁，又避免命名空间污染。

### 1.3 为什么统计学分析需要这两个函数？

本教程的核心任务是：**找出与癌症患者存活/死亡显著相关的特征**。根据特征类型的不同，需要使用不同的统计检验方法：

| 特征类型 | 分布情况  | 使用的检验             | 对应函数                 |
| ---- | ----- | ----------------- | -------------------- |
| 数值型  | 正态分布  | 独立样本 t 检验         | `stats.ttest_ind()`  |
| 数值型  | 非正态分布 | Mann-Whitney U 检验 | `mannwhitneyu()`     |
| 分类型  | —     | 卡方独立性检验           | `chi2_contingency()` |

所以 `mannwhitneyu` 和 `chi2_contingency` 是本教程的**两大主力检验函数**，分别处理非正态数值变量和分类变量。至于正态数值变量用的 t 检验，通过 `stats.ttest_ind()` 调用即可，不需要单独导入。

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

### 2.1 路径配置

- `BASE_DIR`：项目根目录，所有路径都基于它拼接，方便统一管理。
- `DATA_PATH`：数据文件路径，指向 `data/cancer_data_eng.csv`（英文版癌症数据）。
- `IMG_DIR`：图片输出目录，统计图表会保存在这里。
- `RESULTS_DIR`：结果输出目录，统计检验结果（如 p 值表格）会保存在这里。

### 2.2 为什么用 `os.path.join()` 而不是字符串拼接？

```python
# 推荐写法
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")

# 不推荐写法
DATA_PATH = BASE_DIR + "/data/cancer_data_eng.csv"
```

`os.path.join()` 的优势：

1. **跨平台兼容**：Windows 用 `\` 作路径分隔符，Linux/Mac 用 `/`。`os.path.join()` 会自动选择当前系统的分隔符，而字符串拼接硬编码 `/` 在 Windows 上可能出问题。
2. **避免分隔符错误**：如果 `BASE_DIR` 末尾有 `/`，字符串拼接会变成 `//`，虽然多数系统能处理但不规范；`os.path.join()` 会自动处理这种情况。

### 2.3 `os.makedirs(..., exist_ok=True)` 详解

`os.makedirs()` 用于**递归创建目录**（包括所有不存在的父目录）。`exist_ok` 参数的含义：

- `exist_ok=False`（默认）：如果目录已存在，**抛出 FileExistsError 异常**。
- `exist_ok=True`：如果目录已存在，**不报错，直接跳过**。

本教程设置 `exist_ok=True` 是因为脚本可能多次运行，第一次运行时创建目录，后续运行时目录已存在，不希望报错中断。

> 💡 **小贴士**：在数据分析脚本的开头，提前创建好输出目录（`img`、`results`）是好习惯。否则脚本运行到一半才发现目录不存在，保存文件会失败，前面的计算就白做了。

***

## 三、加载数据

<br />

***

## 四、创建目标变量（本模块核心）

这是本模块**最重要**的部分。我们先看代码：

```python
# 创建目标变量: VIVO=1 (alive), MORTO=0 (dead)
# 使用 map 而非 == 比较，避免缺失值被误判为 MORTO(0)
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

### 4.1 这行代码做了什么

它创建了一个新列 `target`，把字符串标签 `Status.Vital` 映射成数值：

| Status.Vital | map 映射    | target |
| ------------ | --------- | ------ |
| VIVO         | 1         | 1      |
| MORTO        | 0         | 0      |
| NaN（缺失）      | NaN（保持缺失） | NaN    |

### 4.2 `map()` 方法详解

`Series.map()` 是 pandas 的一个方法，用于**逐元素映射**。它接受一个字典作为参数，把 Series 中的每个值按照字典的键值对进行替换。

```python
df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

- 字典 `{'VIVO': 1, 'MORTO': 0}` 定义了映射规则：`VIVO` → 1，`MORTO` → 0。
- 对于字典中**没有定义的值**（包括 `NaN`），`map()` 会返回 `NaN`。
- 这是 `map()` 与 `==` 比较**最关键的区别**。

### 4.3 重点对比：`map` vs `==`（本模块核心知识点）

在案例教程 1 中，我们用的是：

```python
# 案例教程 1 的写法
df['target'] = (df['Status.Vital'] == 'VIVO').astype(int)
```

而在案例教程 2 中，我们改用：

```python
# 案例教程 2 的写法
df['target'] = df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

#### 两种方法的输出对比

| Status.Vital 原值 | `(== 'VIVO').astype(int)` | `.map({'VIVO':1, 'MORTO':0})` |
| --------------- | ------------------------- | ----------------------------- |
| VIVO            | 1                         | 1                             |
| MORTO           | 0                         | 0                             |
| NaN（缺失）         | **0** ⚠️                  | **NaN** ✅                     |

#### 为什么 `==` 会把 NaN 当作 0？

这是 Python/Pandas 中一个容易踩坑的特性。当用 `==` 比较时：

```python
NaN == 'VIVO'   # 结果是 False（任何与 NaN 的比较都返回 False）
```

在 Python 中，**任何与** **`NaN`** **的相等比较都返回** **`False`**，包括 `NaN == NaN` 也是 `False`！这是 IEEE 754 浮点数标准的规定。

所以执行 `(df['Status.Vital'] == 'VIVO')` 时：

| Status.Vital | `== 'VIVO'` 的结果          |
| ------------ | ------------------------ |
| VIVO         | True                     |
| MORTO        | False                    |
| NaN          | **False**（被误判为"不是 VIVO"） |

接着 `.astype(int)` 把布尔值转成整数：

| `== 'VIVO'` 的结果 | `.astype(int)` 的结果 |
| --------------- | ------------------ |
| True            | 1                  |
| False           | 0                  |
| False（来自 NaN）   | **0**（被误判为 MORTO）  |

**问题就在这里**：缺失值 `NaN` 被错误地转换成了 `0`（即 MORTO/死亡）！

#### 为什么 `map` 更安全？

`map()` 的行为不同：对于字典中**没有定义的值**，它返回 `NaN`，而不是默认值。

```python
df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})
```

- `VIVO` 在字典中 → 返回 1
- `MORTO` 在字典中 → 返回 0
- `NaN` **不在字典中** → 返回 `NaN`（保持缺失状态）

这样，缺失值就**保持为** **`NaN`**，后续可以用 `dropna()` 正确地把它们过滤掉。而如果用 `==`，缺失值已经被"污染"成了 0，`dropna()` 就无法识别它们了。

> 💡 **核心知识点（本模块最重要）**：
>
> **`map`** **vs** **`==`** **的本质区别在于对缺失值的处理：**
>
> - `==` 比较：`NaN == 'VIVO'` 为 `False`，缺失值被**误判为 0**（MORTO），后续无法过滤。
> - `map` 映射：缺失值不在字典中，**保持为** **`NaN`**，后续 `dropna` 能正确过滤。
>
> **结论**：当原始标签列存在缺失值时，**必须用** **`map`** **而不是** **`==`** 来创建目标变量。这是案例教程 2 相对案例教程 1 的关键改进。
>
> 在案例教程 1 中，由于先做了 `dropna(subset=['Status.Vital'])` 再创建 target，所以 `==` 不会出问题（因为缺失值已经被删了）。但在案例教程 2 中，我们**先创建 target 再 dropna**，这个顺序变化就要求必须用 `map`。

### 4.4 为什么案例教程 2 改变了处理顺序？

| 教程       | 处理顺序                                | 用的方法  | 是否安全              |
| -------- | ----------------------------------- | ----- | ----------------- |
| 案例 1     | 先 `dropna(Status.Vital)`，再创建 target | `==`  | ✅ 安全（缺失值已删）       |
| 案例 2     | 先创建 target，再 `dropna(target)`       | `map` | ✅ 安全（NaN 保持）      |
| 案例 2（假设） | 先创建 target，再 `dropna(target)`       | `==`  | ❌ 不安全（NaN 被误判为 0） |

案例教程 2 改变顺序的原因是：先创建 target 再 dropna，可以**统一用 target 列管理缺失值**，代码逻辑更清晰。但这个顺序要求创建 target 时必须用 `map`。

> ⚠️ **常见问题**：如果我先 `dropna` 再创建 target，是不是就可以用 `==` 了？
>
> 是的！如果在创建 target 之前已经删除了 `Status.Vital` 的缺失值，那么用 `==` 是安全的（因为已经没有 NaN 了）。但本教程为了教学目的，特意演示了"先创建 target 再 dropna"的流程，以强调 `map` 的重要性。实际项目中，两种顺序都可以，关键是**方法要和顺序匹配**。

***

## 五、过滤缺失样本

```python
# 仅保留目标变量非缺失的样本
df_model = df.dropna(subset=['target']).copy()
```

### 5.1 `dropna(subset=['target'])` 详解

`dropna()` 是 pandas 删除缺失值的方法，`subset` 参数指定**只检查哪些列**。

- `subset=['target']`：只检查 `target` 这一列，如果这一列是 `NaN`，就删除整行。
- 为什么用 `target` 而不是 `Status.Vital`？因为前面用 `map` 创建 target 后，缺失值已经"转移"到了 target 列（以 NaN 形式）。直接对 target 做 dropna 即可。

> 💡 **小贴士**：`subset` 参数非常有用。实际数据中，几乎每行都或多或少有缺失值，如果无脑 `dropna()` 会把数据删光。用 `subset` 可以精确控制只针对关键列（比如目标变量）删除。

### 5.2 `.copy()` 的作用——避免 SettingWithCopyWarning

注意代码中 `dropna()` 后面跟着 `.copy()`。这是一个非常重要的细节。

#### 什么是 SettingWithCopyWarning？

当你在 pandas 中对一个"可能是视图（view）也可能是副本（copy）"的对象做修改时，pandas 会发出 `SettingWithCopyWarning` 警告，提示你：**你的修改可能没有生效，或者可能意外修改了原始数据**。

```python
# 危险写法（可能触发警告）
df_model = df.dropna(subset=['target'])   # df_model 可能是视图
df_model['new_col'] = ...                  # 修改可能不生效，或影响 df

# 安全写法
df_model = df.dropna(subset=['target']).copy()  # df_model 是独立副本
df_model['new_col'] = ...                       # 修改只影响 df_model，不影响 df
```

#### 为什么 `dropna()` 的结果可能是视图？

pandas 的某些操作（如 `dropna`、`loc` 切片）返回的可能是原始 DataFrame 的**视图**（view，共享同一块内存），也可能是**副本**（copy，独立的内存）。pandas 内部无法保证一定返回哪种，所以会发出警告。

#### `.copy()` 的解决方案

显式调用 `.copy()` 会强制创建一个**独立的副本**，这样后续对 `df_model` 的任何修改都**不会影响原始的** **`df`**，也不会触发警告。

> ⚠️ **常见问题**：什么时候需要加 `.copy()`？
>
> 经验法则：**当你打算后续修改一个从原 DataFrame 派生出来的子集时，就应该加** **`.copy()`**。比如：
>
> - `df_model = df.dropna(...).copy()`  ✅ 后续要给 df\_model 加新列
> - `subset = df[df['col'] > 0].copy()`  ✅ 后续要修改 subset
>
> 如果你只是**读取**子集而不修改，可以不加 `.copy()`（节省内存）。
>
> 本教程中 `df_model` 后续会被大量修改（添加分析结果列、做分组操作等），所以必须加 `.copy()`。

***

## 六、打印数据概况

```python
print(f"    原始样本量: {len(df):,}")
print(f"    有标签样本: {len(df_model):,}")
print(f"      VIVO(存活): {(df_model['target'] == 1).sum():,}  ({(df_model['target'] == 1).mean() * 100:.2f}%)")
print(f"      MORTO(死亡): {(df_model['target'] == 0).sum():,}  ({(df_model['target'] == 0).mean() * 100:.2f}%)")
```

### 6.1 `len(df)` 和 `:,` 格式化

- `len(df)` 返回 DataFrame 的行数。
- `f"{len(df):,}"` 中的 `:,` 是数字格式化语法，表示**千位分隔符**。比如 `1778176` 会显示为 `1,778,176`，更易读。

### 6.2 `(df_model['target'] == 1).sum()` 计算存活人数

这里用 `==` 是安全的，因为 `df_model` 已经经过 `dropna`，**不再有缺失值**，所以 `==` 不会误判。

- `df_model['target'] == 1` 返回一个布尔 Series（True/False）。
- `.sum()` 对布尔 Series 求和时，**True 当作 1，False 当作 0**，所以求和结果就是 True 的个数，即 target==1（存活）的样本数。

### 6.3 `(df_model['target'] == 1).mean()` 计算存活比例（重要技巧）

这是一个非常优雅的技巧。`.mean()` 对布尔 Series 求均值时，同样是 **True=1，False=0**，所以：

$$\text{mean} = \frac{\text{True 的个数} \times 1 + \text{False 的个数} \times 0}{\text{总个数}} = \frac{\text{True 的个数}}{\text{总个数}} = \text{True 的比例}$$

也就是说，**布尔 Series 的均值 = True 的比例**。

举例：

| target | (target == 1) |
| ------ | ------------- |
| 1      | True          |
| 0      | False         |
| 1      | True          |
| 0      | False         |
| 1      | True          |

`(target == 1).mean() = (1+0+1+0+1) / 5 = 3/5 = 0.6`，即 60% 是存活。

`* 100` 把比例转成百分比，`:.2f` 保留两位小数，所以最终显示 `60.00%`。

> 💡 **小贴士**：用 `.mean()` 计算布尔 Series 的比例，是 pandas 中非常常用的技巧。它比 `len(df[df['col'] == 1]) / len(df)` 更简洁、更高效（向量化操作）。记住：**布尔 Series 的 sum = True 的个数，布尔 Series 的 mean = True 的比例**。

### 6.4 为什么这里用 `==` 又安全了？

你可能会问：前面刚说 `==` 会误判 NaN，为什么这里又用 `==`？

关键区别在于**数据是否还有缺失值**：

- 创建 target 时：`df['Status.Vital']` 有缺失值，用 `==` 会误判 → 必须用 `map`。
- 打印统计时：`df_model['target']` 已经经过 `dropna`，**没有缺失值**，用 `==` 是安全的。

> 💡 **小贴士**：`==` 本身没有错，错的是"在有缺失值的数据上用 `==`"。只要数据已经清洗过（无缺失），`==` 是完全安全的，而且比 `map` 更简洁。

***

## 七、本模块运行结果

运行本模块代码后，控制台输出如下：

```
======================================================================
案例教程 2: 统计学分析
======================================================================

[0] 加载数据...
    原始样本量: 1,778,176
    有标签样本: 209,758
      VIVO(存活): 85,896  (40.95%)
      MORTO(死亡): 123,862  (59.05%)
```

### 关键数字解读

| 指标        | 数值        | 说明                         |
| --------- | --------- | -------------------------- |
| 原始样本量     | 1,778,176 | 约 178 万条记录（含缺失标签）          |
| 有标签样本     | 209,758   | Status.Vital 非缺失的样本        |
| 删除样本数     | 1,568,418 | Status.Vital 缺失的样本（88.20%） |
| VIVO（存活）  | 85,896    | 占有标签样本的 40.95%             |
| MORTO（死亡） | 123,862   | 占有标签样本的 59.05%             |

### 数据平衡性分析

从结果看，存活（40.95%）和死亡（59.05%）的比例约为 4:6，属于**轻度不平衡**（mild imbalance）。这个程度的不平衡通常不需要特殊处理（如过采样/欠采样），大多数机器学习模型和统计检验都能正常工作。

> ⚠️ **常见问题**：什么程度的不平衡需要特殊处理？
>
> 一般经验：
>
> - **1:1 \~ 4:6**：轻度不平衡，通常无需处理。
> - **2:8 \~ 1:9**：中度不平衡，建议使用 `class_weight='balanced'` 或评估指标用 F1/AUC 而非 Accuracy。
> - **< 1:9**：严重不平衡，需要过采样（SMOTE）、欠采样或异常检测思路。
>
> 本数据集 4:6 属于轻度不平衡，统计检验（t 检验、Mann-Whitney、卡方）对此都不敏感，可以放心使用。

***

## 八、小贴士与常见问题

### 💡 小贴士汇总

1. **`map`** **优于** **`==`**：当原始标签列可能有缺失值时，用 `map` 创建目标变量，缺失值会保持 NaN，后续 dropna 才能正确过滤。
2. **`.copy()`** **防警告**：对派生的 DataFrame 做修改前，加 `.copy()` 创建独立副本，避免 SettingWithCopyWarning。
3. **布尔 Series 的 mean = 比例**：`(df['col'] == value).mean()` 直接算出该值的占比，比手动除法更优雅。
4. **千位分隔符** **`:,`**：`f"{num:,}"` 让大数字更易读，如 `1,778,176` 而不是 `1778176`。
5. **提前创建输出目录**：脚本开头用 `os.makedirs(exist_ok=True)` 创建 `img`、`results` 目录，避免后续保存文件失败。

### ⚠️ 常见问题

**Q1：为什么我用了** **`==`** **创建 target，dropna 后样本数比预期多？**

A：因为 `==` 把缺失值误判成了 0（MORTO），这些"假 MORTO"样本没有被 dropna 识别出来，混进了数据集。你应该改用 `map`，让缺失值保持 NaN，dropna 才能正确删除它们。

**Q2：`map`** **和** **`apply`** **有什么区别？**

A：`map` 是 Series 的方法，专门用于**逐元素映射**（接受字典或函数）；`apply` 既能用于 Series 也能用于 DataFrame，更通用但更慢。对于简单的字典映射，`map` 更高效、更语义化。

**Q3：为什么** **`NaN == NaN`** **是 False？**

A：这是 IEEE 754 浮点数标准的规定——NaN（Not a Number）表示"未知/不确定"的值，两个"未知"不能认为相等。要判断一个值是否是 NaN，应该用 `pd.isna(x)` 或 `x != x`（这个奇怪的表达式对 NaN 返回 True）。

**Q4：`SettingWithCopyWarning`** **一定要处理吗？**

A：建议处理。虽然它只是警告（代码仍会运行），但提示你的代码可能有意外的副作用——修改没生效，或意外修改了原数据。加 `.copy()` 是最简单的解决方案。

**Q5：为什么** **`low_memory=False`** **能消除类型推断警告？**

A：pandas 默认分块读取 CSV，每块推断类型后合并，类型不一致时会警告。`low_memory=False` 一次性读取整个文件再推断类型，类型推断更准确，不会警告。

***

## 九、本模块小结

### 核心知识点回顾

1. **库导入**：
   - `from scipy import stats`：通过 `stats.` 前缀访问所有统计函数（如 `stats.normaltest`、`stats.ttest_ind`）。
   - `from scipy.stats import mannwhitneyu, chi2_contingency`：直接导入两个主力检验函数，调用时无需前缀，代码更简洁。
   - 两种方式配合使用：常用的直接导入，不常用的通过前缀访问。
2. **路径与目录**：`os.path.join()` 跨平台拼接路径，`os.makedirs(exist_ok=True)` 安全创建目录。
3. **CSV 读取**：`low_memory=False` 一次性读取避免类型推断警告；`encoding='latin-1'` 处理特殊字符。
4. **目标变量创建（本模块核心）**：
   - 用 `df['Status.Vital'].map({'VIVO': 1, 'MORTO': 0})` 而非 `(df['Status.Vital'] == 'VIVO').astype(int)`。
   - **`map`** **让缺失值保持 NaN，`==`** **会把缺失值误判为 0**。
   - 当原始标签列有缺失值时，必须用 `map`。
5. **缺失值过滤**：`dropna(subset=['target'])` 精确删除 target 缺失的行；`.copy()` 创建独立副本，避免 SettingWithCopyWarning。
6. **比例计算技巧**：`(df['target'] == 1).mean()` 利用"布尔 Series 的均值 = True 的比例"这一特性，优雅地算出存活比例。

### 数据概况

- 原始 1,778,176 条记录 → 过滤后 209,758 条有标签样本
- 存活 85,896 (40.95%)，死亡 123,862 (59.05%)
- 4:6 轻度不平衡，无需特殊处理

### 下一模块预告

本模块完成了数据加载和目标变量创建，得到了一个干净的、有明确标签的数据集（209,758 条记录）。下一模块（**模块 1：特征分类与正态性判断**）将：

- 把 38 个原始特征分类为数值型和分类型
- 编写 `check_normality_practical` 函数判断每个数值特征是否服从正态分布
- 根据正态性结果决定后续用 t 检验还是 Mann-Whitney U 检验

***

