# 模块 0：环境准备与数据加载

> 本模块是整个 EDA（探索性数据分析）教程的起点。在开始任何数据分析工作之前，我们必须先把"地基"打好：导入正确的库、配置好路径、把数据干净地加载进内存，并处理好目标变量的缺失问题。 

***

## 学习目标

学完本模块后，你将能够：

1. **理解每个 import 语句的作用**：知道 pandas、numpy、matplotlib、seaborn、scipy 等核心库分别在数据分析中扮演什么角色。
2. **掌握路径配置的最佳实践**：理解为什么用 `os.path.join()` 而不是字符串拼接，以及 `os.makedirs(..., exist_ok=True)` 的含义。
3. **正确读取 CSV 文件**：理解 `pd.read_csv()` 中 `low_memory=False` 和 `encoding='latin-1'` 这两个关键参数的作用。
4. **处理目标变量的缺失值**：学会用 `dropna(subset=...)` 删除特定列缺失的样本，并理解 `reset_index(drop=True)` 的意义。
5. **掌握布尔转整型的技巧**：理解 `(df['col'] == 'value').astype(int)` 如何把字符串标签转换成 0/1 数值。
6. **理解为什么 Status.Vital 缺失率高达 88%**：从数据采集机制层面理解缺失原因，并明白为什么不能简单地把缺失当作死亡。
7. **评估数据集的内存占用**：学会用 `df.memory_usage(deep=True)` 监控内存。

***

## 一、导入必要的库

数据分析的第一步永远是导入所需的库。下面这段代码完成了所有库的导入：

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime
from scipy import stats
```

### 1.1 逐行解释每个 import

#### `import pandas as pd`

**pandas** 是 Python 中最核心的数据分析库，它提供了 `DataFrame`（数据框）这种二维表格结构，类似于 Excel 表格或 SQL 表。几乎所有数据加载、清洗、变换操作都依赖它。

- `as pd` 是给库起一个简短的别名，这样我们写 `pd.read_csv()` 而不是 `pandas.read_csv()`，更简洁。
- 常用功能：`read_csv()` 读数据、`DataFrame` 数据结构、`dropna()` 删缺失值、`value_counts()` 统计频数、`groupby()` 分组聚合等。

#### `import numpy as np`

**NumPy**（Numerical Python）是 Python 数值计算的基石，提供了高效的多维数组 `ndarray` 和大量数学函数。pandas 底层就是基于 numpy 构建的。

- `as np` 同样是别名约定，整个 Python 社区都遵守这个约定。
- 常用功能：`np.nan`（缺失值表示）、`np.mean()`、`np.std()`、数组运算等。
- 在本教程中，numpy 主要用于数值计算和数组操作。

#### `import matplotlib.pyplot as plt`

**matplotlib** 是 Python 最基础的可视化库，`pyplot` 是它的子模块，提供了类似 MATLAB 的绘图接口。

- `as plt` 是社区约定的别名。
- 常用功能：`plt.figure()`、`plt.subplot()`、`plt.savefig()`、`plt.close()` 等。
- 在本教程中，所有图表（条形图、饼图、直方图等）都用 matplotlib 绘制。

#### `import seaborn as sns`

**seaborn** 是基于 matplotlib 的高级可视化库，提供了更美观的默认样式和更简洁的 API，特别擅长统计图表。

- `as sns` 是社区约定别名（seaborn 名字来源于美剧《西部世界》中的角色）。
- 常用功能：`sns.heatmap()`（热力图）、`sns.boxplot()`（箱线图）、`sns.distplot()`（分布图）等。
- 虽然本模块代码没有直接调用 seaborn，但后续模块会大量使用，所以提前导入。

#### `import os`

**os** 是 Python 标准库，提供了与操作系统交互的功能，比如路径操作、目录创建、环境变量等。

- 不需要安装，Python 自带。
- 常用功能：`os.path.join()`（拼接路径）、`os.makedirs()`（创建目录）、`os.path.exists()`（判断路径是否存在）。

#### `import warnings`

**warnings** 是 Python 标准库，用于控制警告信息的输出。

- 不需要安装，Python 自带。
- 常用功能：`warnings.filterwarnings()`（过滤警告）。

#### `from datetime import datetime`

**datetime** 是 Python 标准库中处理日期时间的模块。`from datetime import datetime` 这种写法表示从 `datetime` 模块中只导入 `datetime` 这个类。

- 常用功能：`datetime.now()`（获取当前时间）、时间格式化、时间差计算等。
- 在本教程中主要用于记录脚本运行时间戳。

#### `from scipy import stats`

**SciPy**（Scientific Python）是建立在 numpy 之上的科学计算库，`stats` 子模块提供了大量统计函数。

- `from scipy import stats` 表示只导入 `stats` 子模块。
- 常用功能：`stats.shapiro()`（正态性检验）、`stats.ttest_ind()`（t 检验）、`stats.mannwhitneyu()`（Mann-Whitney U 检验）、`stats.zscore()`（Z 分数）等。
- 在后续的统计分析和离群值检测模块中会大量使用。

> 💡 **小贴士**：导入库时，`import xxx as yyy` 中的别名是社区约定俗成的，建议遵守这些约定（如 `pd`、`np`、`plt`、`sns`），这样别人看你的代码会更容易理解。

***

## 二、关闭警告信息

```python
warnings.filterwarnings('ignore')
```

### 2.1 这行代码的作用

`warnings.filterwarnings('ignore')` 的作用是**关闭所有警告信息的输出**。在运行数据分析脚本时，pandas、numpy、matplotlib 等库可能会输出大量警告，例如：

- `FutureWarning`：某个函数在未来的版本中会改变行为。
- `DeprecationWarning`：某个函数已经被弃用。
- `RuntimeWarning`：运行时的一些非致命问题（比如除以零、无效值等）。

### 2.2 为什么要关闭警告

在**教学脚本**和**探索性分析**阶段，关闭警告有几个好处：

1. **保持输出干净**：让控制台输出聚焦于真正的结果，不被大量警告淹没。
2. **避免初学者困惑**：初学者看到一堆英文警告容易误以为代码出错了。
3. **教学聚焦**：教学时希望学生关注核心逻辑，而不是被次要信息干扰。

> ⚠️ **常见问题**：在生产环境或正式项目中，**不建议**无脑关闭所有警告。警告往往提示潜在问题（比如某个函数即将废弃、数据类型不匹配等），应该认真对待。更好的做法是针对性地过滤特定警告，例如：
>
> ```python
> warnings.filterwarnings('ignore', category=FutureWarning)
> ```

***

## 三、路径配置

```python
BASE_DIR = ""
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
```

### 3.1 路径配置的整体思路

这段代码定义了四个关键路径变量：

| 变量名           | 含义       | 指向                         |
| ------------- | -------- | -------------------------- |
| `BASE_DIR`    | 项目根目录    | ` `                        |
| `DATA_PATH`   | 数据文件路径   | `data/cancer_data_eng.csv` |
| `IMG_DIR`     | 图片输出目录   | `img/`                     |
| `RESULTS_DIR` | 文本结果输出目录 | `results/`                 |

这种"先定义根目录，再拼接子路径"的方式有两个好处：

1. **可维护性**：如果项目搬家，只需要改 `BASE_DIR` 一处，其他路径自动更新。
2. **可读性**：路径的语义结构清晰，一眼就能看出每个文件属于哪个目录。

### 3.2 `os.path.join()` 的作用

`os.path.join()` 是 Python 中拼接路径的**标准做法**，它会把多个字符串用正确的路径分隔符连接起来。

```python
os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
 
```

### 3.3 为什么用 `os.path.join()` 而不是字符串拼接

你可能会想，为什么不直接写成 `BASE_DIR + "/data/cancer_data_eng.csv"`？原因有三：

1. **跨平台兼容**：不同操作系统的路径分隔符不同（Linux/Mac 用 `/`，Windows 用 `\`）。`os.path.join()` 会自动选择当前系统正确的分隔符，而字符串拼接写死 `/` 在 Windows 上可能出问题。
2. **避免分隔符错误**：如果 `BASE_DIR` 末尾已经有 `/`，字符串拼接会变成 `//`（虽然大多数系统能容忍，但不规范）；`os.path.join()` 会自动处理这种情况。
3. **可读性和规范性**：这是 Python 社区的最佳实践，所有教程和项目都推荐这样做。

> 💡 **小贴士**：在 Python 3.4+ 中，还可以使用 `pathlib.Path` 来处理路径，它更加面向对象，例如：
>
> ```python
> from pathlib import Path
> BASE_DIR = Path("/home/wjj/Documents/ml_template")
> DATA_PATH = BASE_DIR / "data" / "cancer_data_eng.csv"
> ```
>
> 不过本教程为了兼容性和简洁性，仍然使用传统的 `os.path.join()`。

***

## 四、创建输出目录

```python
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
```

### 4.1 `os.makedirs()` 的作用

`os.makedirs()` 用于**递归创建多级目录**。比如 `IMG_DIR` 是 `ml_template/img`，如果 `ml_template` 目录都不存在，`os.makedirs()` 会一次性把所有缺失的父目录都创建出来。

与之相对的是 `os.mkdir()`，它只能创建一级目录，如果父目录不存在会报错。

### 4.2 `exist_ok=True` 参数的含义

`exist_ok` 是 `os.makedirs()` 的一个关键字参数，含义是"如果目录已存在，是否允许（OK）"。

- `exist_ok=True`（本教程使用）：如果目录已经存在，**不报错**，直接跳过。
- `exist_ok=False`（默认值）：如果目录已经存在，会抛出 `FileExistsError` 异常。

> 💡 **小贴士**：在数据分析脚本中，推荐始终使用 `exist_ok=True`，因为脚本可能被多次运行，目录很可能已经存在，不希望每次都因为目录已存在而报错中断。

***

## 五、打印教程标题

```python
print("=" * 70)
print("案例教程 1: 数据探索分析 (EDA)")
print("=" * 70)
```

这段代码用 `=` 重复 70 次作为分隔线，打印出教程标题。这是一种常见的控制台输出美化方式，让输出结果更有层次感。

- `"=" * 70` 是 Python 字符串乘法，表示把 `=` 重复 70 次。
- 这种分隔线在长脚本中能帮助区分不同模块的输出。

***

## 六、加载数据

```python
print("\n[0] 加载数据...")
df = pd.read_csv(DATA_PATH, low_memory=False, encoding='latin-1')
print(f"    原始数据集形状: {df.shape}")
```

### 6.1 `pd.read_csv()` 函数详解

`pd.read_csv()` 是 pandas 读取 CSV 文件的核心函数。本教程使用了两个关键参数：`low_memory=False` 和 `encoding='latin-1'`。

#### 参数 1：`low_memory=False`

**默认情况**：pandas 读取 CSV 时默认 `low_memory=True`，它会分块读取文件，逐块推断每列的数据类型。这种方式内存友好，但有一个问题——如果某一列在前几行是整数，到后面突然变成字符串，pandas 会发出警告：

```
DtypeWarning: Columns (X, Y) have mixed types.
```

**设置** **`low_memory=False`**：pandas 会**一次性读取整个文件**，然后统一推断每列的数据类型。这样不会出现类型推断不一致的问题，但代价是内存占用更高。

**为什么本教程要这样设置**：

- 本数据集有 38 个字段、约 178 万条记录，字段类型复杂，部分列确实存在混合类型。
- 为了避免 `DtypeWarning` 干扰教学，统一关闭分块读取。
- 数据集虽然大，但服务器内存足够，可以承受一次性加载。

> ⚠️ **常见问题**：如果你的数据集非常大（比如几个 GB），设置 `low_memory=False` 可能导致内存溢出（OOM）。这时应该改用分块读取：
>
> ```python
> chunk_iter = pd.read_csv(DATA_PATH, chunksize=100000)
> for chunk in chunk_iter:
>     # 逐块处理
>     process(chunk)
> ```

#### 参数 2：`encoding='latin-1'`

`encoding` 参数指定文件的字符编码。常见的编码有：

- `utf-8`：最通用的编码，支持多语言。
- `latin-1`（也叫 `iso-8859-1`）：西欧语言编码，能表示 0–255 的所有字节。
- `gbk`：简体中文编码。

**为什么本教程用** **`latin-1`**：

- 本数据集来自巴西，包含葡萄牙语字符（如 `ç`、`ã`、`õ`、`é` 等），这些字符在 `latin-1` 编码下能正确表示。
- 如果用默认的 `utf-8` 读取，遇到无法解码的字节会抛出 `UnicodeDecodeError`。
- `latin-1` 的一个特性是：**任何字节序列都能被它解码**（因为它定义了 0–255 的所有字节），所以不会报错。代价是某些字符可能显示不正确，但至少能读进来。

> 💡 **小贴士**：遇到 `UnicodeDecodeError` 时，可以尝试以下几种编码：
>
> 1. 先试 `utf-8`（最通用）
> 2. 再试 `latin-1`（万能兜底，不会报错）
> 3. 如果是中文数据，试 `gbk` 或 `gb2312`
> 4. 用 `chardet` 库自动检测编码：`import chardet; chardet.detect(open(file, 'rb').read(10000))`

### 6.2 `df.shape` 的含义

`df.shape` 返回一个元组 `(行数, 列数)`，表示数据框的形状。

```python
print(f"    原始数据集形状: {df.shape}")
# 输出类似：原始数据集形状: (1778176, 38)
```

- `df.shape[0]` 是行数（样本数）
- `df.shape[1]` 是列数（特征数）

***

## 七、处理目标变量的缺失值

这是本模块**最关键**的一步，也是理解整个数据集的关键。

### 7.1 背景：Status.Vital 的高缺失率

```python
# 处理 Status.Vital (目标变量) 的缺失值
n_before = len(df)
df = df.dropna(subset=['Status.Vital']).reset_index(drop=True)
n_after = len(df)
print(f"    删除 Status.Vital 缺失样本: {n_before:,} → {n_after:,} (删除 {n_before - n_after:,} 条, {(n_before - n_after) / n_before * 100:.2f}%)")
```

`Status.Vital` 是本数据集的**目标变量**（label），表示患者的生死状态：

- `VIVO`（葡萄牙语"活着"）→ 存活
- `MORTO`（葡萄牙语"死亡"）→ 死亡
- 缺失 → 未知

**问题**：这个变量的缺失率高达 **88.20%**（1,568,418 条记录没有明确的生死标签）！

### 7.2 为什么缺失率这么高？4 个原因

在癌症登记数据库中，`Status.Vital` 高缺失率是常见现象，主要有 4 个原因：

1. **患者刚入库，尚未到随访时间**：癌症登记系统会持续随访患者，但新入库的患者还没到随访节点，所以 `Status.Vital` 暂时为空。这部分患者其实可能还活着，只是还没确认。
2. **患者失访（Lost to follow-up）**：有些患者搬走了、换电话了、或者不再回医院复诊，登记中心联系不上他们，无法确认生死状态。这类缺失反映了真实世界的数据采集困难。
3. **数据采集不完整**：登记中心的工作人员可能漏填、错填，或者数据录入系统有 bug，导致部分记录的 `Status.Vital` 字段为空。
4. **该变量本来只对部分患者记录**：在某些登记流程中，`Status.Vital` 只在特定时间点（比如年度复核）才更新，平时不强制填写，所以很多历史记录没有这个字段。

### 7.3 为什么不能把缺失当作死亡？

这是一个**非常容易犯的错误**！初学者可能会想："既然缺失了，那大概率是死了吧？"——**这种想法是危险的**，原因如下：

1. **缺失并非随机（非 MCAR）**：从上面的 4 个原因可以看出，缺失往往和"刚入库"、"失访"等状态相关，并不是随机发生的。如果把缺失都归为死亡，会**严重高估死亡率**。
2. **会严重低估真实存活率**：本数据集中 88% 的样本缺失，如果把它们都算作死亡，那存活率会被压到极低（不到 5%），这显然不符合癌症患者的真实生存情况。
3. **会引入系统性偏差**：缺失样本和有标签样本在特征分布上可能不同（比如缺失样本可能更年轻、更早期），把它们当作死亡会让模型学到错误的模式。
4. **违背统计原则**：在统计学中，"缺失"和"死亡"是两个完全不同的概念。把未知当作已知，是一种"数据造假"。

> ⚠️ **重要概念**：在统计学中，缺失机制分为三类：
>
> - **MCAR（Missing Completely At Random）**：完全随机缺失，缺失和任何变量都无关。
> - **MAR（Missing At Random）**：随机缺失，缺失只和已观测的变量有关。
> - **MNAR（Missing Not At Random）**：非随机缺失，缺失和未观测的值本身有关。
>
> 本数据集的 `Status.Vital` 缺失更接近 **MNAR**（因为缺失和患者是否失访有关，而失访本身又和生死状态相关），所以不能简单处理。

### 7.4 处理方式：删除缺失样本

本教程采取的处理方式是：**删除** **`Status.Vital`** **缺失的样本，仅保留有明确生死标签的记录**。

```python
df = df.dropna(subset=['Status.Vital']).reset_index(drop=True)
```

处理后的数据：

- 原始：1,778,176 条
- 删除：1,568,418 条（88.20%）
- 保留：209,758 条

### 7.5 `dropna(subset=['Status.Vital'])` 详解

`dropna()` 是 pandas 删除缺失值的方法，`subset` 参数指定**只检查哪些列**。

- `subset=['Status.Vital']`：只检查 `Status.Vital` 这一列，如果这一列是 `NaN`，就删除整行。
- 如果不指定 `subset`，`dropna()` 会删除**任何一列**有缺失的行，这在有很多列的数据集上会删掉几乎所有数据。

> 💡 **小贴士**：`subset` 参数非常有用。实际数据中，几乎每行都或多或少有缺失值，如果无脑 `dropna()` 会把数据删光。用 `subset` 可以精确控制只针对关键列（比如目标变量）删除。

### 7.6 `reset_index(drop=True)` 详解

删除行之后，数据框的索引（index）会保留原来的值，出现"空洞"。比如删除了第 3、5、7 行后，索引变成 `0, 1, 2, 4, 6, 8, ...`，不连续。

`reset_index()` 会重新设置索引，让索引从 0 开始连续递增。`drop` 参数的含义：

- `drop=True`：**丢弃原来的索引**，不把它作为新列保留。
- `drop=False`（默认）：把原来的索引作为新的一列（叫 `index`）保留下来。

本教程用 `drop=True`，因为我们不需要保留原来的索引（原来的索引只是行号，没有信息价值）。

> ⚠️ **常见问题**：为什么要 `reset_index`？因为不连续的索引在后续操作中可能引发问题：
>
> - 用 `loc[]` 按索引取值时容易出错。
> - 用 `for i in range(len(df))` 遍历时，`df.loc[i]` 可能找不到。
> - 一些函数（比如 `iloc[]` 和 `loc[]`）的行为会让人困惑。
>
> 删除行后养成 `reset_index(drop=True)` 的习惯是好做法。

***

## 八、创建目标变量（布尔转整型技巧）

```python
# 创建目标变量: VIVO=1 (alive), MORTO=0 (dead)
df['target'] = (df['Status.Vital'] == 'VIVO').astype(int)
```

### 8.1 这行代码做了什么

它创建了一个新列 `target`，把字符串标签转换成数值：

| Status.Vital | (== 'VIVO') | .astype(int) | target |
| ------------ | ----------- | ------------ | ------ |
| VIVO         | True        | 1            | 1      |
| MORTO        | False       | 0            | 0      |

### 8.2 拆解这个技巧

#### 第一步：`df['Status.Vital'] == 'VIVO'`

这是一个**比较运算**，会对 `Status.Vital` 列的每个值判断是否等于 `'VIVO'`，返回一个**布尔 Series**（True/False 序列）。

```
Status.Vital    == 'VIVO'    →    结果
VIVO            True              True
MORTO           True              False
VIVO            True              True
MORTO           True              False
```

#### 第二步：`.astype(int)`

`astype(int)` 把布尔类型转换成整数类型。在 Python 中：

- `True` → 1
- `False` → 0

所以 `True` 变成 1，`False` 变成 0。

### 8.3 为什么不直接用 `if-else`？

初学者可能会想用 `apply` + `if-else`：

```python
# 不推荐的写法（慢且啰嗦）
df['target'] = df['Status.Vital'].apply(lambda x: 1 if x == 'VIVO' else 0)
```

相比之下，`(df['Status.Vital'] == 'VIVO').astype(int)` 有两个优势：

1. **向量化操作，速度快**：pandas 的比较运算和类型转换都是底层 C 实现的向量化操作，比 Python 层的 `apply` + `lambda` 快几个数量级。
2. **代码简洁**：一行搞定，可读性更强。

> 💡 **小贴士**：在 pandas 中，**能用向量化操作就不用** **`apply`**。向量化操作利用了底层的 C/numpy 优化，性能远超 Python 循环。常见的向量化技巧：
>
> - 比较运算：`df['col'] == value`
> - 算术运算：`df['col'] * 2`
> - 字符串方法：`df['col'].str.upper()`
> - 类型转换：`df['col'].astype(int)`

### 8.4 为什么 target 用 1 表示存活、0 表示死亡？

这是一种约定，在二分类问题中：

- **1（正类，positive）**：通常表示我们"关注"的事件。在医学场景中，"存活"是我们希望预测和促进的结果，所以设为 1。
- **0（负类，negative）**：表示另一个类别。

> ⚠️ **注意**：不同场景下正类的定义可能不同。比如在"预测癌症"任务中，"患癌"通常设为 1（因为我们关注的是找出患者）。在本教程中，因为关注的是"存活预测"，所以"存活"设为 1。务必根据具体任务确定正类定义，这会影响后续的评估指标（如 Precision、Recall）解读。

***

## 九、打印处理后数据集信息

```python
print(f"    处理后数据集形状: {df.shape}")
print(f"    内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

### 9.1 `df.memory_usage(deep=True)` 详解

`df.memory_usage()` 返回每列的内存占用（字节数），`deep` 参数的含义：

- `deep=False`（默认）：只估算内存，对于 object 类型的列（通常是字符串）只算指针大小，**严重低估**实际内存。
- `deep=True`：**深入计算** object 列中每个字符串的实际内存占用，结果更准确但计算更慢。

`.sum()` 把所有列的内存加起来，`/ 1024**2` 把字节转换成 MB（1 MB = 1024 × 1024 字节）。

### 9.2 为什么要关注内存占用

- **大数据集**：本数据集有 178 万条记录，内存占用可能达到几百 MB 甚至几个 GB。
- **性能优化**：如果内存占用过高，可以考虑：
  - 把 `object` 类型转成 `category` 类型（对于取值有限的分类变量）。
  - 把 `float64` 转成 `float32`（精度足够时）。
  - 把 `int64` 转成 `int32` 或 `int16`。
- **避免 OOM**：监控内存可以提前发现内存不足的风险。

> 💡 **小贴士**：用 `df.info(memory_usage='deep')` 可以一次性查看所有列的数据类型和总内存占用，非常方便。

***

## 十、本模块运行结果

运行本模块代码后，控制台输出大致如下：

```
======================================================================
案例教程 1: 数据探索分析 (EDA)
======================================================================

[0] 加载数据...
    原始数据集形状: (1778176, 38)
    删除 Status.Vital 缺失样本: 1,778,176 → 209,758 (删除 1,568,418 条, 88.20%)
    处理后数据集形状: (209758, 39)
    内存占用: 245.67 MB
```

### 关键数字解读

| 指标     | 数值        | 说明                       |
| ------ | --------- | ------------------------ |
| 原始样本数  | 1,778,176 | 约 178 万条记录               |
| 原始特征数  | 38        | 38 个原始字段                 |
| 删除样本数  | 1,568,418 | Status.Vital 缺失的样本       |
| 删除比例   | 88.20%    | 缺失率非常高                   |
| 处理后样本数 | 209,758   | 保留有明确标签的记录               |
| 处理后特征数 | 39        | 38 个原始字段 + 1 个新建的 target |
| 内存占用   | \~245 MB  | 处理后数据集的内存                |

> ⚠️ **常见问题**：为什么处理后特征数从 38 变成 39？
>
> 因为我们新建了 `target` 列（`df['target'] = ...`），它是从 `Status.Vital` 派生出来的。所以处理后数据集有 38 个原始字段 + 1 个 target = 39 列。注意 `Status.Vital` 本身没有被删除，它仍然保留在数据集中。

***

## 十一、本模块小结

### 核心知识点回顾

1. **库导入**：pandas（数据处理）、numpy（数值计算）、matplotlib/seaborn（可视化）、scipy（统计）、os（路径）、warnings（警告控制）。
2. **路径管理**：用 `os.path.join()` 拼接路径，跨平台兼容；用 `os.makedirs(exist_ok=True)` 创建目录，避免重复创建报错。
3. **CSV 读取**：`low_memory=False` 一次性读取避免类型推断警告；`encoding='latin-1'` 处理葡萄牙语字符。
4. **缺失值处理**：`dropna(subset=['col'])` 只针对关键列删除；`reset_index(drop=True)` 重置索引。
5. **标签编码**：`(df['col'] == 'value').astype(int)` 是向量化布尔转整型的优雅技巧。
6. **目标变量缺失的本质**：Status.Vital 88% 缺失不是数据质量问题，而是癌症登记系统的固有特征（随访机制、失访、采集流程）。不能把缺失当作死亡，否则会引入严重偏差。

### 下一模块预告

本模块完成了数据加载和目标变量处理，得到了一个干净的、有明确标签的数据集（209,758 条记录，39 列）。下一模块（**模块 1：数据概况**）将深入分析这个数据集的：

- 样本数量和特征数量
- 标签分布（存活 vs 死亡的比例）
- 数据平衡性判断
- 可视化标签分布

***

> 📌 **思考题**：
>
> 1. 如果你的服务器内存只有 2GB，而数据集有 5GB，你会怎么修改本模块的代码？
> 2. 除了删除缺失样本，还有哪些处理目标变量缺失的方法？各有什么优缺点？
> 3. 为什么 `Status.Vital` 的缺失更接近 MNAR 而不是 MCAR？请结合 4 个缺失原因分析。

