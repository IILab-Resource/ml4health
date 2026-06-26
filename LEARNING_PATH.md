# 机器学习教学分级学习路径

> 本教程体系包含 **19 个案例**（01~18 + 12b），覆盖从 EDA 到因果推断的完整机器学习链路。每个案例内部的知识点按难度分为三级：
>
> - **L1 核心概念与实践**：必须掌握的基础概念和代码实现
> - **L2 方法深入**：在 L1 基础上的方法理解、参数调优、对比分析
> - **L3 理论扩展**：数学理论基础、扩展内容、论文级分析方法

---

## 一、三级学生画像

### L1：零基础入门

| 维度 | 描述 |
|------|------|
| **Python 水平** | 学过基本语法（变量、循环、函数、条件判断），能读懂简单的 pandas/numpy 代码 |
| **数学基础** | 高中数学（均值、标准差、百分比），不需要概率论和线性代数 |
| **ML 经验** | 零经验，第一次接触机器学习流程 |
| **学习目标** | 理解 ML 项目的完整流程，掌握每个案例的核心概念，能跑通代码并解释关键结果 |
| **时间预估** | 2~3 周 |

**L1 学生应该做到：**
- 每个案例都阅读教学文档（lectures/）的"前序知识"+"实践目的"+"结果解读"
- 深入理解每个案例的 L1 核心知识点，能用自己的话解释
- 阅读 L1 对应的详细笔记（notes/），逐行理解代码
- 跑通代码，能看懂关键图表和数字
- **跳过**教学文档中的"扩展内容"章节

### L2：有一定基础

| 维度 | 描述 |
|------|------|
| **Python 水平** | 熟练使用 pandas、numpy、matplotlib，能独立写数据处理脚本 |
| **数学基础** | 大学基础统计学（假设检验、p 值、置信区间），了解线性代数基本概念 |
| **ML 经验** | 了解基本模型（LR、RF），知道 train/test split、交叉验证 |
| **学习目标** | 在 L1 基础上，深入理解方法原理，掌握参数调优，能对比不同方法 |
| **时间预估** | 3 周  |

**L2 学生应该做到：**
- L1 的全部内容
- 阅读教学文档的全部内容（包括"扩展内容"章节）
- 阅读 L2 对应的详细笔记（notes/），理解方法原理和参数含义
- 能修改参数（如换插补方法、换模型），观察结果变化
- 理解方法之间的对比和选择逻辑

### L3：完整版精通

| 维度 | 描述 |
|------|------|
| **Python 水平** | 精通科学计算栈，能阅读库源码 |
| **数学基础** | 扎实的概率统计和线性代数，理解优化理论基本概念 |
| **ML 经验** | 有过完整 ML 项目经验，了解模型原理 |
| **学习目标** | 掌握全部细节和理论基础，理解扩展内容，能做论文级分析 |
| **时间预估** | 4 周  |

**L3 学生应该做到：**
- L1 + L2 的全部内容
- 阅读所有 L3 对应的详细笔记（notes/）
- 理解教学文档中所有扩展内容的理论基础
- 能独立完成类似的 ML 项目
- 能撰写论文级的分析方法描述

---

## 二、案例知识点分级总览

> 每个案例不再按"做/不做"划分，而是按**知识点难度**划分。L1 学生也要学每个案例，但只学到 L1 深度。

| 案例 | 名称 | L1 核心概念 | L2 方法深入 | L3 理论扩展 |
|:----:|------|:-----------|:-----------|:-----------|
| 01 | EDA | 缺失值/分布/离群值概念 | 缺失机制 MCAR/MAR/MNAR、IQR vs Z-score 选择 | 多维缺失模式诊断、面向建模 EDA、时间序列分析 |
| 02 | 统计分析 | p 值/效应量/假设检验 | 多重比较校正、效应量解读规则、检验方法选择 | 非参数方法、贝叶斯因子、统计显著 vs 预测力 |
| 03 | 预处理与插补 | 缺失值插补/编码/标准化 | MICE 迭代机制、KNN 时间复杂度、插补方法对比 | 插补理论推导、分布失真效应、校准度影响 |
| 04 | 特征工程 | StandardScaler/RobustScaler | 梯度下降影响、缩放器选择策略、特征构造方法 | 组合爆炸应对、医学特征构造、Recall 下降反直觉 |
| 05 | 特征选择 | 相关性/VIF/Filter/LASSO | RF 重要性、Boruta 原理、六层选择体系 | 稳定性选择、Wrapper 方法、模型特定特征选择 |
| 06 | 降维与聚类 | PCA 基本概念、K-Means | t-SNE/UMAP 区别、维度灾难、聚类评估指标 | 降维理论推导、其他降维方法、层次聚类/DBSCAN |
| 07 | 数据泄漏 | 泄漏本质、fit/transform 分离 | Pipeline 防泄漏、三种泄漏对比 | 泄漏剂量效应、时间序列泄漏、真实案例 |
| 08 | 交叉验证 | K-Fold/StratifiedKFold | RepeatedKFold、LOOCV、偏差-方差权衡 | 嵌套 CV、评估可信度、时序 CV |
| 09 | 模型比较 | 7 种模型基本概念 | 模型参数配置、模型选择策略 | 集成学习理论、深度学习、超参数优化 |
| 10 | 不平衡数据 | 准确率悖论、SMOTE 概念 | ADASYN、SMOTEENN、SMOTE 泄漏、CV 重采样 | 高级合成方法、阈值移动、医学代价分析 |
| 11 | 校准与 DCA | 校准度概念、校准曲线 | Brier 分数分解、Platt Scaling、DCA 解读 | DCA 变体、可用性评估流程、HL 检验局限 |
| 12 | 模型解释 | SHAP 值/蜂群图/瀑布图 | Gini/Permutation/SHAP 三种重要性对比、LIME | SHAP 理论（Shapley Value 四大公理）、LIME 稳定性 |
| 12b | 高级 SHAP | 二次拟合依赖图、3D 空间 | 力量矩阵、分组饼图、趋势图、网络图 | 五维评估框架、多项式拟合统计检验、特征工程决策树 |
| 13 | SHAP 交互 | 交互效应概念、交互依赖图 | 交互矩阵、两种度量方法对比 | 交互→特征工程、Friedman's H-statistic、论文金三角 |
| 14 | SHAP 依赖分布 | 双轴依赖图概念、数据密度 | Ratio 风险评估、五步读图法 | 特征可信度映射、分位数敏感性、Ratio 统计检验 |
| 15 | SHAP 聚类 | SHAP 聚类 vs 特征聚类 | 聚类画像、相对重要性 | 其他聚类算法、SHAP 空间本质低维、全流程建议 |
| 16 | SHAP 决策路径 | 决策路径概念、三列可视化 | 路径形状分类、跨样本比较 | 批量路径热图、概率转换、临床决策支持 |
| 17 | SHAP 稳定性 | Bootstrap 概念、CV/CI 指标 | 四象限判断、排名稳定性 | 半衰期分析、Sub-sampling Bootstrap、论文用法 |
| 18 | HTE DML | Y/T/X/W 四要素、ATE vs CATE | DROrthoForest 双重稳健性、CausalForestDML 分裂准则 | 反事实预测、政策学习、敏感性分析、DML vs LR |

---

## 三、每个案例的详细知识点分级

### 案例 01：EDA

**背景**：对巴西癌症登记数据（~178 万条）进行系统性探索性数据分析。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [00_module0_data_loading.md](notes/00_module0_data_loading.md) | L1 |
| 1 | [01_module1_data_overview.md](notes/01_module1_data_overview.md) | L1 |
| 2 | [02_module2_missing_values.md](notes/02_module2_missing_values.md) | L1 |
| 3 | [03_module3_distribution.md](notes/03_module3_distribution.md) | L1 |
| 4 | [04_module4_outliers.md](notes/04_module4_outliers.md) | L1 |

#### L1 核心概念与实践

**知识点：**
- 数据的全局描述（`shape`、`describe`、`value_counts`）
- 缺失值的概念：缺失比例、缺失模式
- 三种缺失机制的概念：MCAR（完全随机缺失）、MAR（条件随机缺失）、MNAR（非随机缺失）
- 分布分析：均值 vs 中位数、偏度、直方图 + KDE、Q-Q 图
- 离群值的基本概念：IQR 方法（Q1 - 1.5×IQR, Q3 + 1.5×IQR）、Z-score 方法（|Z| > 3）
- 类别平衡性（标签分布）

**代码理解重点：**
- `df.isnull().sum()` 缺失值统计
- `df.describe()` 描述统计
- `sns.histplot()` / `sns.boxplot()` 分布可视化
- `sns.heatmap(df.isnull())` 缺失值矩阵
- IQR 和 Z-score 的 NumPy 实现

**关键概念：** 缺失值、分布形态、离群值、类别平衡、EDA 的"提问-回答"迭代思维

#### L2 方法深入

**在 L1 基础上增加：**
- 缺失机制的深入判断：如何区分 MCAR/MAR/MNAR
- 缺失值层次聚类热图（`missingno` 库）
- IQR 和 Z-score 的适用场景对比：正态分布用 Z-score，偏态分布用 IQR
- 离群值 vs 错误值的区分：结合领域知识判断
- 目标变量缺失 88% 的选择偏差（survivorship bias）讨论

#### L3 理论扩展

**扩展内容（教学文档第三章）：**
- 缺失值矩阵的层次聚类
- 缺失指示变量与目标变量的关联分析
- 面向建模的 EDA：单变量关联筛选、VIF 初步诊断、交互项建议
- 时间序列分析视角：年份趋势、存活率年度变化
- 数据质量报告自动化：`ydata-profiling`、`Great Expectations`
- 子群体分析：整体 EDA 可能掩盖子群体特殊模式

---

### 案例 02：统计分析

**背景**：用统计检验方法筛选与目标变量相关的特征，建立统计学与机器学习的桥梁。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [10_module0_data_loading.md](notes/10_module0_data_loading.md) | L1 |
| 1 | [11_module1_feature_classification.md](notes/11_module1_feature_classification.md) | L1 |
| 2 | [12_module2_numerical_tests.md](notes/12_module2_numerical_tests.md) | L1 |
| 3 | [13_module3_chi_square.md](notes/13_module3_chi_square.md) | L1 |
| 4 | [14_module4_summary_visualization.md](notes/14_module4_summary_visualization.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 假设检验框架：原假设 H₀、备择假设 H₁、p 值、显著性水平 α
- 第 I 类错误（假阳性）与第 II 类错误（假阴性）
- 正态性判断：偏度检验 + D'Agostino-Pearson 检验
- T 检验（正态分布）vs Mann-Whitney U 检验（非正态分布）
- 卡方检验（分类特征 × 分类目标）
- 效应量的概念：Cohen's d、Rank-biserial r、Cramér's V
- **核心论点：** 统计显著 ≠ 预测力强

**代码理解重点：**
- `scipy.stats.normaltest()` 正态性检验
- `scipy.stats.ttest_ind()` T 检验
- `scipy.stats.mannwhitneyu()` Mann-Whitney U 检验
- `scipy.stats.chi2_contingency()` 卡方检验
- 效应量的手动计算
- p 值的 `-log10` 转换用于可视化

**关键概念：** p 值、显著性水平、效应量、T 检验、Mann-Whitney U、卡方检验

#### L2 方法深入

**在 L1 基础上增加：**
- 多重比较校正：Bonferroni 校正、Holm 校正、Benjamini-Hochberg (FDR)
- 效应量的通用解读规则（小/中/大的阈值）
- 方法选择流程图：正态→T 检验，非正态→Mann-Whitney U，分类→卡方检验
- 结果可视化：p 值对比图、效应量对比图、p 值 vs 效应量散点图
- 大样本量的"放大镜效应"：样本量越大，可检测到的效应量越小

#### L3 理论扩展

**扩展内容（教学文档第三章）：**
- 非参数方法：置换检验、Bootstrap 置信区间、Wilcoxon 符号秩检验
- 贝叶斯因子：量化支持 H₀ 和 H₁ 的证据强度
- 分类特征稀疏性处理：Fisher 精确检验、模拟 p 值
- 统计显著 vs 临床显著：即使效应量小，也可能有临床意义
- 交互效应检验：双因素 ANOVA、Logistic 交互项、Mantel-Haenszel 分层检验

---

### 案例 03：预处理与插补

**背景**：对比四种缺失值插补方法（Complete Case、Mean、KNN、MICE），评估其对模型性能的影响。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [20_module0_data_loading.md](notes/20_module0_data_loading.md) | L1 |
| 1 | [21_module1_split_encoding.md](notes/21_module1_split_encoding.md) | L1 |
| 2 | [22_module2_imputation_methods.md](notes/22_module2_imputation_methods.md) | L1 |
| 3 | [23_module3_model_evaluation.md](notes/23_module3_model_evaluation.md) | L2 |
| 4 | [24_module4_visualization.md](notes/24_module4_visualization.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 四种插补方法的基本概念：
  - Complete Case（完整案例）：删除所有含缺失值的行
  - Mean（均值插补）：用该特征的均值填充
  - KNN（K 近邻插补）：用 K 个最近邻样本的均值填充
  - MICE（多重插补）：迭代预测缺失值
- 类别特征编码：LabelEncoder（有序）vs One-Hot Encoding（无序）
- 数值特征标准化：StandardScaler（均值 0，方差 1）
- 数据划分（train_test_split）的重要性
- 插补要在划分之后做（防止数据泄漏）

**代码理解重点：**
- `SimpleImputer(strategy='mean')` 均值插补
- `KNNImputer(n_neighbors=5)` KNN 插补
- `IterativeImputer()` MICE 插补
- `LabelEncoder().fit_transform()` 标签编码
- `StandardScaler().fit_transform()` 标准化
- `train_test_split(test_size=0.2, random_state=42)`

**关键概念：** 缺失值插补、编码、标准化、数据划分、fit/transform 分离

#### L2 方法深入

**在 L1 基础上增加：**
- KNN 插补的时间复杂度：O(n²d)，通过 KD-Tree 加速
- MICE 的迭代机制：每个特征轮流作为目标，用其他特征预测
- 四种插补方法的模型性能对比（AUC、Recall、Brier）
- 均值插补的缺陷：低估方差、破坏协方差结构
- 插补方法的选择策略：缺失率低→Mean，缺失率高→MICE，特征间相关强→KNN
- 训练集大小对插补效果的影响

#### L3 理论扩展

**扩展内容：**
- 插补的理论推导：均值插补为什么低估方差
- 分布失真效应：插补后分布与原始分布的差异
- 校准度影响：不同插补方法对预测概率校准度的影响
- 其他插补方法：missForest、热卡插补、回归插补、指示变量法
- 先聚类再插补的策略

---

### 案例 04：特征工程

**背景**：对比四种标准化方法，构造新特征，评估标准化前后模型性能的变化。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [30_module0_data_loading.md](notes/30_module0_data_loading.md) | L1 |
| 1 | [31_module1_scaling.md](notes/31_module1_scaling.md) | L1 |
| 2 | [32_module2_feature_construction.md](notes/32_module2_feature_construction.md) | L2 |
| 3 | [33_module3_comparison.md](notes/33_module3_comparison.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 四种标准化方法的基本概念：
  - Raw（不标准化）：原始数据
  - StandardScaler：`(x - mean) / std`，均值 0，方差 1
  - RobustScaler：`(x - median) / IQR`，对离群值鲁棒
  - MaxAbsScaler：`x / max(|x|)`，保留稀疏性
- 为什么需要标准化：不同特征量纲不同，梯度下降受影响
- 特征构造的基本方法：分箱（Age_Group）、平方项（Age_Sq）、交互项

**代码理解重点：**
- `StandardScaler().fit_transform(X_train)` 标准化
- `pd.cut(df['Age'], bins=[0, 18, 45, 65, 100])` 分箱
- `df['Age'] ** 2` 平方项构造
- 交互特征的构造：`df['A'] * df['B']`

**关键概念：** 标准化、量纲、分箱、特征构造、交互特征

#### L2 方法深入

**在 L1 基础上增加：**
- 标准化对梯度下降的影响：等高线形态、收敛路径、迭代次数
- 缩放器选择策略：正态→StandardScaler，离群值多→RobustScaler，稀疏→MaxAbsScaler
- 特征构造的黄金法则：可解释性、不泄漏、避免冗余、避免爆炸
- 标准化前后模型性能对比（AUC、Recall、Brier、收敛速度）
- 特征构造中的常见陷阱：Age 和 Age_Sq 高度共线

#### L3 理论扩展

**扩展内容：**
- 组合爆炸问题：C(n,2) 两两交互 + 二次多项式，L1 正则化应对
- 医学特征构造：Charlson 合并症、年龄调整、风险评分、癌症亚型
- Recall 下降的反直觉现象：决策边界复杂度增加导致线性模型 Recall 下降
- 其他特征构造方法：多项式特征、比例特征、聚合特征、目标编码

---

### 案例 05：特征选择

**背景**：从六个不同角度对特征进行层层筛选，构建"最小充分特征集"。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [50_module0_data_loading.md](notes/50_module0_data_loading.md) | L1 |
| 1 | [51_module1_correlation_vif.md](notes/51_module1_correlation_vif.md) | L1 |
| 2 | [52_module2_filter_lasso.md](notes/52_module2_filter_lasso.md) | L2 |
| 3 | [53_module3_rf_boruta.md](notes/53_module3_rf_boruta.md) | L2 |
| 4 | [54_module4_validation_summary.md](notes/54_module4_validation_summary.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 相关性分析：Pearson r，|r| > 0.8 视为高度共线，删除冗余对
- 方差膨胀因子 VIF：VIF = 1/(1-R²)，VIF > 10 视为严重多重共线性
- VIF vs 相关性分析：VIF 是"一个 vs 所有"，比"一对一"更全面
- Filter 方法：ANOVA F 检验（线性区分能力）、互信息 MI（非线性关联）
- LASSO：L1 正则化，将不重要特征的系数压缩至 0
- 特征选择的核心目标：降维不降质

**代码理解重点：**
- `df.corr()` 相关性矩阵
- `variance_inflation_factor()` VIF 计算
- `SelectKBest(f_classif, k=8)` ANOVA Filter
- `LogisticRegression(penalty='l1', solver='saga', C=1/alpha)` LASSO
- 正则化路径图的解读：alpha 越大，被压缩到 0 的系数越多

**关键概念：** 共线性、多重共线性、VIF、LASSO、L1 正则化、降维不降质

#### L2 方法深入

**在 L1 基础上增加：**
- RF 特征重要性：累积重要性阈值（如 90%）选择 Top-K 特征
- Boruta 原理：阴影特征 + 随机森林 + 统计检验，Confirmed/Tentative/Rejected 三分类
- 六层特征选择流水线的完整理解：相关性→VIF→Filter→LASSO→RF→Boruta
- 为什么不同方法给出的"最佳特征集"不同：视角不同
- 特征选择后的模型验证：不同特征集在 LR 上的 AUC/Recall/Brier 对比
- LASSO 路径图的教学价值：不仅选特征，还展示重要性排序

#### L3 理论扩展

**扩展内容：**
- 稳定性选择（Stability Selection）：Bootstrap + 多次 LASSO，统计选中频率
- Wrapper 方法：RFE（递归特征消除）、前向选择、后向消除
- 模型特定特征选择：树模型用 SHAP/Permutation，线性模型用 LASSO/ElasticNet
- 特征选择后性能不降的原因：信息冗余 + 线性模型局限
- 更多 Filter 方法：ReliefF、Fisher Score、信息增益

---

### 案例 06：降维与聚类

**背景**：理解维度灾难，掌握 PCA/t-SNE/UMAP 三种降维方法，将降维用于 ML 加速。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [60_module0_data_loading.md](notes/60_module0_data_loading.md) | L1 |
| 1 | [61_module1_curse_of_dimensionality.md](notes/61_module1_curse_of_dimensionality.md) | L2 |
| 2 | [62_module2_dim_reduction_visualization.md](notes/62_module2_dim_reduction_visualization.md) | L2 |
| 3 | [63_module3_clustering.md](notes/63_module3_clustering.md) | L2 |
| 4 | [64_module4_pca_for_ml.md](notes/64_module4_pca_for_ml.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 降维的目的：可视化、去噪、加速训练
- PCA 的基本直觉：找方差最大的方向（主成分），投影到低维空间
- 方差解释率（explained_variance_ratio_）：每个主成分保留了多少信息
- 累积方差解释率：前 K 个主成分一共保留了多少信息
- K-Means 聚类的基本概念：K 个中心点，迭代分配和更新
- 聚类评估指标：Silhouette Score（轮廓系数）

**代码理解重点：**
- `PCA(n_components=2).fit_transform(X)` PCA 降维
- `pca.explained_variance_ratio_` 方差解释率
- `KMeans(n_clusters=3, random_state=42).fit_predict(X)` K-Means
- `silhouette_score(X, labels)` 轮廓系数

**关键概念：** PCA、方差解释率、主成分、K-Means、轮廓系数

#### L2 方法深入

**在 L1 基础上增加：**
- 维度灾难的演示：高维空间中距离均匀化，KNN 失效
- PCA 的数学直觉：数据中心化 → 协方差矩阵 → 特征值分解 → 主成分方向
- t-SNE 原理：高维高斯分布 → 低维 t 分布，KL 散度最小化，Perplexity 参数
- UMAP 原理：拓扑学方法，保留局部+全局结构，n_neighbors 参数
- t-SNE vs UMAP：t-SNE 只看局部，UMAP 保留全局；t-SNE 随机性大，UMAP 更稳定
- t-SNE 的常见误解：距离和密度不可靠，不能用于性能评价
- PCA vs 特征选择：PCA 是信息压缩，特征选择是子集筛选

#### L3 理论扩展

**扩展内容：**
- PCA 的数学推导：特征值分解、载荷向量、基变换+投影
- PCA 用于 ML 加速：信息压缩 vs 去噪，PCA 有帮助/没帮助的场景
- 其他降维方法：因子分析（FA）、线性判别分析（LDA）、Isomap、LLE、Autoencoder
- 其他聚类算法：层次聚类、DBSCAN、GMM、Spectral Clustering
- 距离度量：欧几里得、曼哈顿、余弦相似度，高维空间中的距离均匀化

---

### 案例 07：数据泄漏

**背景**：演示 ML 中最容易被忽视的致命错误——数据泄漏，及其防范方法。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [70_module0_data_loading.md](notes/70_module0_data_loading.md) | L1 |
| 1 | [71_module1_standardization_leakage.md](notes/71_module1_standardization_leakage.md) | L1 |
| 2 | [72_module2_feature_selection_leakage.md](notes/72_module2_feature_selection_leakage.md) | L2 |
| 3 | [73_module3_comprehensive_pipeline.md](notes/73_module3_comprehensive_pipeline.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 数据泄漏的本质：测试集信息"污染"了训练集，导致性能虚高
- 标准化泄漏：在全数据上 fit StandardScaler，再划分 = 错误
- 正确做法：先 train_test_split，再在训练集上 fit，然后 transform 测试集
- `fit_transform(X_train)` vs `transform(X_test)` 的区别
- Pipeline 的概念：`Pipeline([('scaler', StandardScaler()), ('model', LR())])`

**代码理解重点：**
- 错误写法：`scaler.fit(X).transform(X)` → `train_test_split()`
- 正确写法：`train_test_split()` → `scaler.fit(X_train).transform(X_train)` + `scaler.transform(X_test)`
- 泄漏版本 vs 正确版本的 AUC 差异对比

**关键概念：** 数据泄漏、fit/transform 分离、Pipeline、乐观偏倚

#### L2 方法深入

**在 L1 基础上增加：**
- 特征选择泄漏：在全数据上做特征选择，再划分 = 错误
- SMOTE 泄漏：先 SMOTE 再划分 = 错误（合成样本污染了测试集）
- Pipeline 防泄漏：`Pipeline([('scaler', StandardScaler()), ('selector', SelectKBest()), ('model', LR())])`
- 泄漏检查清单：标准化-fit 仅训练集、缺失值插补-fit 仅训练集、特征选择-仅训练集、SMOTE-仅训练集
- 核心原则：锁住测试集，任何涉及数据分布的操作都只在训练集上做

#### L3 理论扩展

**扩展内容：**
- 泄漏的剂量效应：标准化泄漏（小）、缺失值填充泄漏（小）、特征排名泄漏（中高）、目标编码泄漏（高）、SMOTE 泄漏（高）
- 时间序列泄漏：测试集时间必须晚于训练集
- 嵌套 CV 防泄漏：外层评估泛化、内层选参数
- 真实案例：影像 AI 同患者切片泄漏、基因组学全样本选择、多中心标准化遗留中心效应

---

### 案例 08：交叉验证

**背景**：对比 KFold、StratifiedKFold、RepeatedKFold、LOOCV、Nested CV 五种方法。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [80_module0_data_and_pipeline.md](notes/80_module0_data_and_pipeline.md) | L1 |
| 1 | [81_module1_split_and_kfold.md](notes/81_module1_split_and_kfold.md) | L1 |
| 2 | [82_module2_repeated_and_loocv.md](notes/82_module2_repeated_and_loocv.md) | L2 |
| 3 | [83_module3_nested_and_summary.md](notes/83_module3_nested_and_summary.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 单次划分的问题：随机种子敏感，AUC 极差可能很大
- K-Fold 交叉验证：将数据分成 K 份，每份轮流做测试集
- K 值的选择：K=5 或 K=10 最常见
- StratifiedKFold：保持每折的类别比例与原始数据一致
- 分层抽样的重要性：医学数据（不平衡）必须用 StratifiedKFold
- Mean±Std：交叉验证的 AUC 均值和标准差，标准差越小越稳定

**代码理解重点：**
- `KFold(n_splits=5, shuffle=True, random_state=42)` K-Fold
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 分层 K-Fold
- `cross_val_score(model, X, y, cv=skf, scoring='roc_auc')` 交叉验证
- `np.mean(scores)` 和 `np.std(scores)` 汇总

**关键概念：** K-Fold、StratifiedKFold、Mean±Std、泛化性能、随机种子敏感

#### L2 方法深入

**在 L1 基础上增加：**
- RepeatedKFold：多次重复 K-Fold 取平均，均值更精确，标准差更大
- LOOCV（留一法）：每次留一个样本做测试，几乎无偏但方差大、计算成本高
- 偏差-方差-成本的三角权衡：K-Fold（偏差中、方差低、成本低）、LOOCV（偏差低、方差高、成本高）
- 为什么 LOOCV 方差大：n 个训练集高度相似（只有 1 个样本不同）
- 不同 CV 方法的 AUC 对比（Mean±Std）

#### L3 理论扩展

**扩展内容：**
- 嵌套交叉验证（Nested CV）：外层评估泛化性能，内层选择超参数
- Nested CV 的重要性：避免调参偏倚，高水平论文标配
- 不同外折可能选出不同参数：这就是嵌套 CV 要告诉你的
- CV 与泄漏：Pipeline 每折独立 fit，但 CV 本身不防泄漏
- 评估可信度：可信度 = Mean / Std（越大越可信）
- 时序 CV：保持时间顺序，测试集时间晚于训练集

---

### 案例 09：模型比较

**背景**：在统一框架下对比 LR、KNN、SVM、DT、RF、XGBoost、LightGBM 七种模型。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [90_module0_data_loading.md](notes/90_module0_data_loading.md) | L1 |
| 1 | [91_module1_pipeline_cv_framework.md](notes/91_module1_pipeline_cv_framework.md) | L1 |
| 2 | [92_module2_seven_models.md](notes/92_module2_seven_models.md) | L2 |
| 3 | [93_module3_training_evaluation.md](notes/93_module3_training_evaluation.md) | L2 |
| 4 | [94_module4_visualization_results.md](notes/94_module4_visualization_results.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 七种模型的基本概念：
  - LR（逻辑回归）：线性决策边界，可解释性强
  - KNN（K 近邻）：距离度量，邻居投票
  - SVM（支持向量机）：最大间隔超平面
  - DT（决策树）：递归二分，规则可视化
  - RF（随机森林）：Bagging + 随机特征子空间
  - XGBoost：梯度提升 + 正则化 + 并行化
  - LightGBM：Leaf-wise 生长 + GOSS + EFB，更快更省内存
- 评估指标：AUC、Recall、Precision、F1、Brier Score、PR-AUC、Accuracy
- 统一 Pipeline + CV 框架：公平比较各模型

**代码理解重点：**
- 每个模型的 `__init__` 参数配置
- `Pipeline([('scaler', StandardScaler()), ('model', model)])`
- `cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')`

**关键概念：** LR、RF、XGBoost、AUC、集成学习、Pipeline

#### L2 方法深入

**在 L1 基础上增加：**
- 7 种模型的配置参数详解（n_estimators、max_depth、learning_rate 等）
- 模型选择策略：可解释性→LR/DT，性能→RF/XGBoost，稳定性→RF/LR
- 速度 vs 性能的权衡：LR 最快，XGBoost/LightGBM 最强
- 交叉验证下的稳定性（σ 跨折标准差）：RF 通常最稳定
- 多维度对比：AUC 条形图、稳定性雷达图、速度 vs 性能散点图、ROC 曲线
- 模型排名的变化：不同样本量下最优模型可能不同

#### L3 理论扩展

**扩展内容：**
- 集成学习三大流派：Bagging（并行）、Boosting（串行）、Stacking（学习投票）
- XGBoost 原理深入：二阶泰勒展开、正则化、列采样、Level-wise 生长
- LightGBM 原理深入：Leaf-wise 生长、GOSS 梯度单边采样、EFB 互斥特征捆绑
- 表格深度学习：TabNet、NODE、FT-Transformer
- 深度森林 gcForest：级联结构 + 多粒度扫描
- 超参数优化：Grid Search、Random Search、Bayesian Optimization、群体智能算法

---

### 案例 10：不平衡数据

**背景**：理解类别不平衡的准确率悖论，对比四种重采样策略，掌握 SMOTE 的正确使用方法。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [100_module0_data_loading_and_eda.md](notes/100_module0_data_loading_and_eda.md) | L1 |
| 1 | [101_module1_accuracy_paradox.md](notes/101_module1_accuracy_paradox.md) | L1 |
| 2 | [102_module2_resampling_strategies.md](notes/102_module2_resampling_strategies.md) | L2 |
| 3 | [103_module3_smote_leakage.md](notes/103_module3_smote_leakage.md) | L2 |
| 4 | [104_module4_cv_resampling.md](notes/104_module4_cv_resampling.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 类别不平衡的度量：Imbalance Ratio（IR），本数据集 IR = 1.4:1（轻度不平衡）
- 准确率悖论：全预测多数类就能获得高准确率，但 Recall = 0
- 为什么 Accuracy 在不平衡数据中危险：需要结合 Recall 和 PR-AUC
- SMOTE 的基本原理：找到少数类样本的 K 个近邻，在样本与近邻之间随机插值合成新样本
- 评估指标：ROC-AUC vs PR-AUC，PR-AUC 对不平衡更敏感
- 混淆矩阵：TP、TN、FP、FN 的含义

**代码理解重点：**
- `SMOTE(random_state=42).fit_resample(X_train, y_train)` SMOTE
- `confusion_matrix(y_true, y_pred)` 混淆矩阵
- `precision_recall_curve(y_true, y_proba)` PR 曲线

**关键概念：** 准确率悖论、SMOTE、混淆矩阵、PR-AUC、类别权重

#### L2 方法深入

**在 L1 基础上增加：**
- 四种重采样策略对比：不重采样、欠采样、过采样、SMOTE、ADASYN、SMOTEENN
- ADASYN vs SMOTE：ADASYN 在难分类区域生成更多样本
- SMOTE 泄漏：先在全数据上 SMOTE 再划分 → 合成样本污染测试集 → AUC 虚高
- 正确做法：先划分，再仅对训练集 SMOTE
- CV 中的 SMOTE：SMOTE 必须在每折内部独立执行（Pipeline + SMOTE）
- 阈值移动：不改变数据分布，调整决策阈值
- 类别权重：`class_weight='balanced'` 等效于给少数类更大权重

#### L3 理论扩展

**扩展内容：**
- 医学代价分析：FP vs FN 的临床代价，筛查→Recall 优先，确诊→Precision 优先
- 高级合成方法：DeepSMOTE、CTGAN、VAE
- 极端不平衡处理：异常检测（Isolation Forest）
- 评价指标选择：PR-AUC 比 ROC-AUC 更诚实

---

### 案例 11：校准与 DCA

**背景**：评估预测概率的可靠性（校准度），通过决策曲线分析（DCA）评估临床实用性。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [110_module0_data_loading.md](notes/110_module0_data_loading.md) | L1 |
| 1 | [111_module1_calibration_curves.md](notes/111_module1_calibration_curves.md) | L2 |
| 2 | [112_module2_brier_decomposition.md](notes/112_module2_brier_decomposition.md) | L2 |
| 3 | [113_module3_dca_analysis.md](notes/113_module3_dca_analysis.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 校准度的概念：预测概率与实际观测频率的一致性
- 校准曲线：X 轴=预测概率，Y 轴=实际频率，完美校准=对角线
- 过度自信：预测概率偏高，实际频率偏低
- 不够自信：预测概率偏低，实际频率偏高
- AUC 与校准度是独立的维度：高 AUC 不等于高校准
- DCA 的概念：决策曲线分析，评估在不同决策阈值下的净获益

**代码理解重点：**
- `calibration_curve(y_true, y_proba, n_bins=10)` 校准曲线
- 校准曲线的绘制和解读
- DCA 曲线的绘制和解读

**关键概念：** 校准度、校准曲线、过度自信、DCA、净获益

#### L2 方法深入

**在 L1 基础上增加：**
- Brier 分数：MSE 的概率版本，同时惩罚排序错误和校准错误
- Brier 分数的 Murphy 分解：鉴别力 - 校准度 + 不确定性
- KNN 的 Brier 假象：Brier 低但校准差（因为预测概率集中在 0/1）
- 校准改善方法：Platt Scaling（Sigmoid 校准）、Isotonic Regression（保序回归）
- `CalibratedClassifierCV` 的使用
- DCA 三条基线：Treat All（全部治疗）、Treat None（全部不治疗）、Model
- 临床获益范围：模型优于两条基线的决策阈值区间

#### L3 理论扩展

**扩展内容：**
- HL 检验（Hosmer-Lemeshow）：χ² 统计量，局限是样本量敏感和分组方式影响
- DCA 的临床解读：低阈值→筛查，高阈值→化疗，不同阈值不同决策
- 可用性评估流程：AUC→校准曲线→Brier→DCA→最佳阈值
- DCA 变体：标准 DCA、多状态 DCA、生存 DCA、成本效益 DCA
- 常见误解：DCA 和 AUC 回答不同问题，负获益≠没用

---

### 案例 12：模型解释（SHAP + LIME）

**背景**：用 SHAP 和 LIME 解释模型预测，从全局解释到局部解释。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [120_module0_data_model_baseline.md](notes/120_module0_data_model_baseline.md) | L1 |
| 1 | [121_module1_global_interpretation.md](notes/121_module1_global_interpretation.md) | L1 |
| 2 | [122_module2_local_interpretation.md](notes/122_module2_local_interpretation.md) | L2 |
| 3 | [123_module3_shap_vs_lime_stability.md](notes/123_module3_shap_vs_lime_stability.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 模型可解释性的重要性：医学场景不能只看预测结果
- 三种全局特征重要性方法：
  - Gini Importance：基于树模型的 Gini 减少量
  - Permutation Importance：打乱特征后性能下降量
  - SHAP Importance：基于 Shapley Value，最严谨
- SHAP 值的核心概念：base_value + Σ SHAP = 预测值，正值推高预测，负值推低预测
- SHAP 可视化：蜂群图（summary_plot）、条形图（bar_plot）
- LIME 的基本概念：局部扰动采样 → 黑箱预测 → 拟合局部线性模型

**代码理解重点：**
- `shap.TreeExplainer(model).shap_values(X)` 计算 SHAP 值
- `shap.summary_plot(shap_values, X)` 蜂群图
- `shap.waterfall_plot(explainer(X.iloc[i]))` 瀑布图
- `LimeTabularExplainer().explain_instance()` LIME 解释

**关键概念：** SHAP 值、特征重要性、蜂群图、瀑布图、LIME、全局解释 vs 局部解释

#### L2 方法深入

**在 L1 基础上增加：**
- Gini vs Permutation vs SHAP 三种重要性的对比：Gini 有高基数偏倚，Permutation 有负值（冗余），SHAP 最严谨
- Permutation Importance 的教学陷阱：负值≠噪声，可能是特征冗余
- SHAP vs LIME：Shapley Value vs 局部线性，TreeExplainer 快 vs 采样，确定性 vs 随机性
- 选型指南：树模型→TreeExplainer，任意模型→KernelExplainer 或 LIME，论文→SHAP
- LIME 稳定性问题：数据密度影响、边缘不稳定、固定 random_state 增大采样

#### L3 理论扩展

**扩展内容：**
- Shapley Value 四大公理：效率、对称性、虚拟性、可加性
- SHAP 的局限性：计算成本、独立性假设、相关≠因果、异常值敏感
- 其他解释方法：PDP、ICE、Anchors、Grad-CAM、Integrated Gradients
- SP-LIME：代表性子集选择，子模优化
- 医学论文 SHAP 标准流程：声明 TreeExplainer → Summary + Dependence → 临床文献关联 → Waterfall 叙事

---

### 案例 12b：高级 SHAP 可视化

**背景**：从"哪些特征重要"进阶到"特征如何影响预测"，通过二次拟合、3D 空间、矩阵热图、网络图等高级可视化。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [125_module0_data_model_shap.md](notes/125_module0_data_model_shap.md) | L1 |
| 1 | [126_module1_advanced_dependence_3d_matrix_pie.md](notes/126_module1_advanced_dependence_3d_matrix_pie.md) | L2 |
| 2 | [127_module2_trend_network_quantiles.md](notes/127_module2_trend_network_quantiles.md) | L2 |

#### L1 核心概念与实践

**知识点：**
- 基础 SHAP vs 高级 SHAP：从"哪些重要"到"直曲线/联合分布/分区间贡献/交互结构"
- 依赖图中的二次拟合：线性 R² vs 二次 R²，ΔR² = R²_quad - R²_linear 判断非线性程度
- 高 R² 的含义：该特征的变化能很好地解释 SHAP 值的变化

**关键概念：** 二次拟合、ΔR²、非线性、基础 vs 高级

#### L2 方法深入

**在 L1 基础上增加：**
- 3D 特征空间：Top3 特征的联合分布，旋转交互，颜色=目标
- 力量矩阵热图：横向看特征稳定性，纵向看样本模式，相近行=相似决策
- 分组饼图：主导特征、辅助特征、次要特征的分组
- 趋势图：样本按预测概率排列，SHAP 累积趋势
- 交互网络图：节点=特征，边=交互强度，中心节点=决策核心
- 分位数分布：P10-P25-P50-P75-P90，跨度=贡献动态范围

#### L3 理论扩展

**扩展内容：**
- 五维评估框架：重要性 + 非线性 ΔR² + 交互强度 + 趋势一致性 + 分位数跨度
- 多项式 vs 非参数拟合：线性 1 阶、二次 2 阶、三次 3 阶、LOESS、GAM 样条
- 二次拟合的统计检验：F 检验 ΔR² 的显著性
- 特征工程决策树：非线性高→多项式或分箱，交互高→创建交互特征，跨度大→杠杆作用
- 论文进阶三部曲：必选 Summary+Bar，推荐 Dependence+拟合对比，高分 交互网络+趋势+分位数

---

### 案例 13：SHAP 交互分析

**背景**：量化特征之间的交互效应，理解它们如何联合影响预测。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [130_module0_data_model_shap.md](notes/130_module0_data_model_shap.md) | L1 |
| 1 | [131_module1_interaction_scan_dependence.md](notes/131_module1_interaction_scan_dependence.md) | L2 |
| 2 | [132_module2_matrix_ranking.md](notes/132_module2_matrix_ranking.md) | L2 |
| 3 | [133_module3_deepdive_method_comparison.md](notes/133_module3_deepdive_method_comparison.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 交互效应的三种类型：增强（Synergistic）、抑制（Antagonistic）、调节（Moderation）
- 交互依赖图的读法：颜色范围、颜色与 y 的相关性、趋势线弯曲
- 交互强度定义：M1 = |corr(SHAP_i, X_j)|，特征 j 的值变化时特征 i 的贡献是否变化

**关键概念：** 交互效应、增强/抑制/调节、交互依赖图

#### L2 方法深入

**在 L1 基础上增加：**
- 交互扫描：每个特征的最强交互对方是谁
- 交互矩阵热图：对称=互为最强，非对称=单向调节，中心节点=调节者
- 交互强度排名：全局 Top 10 交互对的排序
- 深度交互分析：单一特征与所有其他特征的交互全景

#### L3 理论扩展

**扩展内容：**
- 两种度量方法对比：M1 = corr(SHAP_i, X_j)（真正交互调节）vs M2 = corr(SHAP_i, SHAP_j)（共同贡献模式），交叉判断
- 交互→特征工程：高交互→创建交互特征，LR 有帮助但树模型已捕捉
- 交互→模型选择：高交互→树模型，低交互→可加模型可能够用
- Friedman's H-statistic：基于 PDP 的全局交互强度
- SHAP Interaction Values：`shap_interaction_values`，计算成本 = 标准 SHAP × N_features
- 论文金三角：交互矩阵热图全景 + Top 交互面板深度 + 临床意义文本解读

---

### 案例 14：SHAP 依赖分布

**背景**：引入数据密度信息，通过双轴依赖图和 Ratio 指标评估 SHAP 结论的可信度。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [140_module0_data_preprocessing.md](notes/140_module0_data_preprocessing.md) | L1 |
| 1 | [141_module1_model_shap.md](notes/141_module1_model_shap.md) | L1 |
| 2 | [142_module2_dual_axis_dependence.md](notes/142_module2_dual_axis_dependence.md) | L2 |
| 3 | [143_module3_summary_density_risk.md](notes/143_module3_summary_density_risk.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 传统依赖图的问题：缺少数据密度信息，稀疏区域的散点可能不可靠
- 双轴依赖图的结构：上层直方图密度 + 下层散点 + 拟合线
- 数据密度的重要性：样本密集区的结论可信，稀疏区的结论需谨慎

**关键概念：** 数据密度、双轴依赖图、稀疏区域、可信度

#### L2 方法深入

**在 L1 基础上增加：**
- 五步读图法：直方图形状→密集区模式→稀疏区模式→拟合弯曲→密度标注
- Ratio 风险评估：Ratio = sparse / dense，低风险 < 2，中风险 2-4，高风险 > 4
- 教学陷阱：高 R² + 高 Ratio = 极端值驱动的虚假结论
- 三类特征：全范围一致、边缘微调、边缘主导

#### L3 理论扩展

**扩展内容：**
- 特征可信度映射：高重要+低 Ratio（黄金）、高重要+中 Ratio（慎重）、高重要+高 Ratio（需验证）
- 特征工程优先级：低 Ratio → 无需处理，中 Ratio → Winsorization，高 Ratio → 检查合法性再分箱
- 离群值协同效应：IQR + Ratio > 4 + Boruta → 重要性可能被人为抬高
- Ratio 统计检验：Mann-Whitney U 检验密集区 vs 稀疏区 SHAP 值
- 分位数敏感性：20-80% 稳健保守，25-75% 标准，30-70% 严格小样本
- 三维评估框架：重要性 + 非线性 + 交互强度 + 密度风险

---

### 案例 15：SHAP 聚类分析

**背景**：基于 SHAP 值对样本聚类，发现不同决策模式的患者亚群。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [150_module0_data_model_shap_clustering.md](notes/150_module0_data_model_shap_clustering.md) | L1 |
| 1 | [151_module1_kmeans_clustering.md](notes/151_module1_kmeans_clustering.md) | L2 |
| 2 | [152_module2_pca_visualization_panel.md](notes/152_module2_pca_visualization_panel.md) | L2 |
| 3 | [153_module3_cluster_profiling_clinical.md](notes/153_module3_cluster_profiling_clinical.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- SHAP 聚类 vs 原始特征聚类：特征值相似 vs 决策逻辑相似
- 为什么要对 SHAP 值聚类：全局解释掩盖亚群，局部解释只看个体，聚类连接全局和局部
- PCA 对 SHAP 空间的可视化：SHAP 空间本质是低维的

**关键概念：** SHAP 聚类、决策逻辑相似、亚群分析

#### L2 方法深入

**在 L1 基础上增加：**
- K-Means 聚类：K 值选择（Silhouette Score）
- 聚类可视化：PCA 2D 散点图，颜色=聚类标签
- 相对重要性：rel_imp = cluster / global，> 1.0x 更重要，< 1.0x 更不重要
- 聚类画像：样本数、目标比例、PCA 位置、突出特征

#### L3 理论扩展

**扩展内容：**
- 其他聚类算法：DBSCAN（任意形状）、层次聚类、GMM（软分类）
- SHAP 空间本质低维的原因：模型决策主要由少数特征驱动
- 聚类→临床洞察的转化路径：发现亚群→临床含义→论文写作模板
- 全流程建议：SHAP 聚类→特征定义→外部验证→临床转化
- 论文位置：探索性分析/亚组分析/讨论

---

### 案例 16：SHAP 决策路径

**背景**：通过决策路径图展示单个预测是如何逐步产生的。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [160_module0_data_model_baseline.md](notes/160_module0_data_model_baseline.md) | L1 |
| 1 | [161_module1_representative_samples.md](notes/161_module1_representative_samples.md) | L2 |
| 2 | [162_module2_three_column_visualization.md](notes/162_module2_three_column_visualization.md) | L2 |
| 3 | [163_module3_joint_analysis.md](notes/163_module3_joint_analysis.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 决策路径的核心概念：基准值（base_value）→ 逐步累加 SHAP 值 → 最终预测
- 三列可视化的结构：折线图（累积路径）+ 条形图（特征贡献）+ 雷达图（特征值）
- 每条路径的含义：上升推高预测，下降推低预测

**关键概念：** 决策路径、基准值、累积 SHAP、三列可视化

#### L2 方法深入

**在 L1 基础上增加：**
- 五组代表性样本的选择：极低/中低/边际/中高/极高预测概率
- 路径形状分类：陡升型（少数强特征）、缓升型（多特征共同）、陡降型（主要驱动明确）
- 特征值雷达图：轴=特征，值=归一化，形状=画像
- 跨样本比较：一致模式 vs 特定样本模式

#### L3 理论扩展

**扩展内容：**
- 局限性：路径顺序人为、只展示 TopK、不反映交互、单样本局限
- 与其他方法对比：Waterfall（直观标准）、决策路径（多维整合）、LIME（快但不稳定）
- 排序方式影响：按 |SHAP| 降序/按特征值/随机，关注累积值终点
- 批量路径热图：每行=患者路径，颜色=SHAP 正负大小，相似模式
- 概率转换：log-odds → 概率，非线性改变可视化
- 临床决策支持：决策路径→临床报告→医生验证

---

### 案例 17：SHAP 稳定性 Bootstrap

**背景**：通过 30 次 Bootstrap 重采样，量化特征重要性的稳定性。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [170_module0_data_preprocessing.md](notes/170_module0_data_preprocessing.md) | L1 |
| 1 | [171_module1_bootstrap_loop.md](notes/171_module1_bootstrap_loop.md) | L2 |
| 2 | [172_module2_stability_stats.md](notes/172_module2_stability_stats.md) | L2 |
| 3 | [173_module3_visualization_results.md](notes/173_module3_visualization_results.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 核心问题：单次训练的特征重要性是否可靠？
- Bootstrap 基本概念：有放回抽样，生成略有差异的"新"数据集
- 五个量化指标：Mean SHAP、Std SHAP、CV（变异系数）、CI Width（置信区间宽度）、Rank ρ（排名一致性）
- 越重要越稳定的规律：真实信号在数据扰动中幸存，噪声忽高忽低

**关键概念：** Bootstrap、CV、CI、排名稳定性、越重要越稳定

#### L2 方法深入

**在 L1 基础上增加：**
- Bootstrap 循环：30 次重采样 → 30 次模型训练 → 30 组 SHAP 值
- 四象限判断：高重要+低 CV（可信）、低重要+高 CV（不可信）、高重要+高 CV（需领域知识）、低重要+低 CV（确实不重要）
- 不稳定原因：特征值大量为零、非零值出现频率差异大
- 六张子图：重要性+CI 条形图、CV 稳定性排名、Bootstrap 分布箱线图、变化轨迹、排名稳定性直方图、CI 宽度排名

#### L3 理论扩展

**扩展内容：**
- 半衰期分析：sem = std / sqrt(i)，判断迭代次数是否足够
- Sub-sampling Bootstrap：有放回 vs 无放回，有偏修正
- CV vs Bootstrap：CV 评估泛化稳定性，Bootstrap 评估重要性稳定性，互补
- 点估计→区间估计：单次值→均值±标准差+95%CI
- 论文用法：方法声明重采样，结果均值+CI+CV 表，讨论确认可靠+标记不稳定
- 迭代次数选择：10-20 快速探索，30-50 精度平衡，100-200 正式论文

---

### 案例 18：HTE DML

**背景**：用双重机器学习（DML）估计癌症转移对生存的异质性因果效应。

| 模块 | 笔记文件 | 难度 |
|:----:|----------|:----:|
| 0 | [180_module0_data_preprocessing.md](notes/180_module0_data_preprocessing.md) | L1 |
| 1 | [181_module1_drorthoforest.md](notes/181_module1_drorthoforest.md) | L2 |
| 2 | [182_module2_causalforestdml_comparison.md](notes/182_module2_causalforestdml_comparison.md) | L2 |
| 3 | [183_module3_inference.md](notes/183_module3_inference.md) | L3 |
| 4 | [184_module4_clustering.md](notes/184_module4_clustering.md) | L3 |

#### L1 核心概念与实践

**知识点：**
- 因果推断的四要素：Y（结果）、T（处理）、X（异质性特征）、W（控制变量）
- 相关≠因果：混淆变量（同时影响 T 和 Y）导致虚假关联
- ATE（平均处理效应）vs CATE（条件平均处理效应）vs ITE（个体处理效应）
- ATE 掩盖人群差异，CATE 揭示"谁受益、谁受损、谁不受影响"
- 观察性研究 vs 随机对照试验：非随机分配 → 需要因果推断方法

**代码理解重点：**
- Y/T/X/W 的构造：Y=target, T=Extension (LOCALIZADO=0, METÁSTASE=1), X=Age, W=控制变量
- `LabelEncoder` 和 `StandardScaler` 的预处理
- 子采样到 3000 条的原因（DML 计算量大）

**关键概念：** 处理/结果/混淆/异质性特征、ATE/CATE、相关≠因果、观察性研究

#### L2 方法深入

**在 L1 基础上增加：**
- DML 的核心机制：交叉拟合（K 折数据划分）→ 结果模型 + 倾向模型 → 残差计算 → 残差回归
- 正交化：效应估计与讨厌参数估计分离
- DROrthoForest：双重稳健性（结果模型或倾向模型至少一个正确→估计一致）
- CausalForestDML：最大化处理效应异质性（非 MSE），加权平均聚合
- 两种模型的 CATE 曲线对比可视化
- ATE 的近似计算和解读

#### L3 理论扩展

**扩展内容：**
- 双重稳健性的理论基础：为什么只要一个模型正确就够
- 半参数有效性：DML 的理论最优效率
- 恒定边际效应 vs 异质性效应：政策制定用恒定，个性化决策用异质性
- 处理效应推断：CATE 的 99% 置信区间和 p 值
- 基于 CATE 的聚类分析：K-Means 发现不同效应模式的子群体
- 反事实预测：Y_cf_1、Y_cf_0、ITE = Y_cf_1 - Y_cf_0
- 政策学习：CATE(x) > 阈值 → 推荐干预
- 敏感性分析：lambda_reg、subsample_ratio、n_estimators 的稳定性
- DML vs Logistic Regression：LR 假设线性常数效应，DML 无此假设且更鲁棒
- 其他 DML 变体：LinearDML、NonParamDML、KernelDML、SparseLinearDML

---

## 四、各级学习材料对照表

| 材料类型 | L1 | L2 | L3 |
|----------|:--:|:--:|:--:|
| 教学文档（lectures/） | 前序知识 + 实践目的 + 结果解读 | + 全部章节 | + 全部章节 |
| 教学文档扩展内容 | 跳过 | 阅读 | 深入理解 |
| 详细笔记（notes/） | L1 标记的模块 | L1+L2 标记的模块 | 全部模块 |
| 源代码（src/） | 逐行理解 L1 核心代码 | 逐行理解全部代码 | 吃透全部代码 |
| 知识树（knowledge_tree.md） | 浏览 | 参考 | 掌握 |
| 结果文件（results/） | 看懂 | 能复现 | 能解释 |

---

## 五、技能树达成情况

| 知识领域 | L1 | L2 | L3 |
|----------|:--:|:--:|:--:|
| Python 编程基础 | 会用 | 熟练 | 精通 |
| 统计学基础 | 核心概念 | 方法选择 | 理论推导 |
| EDA | 完整流程 | 深入分析 | 面向建模 |
| 统计分析/特征筛选 | 假设检验 | 多重比较 | 非参数/贝叶斯 |
| 预处理与插补 | 四种方法 | 方法对比 | 理论推导 |
| 特征工程 | 标准化方法 | 选择策略 | 高级构造 |
| 特征选择 | 相关性/VIF/LASSO | 六层体系 | 稳定性选择 |
| 降维与聚类 | PCA/K-Means | t-SNE/UMAP | 理论推导 |
| 数据泄漏 | 致命错误 | 检查清单 | 剂量效应 |
| 交叉验证 | K-Fold | CV 方法对比 | 嵌套 CV |
| 模型比较 | 模型概念 | 选型策略 | 集成理论 |
| 不平衡处理 | SMOTE 概念 | 重采样策略 | 高级合成 |
| 校准与 DCA | 校准概念 | 校准改善 | 临床解读 |
| SHAP 基础 | SHAP 值/可视化 | 三种重要性 | 理论公理 |
| SHAP 高级可视化 | 二次拟合 | 高级图表 | 五维评估 |
| SHAP 交互分析 | 交互概念 | 交互扫描 | 方法对比 |
| SHAP 依赖分布 | 密度概念 | Ratio 风险 | 可信度映射 |
| SHAP 聚类 | 聚类概念 | 聚类画像 | 临床转化 |
| SHAP 决策路径 | 路径概念 | 三列可视化 | 临床决策 |
| SHAP 稳定性 | Bootstrap/CV | 四象限判断 | 半衰期分析 |
| HTE DML | Y/T/X/W/ATE | DML 机制 | 理论扩展 |

---

## 六、使用建议

### 对于教师

1. **L1 适合**：本科机器学习入门课、医学信息学选修课、暑期培训班（2~3 周）
2. **L2 适合**：研究生机器学习课程、医学 AI 方向必修课（4~6 周）
3. **L3 适合**：博士生高阶课程、科研训练、论文级分析方法培训（6~8 周）

### 对于学生

1. **按顺序学习**：案例编号 01→18 是递进关系，每个案例都从 L1 开始
2. **逐级深入**：L1 核心概念掌握后再进入 L2 方法深入，L2 完成后再进入 L3 扩展
3. **自测标准**：能用自己的话解释每个知识点
4. **L2→L3 的升级标志**：能独立完成一个类似的 ML 项目，从数据清洗到模型解释

### 学习节奏

```
L1（核心概念）：
  第 1 周：案例 01 EDA（L1） + 案例 02 统计分析（L1）
  第 2 周：案例 03 预处理（L1） + 案例 04 特征工程（L1） + 案例 07 数据泄漏（L1）
  第 3 周：案例 08 交叉验证（L1） + 案例 10 不平衡（L1） + 案例 12 模型解释（L1）

L2（方法深入）：
  第 4 周：在 L1 基础上深入案例 01-04 + 案例 05 特征选择（L1+L2） + 案例 06 降维（L1+L2）
  第 5 周：案例 09 模型比较（L1+L2） + 案例 11 校准 DCA（L1+L2）
  第 6 周：案例 12b 高级 SHAP（L1+L2） + 复习总结

L3（理论扩展）：
  第 7 周：案例 13 交互（L1+L2+L3） + 案例 14 依赖（L1+L2+L3） + 案例 15 聚类（L1+L2+L3）
  第 8 周：案例 16 决策路径（L1+L2+L3） + 案例 17 稳定性（L1+L2+L3） + 案例 18 HTE DML（L1+L2+L3）
```
