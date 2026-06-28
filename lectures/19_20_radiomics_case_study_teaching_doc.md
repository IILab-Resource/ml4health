# 案例教程 19-20: 影像组学 (Radiomics) 端到端机器学习 — 教学文档

> 本案例把前 12 个教程的"表格数据 ML 流程"迁移到**医学影像数据**, 帮助学生掌握
> "换一个新数据集/新模态, 该怎么走完整流程"的迁移能力. 这是教程 01-18 之后的
> 第一个**影像数据综合案例**.

***

## 一、案例背景

| 项      | 内容                                           |
| ------ | -------------------------------------------- |
| 数据     | 100 张腹部 CT (DICOM + TIFF), 见 `data/ct_data/` |
| 任务     | 二分类: 预测 CT 是否增强 (Contrast: 1=增强, 0=平扫)       |
| 标签平衡   | 50/50 完全平衡 (无需处理不平衡)                         |
| 样本量    | 100 (小样本 → 凸显 p≫n 问题)                        |
| 临床特征   | Age (年龄 39\~83)                              |
| 影像组学特征 | 93 维 (pyradiomics 标准特征)                      |

**为什么选这个任务?** 增强 CT 含碘造影剂, 血管/脏器密度显著升高, 影像组学的一阶
统计 (密度均值/分位数) 与纹理特征应能有效区分 → 模型 AUC 高, 适合做教学示范, 且
结论可被临床直觉验证 (SHAP 解释会指向密度类特征).

***

## 二、影像组学核心概念 (新知识点)

前 18 个教程的数据都是"现成的表格", 而影像数据必须先**特征化**. 影像组学
(Lambin et al., 2012) 就是这套标准化流程:

```
①图像获取 → ②ROI分割 → ③特征提取 → ④特征分析 → ⑤建模
   DICOM      HU阈值      pyradiomics    本案例     本案例
```

### 2.1 HU 值 (Hounsfield Unit)

CT 的标准化密度刻度: 水=0, 空气=-1000, 骨=+400\~+1000, 软组织≈+40\~+80.
DICOM 像素值经 `HU = pixel × slope + intercept` 映射到 HU. **一切影像组学分析都
在 HU 空间进行**, 这是与普通图像 (0\~255 灰度) 的根本区别.

### 2.2 ROI (Region of Interest) 分割

真实影像组学需手工/算法勾画肿瘤或器官. 本数据集无分割标注, 故采用**体部自动分割**:
`HU > -300` 取身体区域 → 保留最大连通域 (剔除背景空气与床板噪声).

> ⚠️ 教学陷阱: ROI 质量直接决定特征质量. 换数据时, ROI 策略必须重新设计
> (如肿瘤任务需勾画肿瘤, 不能用体部).

### 2.3 特征类 (pyradiomics)

| 类          | 含义          | 典型特征                                              |
| ---------- | ----------- | ------------------------------------------------- |
| firstorder | 一阶统计 (密度分布) | Mean, Median, Energy, Entropy, Skewness, Kurtosis |
| glcm       | 灰度共生矩阵 (纹理) | Contrast, Correlation, Homogeneity, Idn           |
| glrlm      | 灰度游程矩阵      | RunEntropy, RunLengthNonUniformity                |
| glszm      | 灰度大小区域矩阵    | SmallAreaEmphasis, ZoneEntropy                    |
| gldm       | 灰度依赖矩阵      | DependenceEntropy, LargeDependenceEmphasis        |
| ngtdm      | 灰度邻域差矩阵     | Coarseness, Complexity                            |

本案例 93 维特征 → 印证"影像组学特征会很多", **必须特征选择**.

***

## 三、★ 迁移对照表 (本案例核心教学价值)

下表把影像组学案例的每一步与前面教程 01-12 显式对照, 让学生看清:
**哪些步骤完全通用, 哪些步骤因数据特性而需要调整**.

| 步骤         | 表格数据 (教程 01-12)          | 影像组学 (本案例 19-20)              | 相同 / 不同                   |
| ---------- | ------------------------ | ----------------------------- | ------------------------- |
| **数据加载**   | `pd.read_csv` 直接读        | DICOM → HU → ROI → 特征 (教程 19) | 🔴 不同: 影像需先特征化            |
| **①EDA**   | 缺失值/分布/离群                | 特征量级跨度/相关性冗余                  | 🟡 相同思路, 不同关注点 (影像特征高度冗余) |
| **②统计**    | T检验/MW-U/卡方              | Mann-Whitney U + BH-FDR       | 🟢 相同 (影像特征偏态 → 用非参数)     |
| **③预处理**   | 均值插补/编码/标准化              | StandardScaler (量级跨13个数量级)    | 🟢 相同, 但标准化是必选项           |
| **④特征选择**  | 相关性→VIF→LASSO→RF→Boruta  | **LASSO (影像组学标准)**            | 🟡 方法子集相同, 影像组学惯用 LASSO   |
| **⑤防泄漏**   | Pipeline + 训练集内fit       | 同: 先划分, 训练集内标准化+LASSO         | 🟢 完全相同 (教程 07 的核心)       |
| **⑥交叉验证**  | K-Fold/分层/重复/嵌套          | 重复分层 5-Fold×10                | 🟢 完全相同                   |
| **⑦建模**    | LR/SVM/KNN/树/RF/XGB/LGBM | LR/SVM/RF/XGB/LightGBM        | 🟢 完全相同 (小样本下 LR/SVM 常最优) |
| **⑧不平衡**   | SMOTE/阈值/加权              | 数据平衡, 跳过                      | 🟡 相同方法, 看数据是否失衡          |
| **⑨校准DCA** | 校准曲线/Brier/DCA           | 同                             | 🟢 完全相同                   |
| **⑩SHAP**  | 蜂群/依赖/瀑布/LIME            | 蜂群/重要性/瀑布                     | 🟢 完全相同                   |

**一句话迁移法则**: 流程骨架 (EDA→统计→预处理→特征选择→CV→建模→校准→SHAP) 完全
通用; 影像数据的特殊性只在前端 (需特征化) 和特征选择 (p≫n 必须 LASSO 稀疏).

***

## 四、为什么特征选择用 LASSO?

用户问题: *"影像组学特征会很多, 选一个最常用的方法"*.

**答: LASSO (L1 正则逻辑回归) 是影像组学论文 (radiomics signature) 的事实标准.**

- Aerts et al., *Nature* 2014 (引用 5000+) 的影像组学 signature 即用 LASSO.
- 优点: ① 自动稀疏 (系数压到 0 = 自动选特征); ② 线性可解释; ③ p≫n 时仍稳定.

本案例标准流程 (教程 20 第 ③④ 步):

1. 相关性去冗余 (|r|>0.9): 94 → 39 特征
2. LASSO (L1, 5-fold CV 选 C): 39 → **13 个 signature 特征**
3. 后续所有建模只在这 13 个特征上做 (解决 p≫n)

***

## 五、运行指南

### 5.1 环境 (专用 conda 环境)

<br />

```bash
# 创建环境 (Python 3.11 + pyradiomics 从源码编译)
conda create -n radiomics -c conda-forge python=3.11 scikit-learn pandas numpy \
    scipy matplotlib seaborn tqdm pydicom simpleitk -y
conda activate radiomics
pip install versioneer
# pyradiomics 需从 git 源码编译 (sdist 缺 Cython 头文件):
git clone --depth 1 --branch v3.1.0 https://github.com/AIM-Harvard/pyradiomics.git
cd pyradiomics && pip install --no-build-isolation .
pip install xgboost lightgbm imbalanced-learn boruta shap lime
```

###

***

## 六、典型结果

| 指标              | 值                                   | 说明                |
| --------------- | ----------------------------------- | ----------------- |
| LASSO signature | 13 个特征                              | 从 93 维稀疏到 13 维    |
| LogReg CV AUC   | 0.95 ± 0.05                         | 训练集内 5×10 重复分层 CV |
| LogReg 测试 AUC   | 0.96                                | holdout 20 样本     |
| SHAP Top 特征     | firstorder\_RobustMAD, 90Percentile | 密度类特征主导, 符合临床预期   |

> 教学解读: SHAP 指向"一阶密度统计"为最强预测因子, 这正是造影剂提升组织密度的
> 直接体现 → 模型学到了**物理上可解释**的特征, 而非伪相关.

***

## 七、思考题 (迁移练习)

1. 若任务改为"区分良恶性肿瘤" (而非增强/平扫), ROI 策略应如何调整? signature 会变吗?
2. 本案例样本量仅 100, 若扩到 1000, 是否还需要 LASSO? 可否直接用 RF/XGBoost?
3. 若 CT 是 3D 体积 (多层) 而非单层, `force2D` 应如何设置? shape 特征会怎样?
4. 为什么 Age 在统计检验中不显著 (p≈0.09), 而影像组学特征大多显著? 这说明什么?
5. 把本案例的 LASSO 换成 Boruta (教程 05), signature 会一致吗? 哪个更适合小样本?

***

##

