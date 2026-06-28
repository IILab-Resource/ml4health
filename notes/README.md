
# 代码精读笔记

`notes/` 目录包含**逐行代码精读笔记**，是学习流程中的 Step 2 — 在你读完讲义、理解"为什么做"之后，用来理解"代码为什么这么写"。


## 笔记结构

每篇笔记按模块划分，遵循统一的结构：

```
模块标题 + 一句话概要
    ↓
学习目标（学完能做什么）
    ↓
一、代码逐段精讲 ← 核心内容
    ├── 代码段 + 行号引用
    ├── "为什么这么写"的解释
    └── 常见陷阱 / 教学要点标注
    ↓
二、……（后续段落，每个模块不同）
    ↓
三、关键概念对比（部分模块有）
```

### 每篇笔记 vs 讲义

| 维度 | 讲义（lectures/） | 笔记（notes/） |
|------|-----------------|---------------|
| 回答的问题 | 为什么需要做这件事？ | 这段代码为什么这么写？ |
| 关注层面 | 概念、方法论、流程 | 变量名、函数参数、代码逻辑 |
| 阅读时机 | 跑代码之前 | 跑代码之前 / 之中 |


## 四组笔记速览



### 教程 01：EDA（模块 00-04）

| 模块 | 内容 |
|------|------|
| [00_module0_data_loading.md](00_module0_data_loading.md) | 环境准备与数据加载 |
| [01_module1_data_overview.md](01_module1_data_overview.md) | 数据集概览与缺失值矩阵 |
| [02_module2_missing_values.md](02_module2_missing_values.md) | 缺失值深度分析 |
| [03_module3_distribution.md](03_module3_distribution.md) | 数值特征分布分析 |
| [04_module4_outliers.md](04_module4_outliers.md) | 离群值检测与处理 |

### 教程 02：统计分析（模块 10-14）

| 模块 | 内容 |
|------|------|
| [10_module0_data_loading.md](10_module0_data_loading.md) | 数据加载与目标变量创建 |
| [11_module1_feature_classification.md](11_module1_feature_classification.md) | 特征分类与方法选择 |
| [12_module2_numerical_tests.md](12_module2_numerical_tests.md) | 数值特征统计检验 |
| [13_module3_chi_square.md](13_module3_chi_square.md) | 分类特征卡方检验 |
| [14_module4_summary_visualization.md](14_module4_summary_visualization.md) | 结果汇总与可视化 |

### 教程 03：预处理与插补（模块 20-24）

| 模块 | 内容 |
|------|------|
| [20_module0_data_loading.md](20_module0_data_loading.md) | 数据加载与特征选择 |
| [21_module1_split_encoding.md](21_module1_split_encoding.md) | 数据集划分与编码 |
| [22_module2_imputation_methods.md](22_module2_imputation_methods.md) | 四种插补方法对比 |
| [23_module3_model_evaluation.md](23_module3_model_evaluation.md) | 模型训练与评估 |
| [24_module4_visualization.md](24_module4_visualization.md) | 结果可视化 |





### 教程 04：特征工程（模块 30-33）

| 模块 | 内容 |
|------|------|
| [30_module0_data_loading.md](30_module0_data_loading.md) | 数据加载与基础预处理 |
| [31_module1_scaling.md](31_module1_scaling.md) | 四种标准化方法比较 |
| [32_module2_feature_construction.md](32_module2_feature_construction.md) | 特征构造（多项式、交互、分组） |
| [33_module3_comparison.md](33_module3_comparison.md) | 标准化前后模型对比 |


### 教程 05：特征选择（模块 50-54）

| 模块 | 内容 |
|------|------|
| [50_module0_data_loading.md](50_module0_data_loading.md) | 数据加载与基础预处理 |
| [51_module1_correlation_vif.md](51_module1_correlation_vif.md) | 相关性分析与 VIF |
| [52_module2_filter_lasso.md](52_module2_filter_lasso.md) | 过滤法与 Lasso |
| [53_module3_rf_boruta.md](53_module3_rf_boruta.md) | 随机森林与 Boruta |
| [54_module4_validation_summary.md](54_module4_validation_summary.md) | 验证与总结 |


### 教程 06：降维与聚类（模块 60-64）

| 模块 | 内容 |
|------|------|
| [60_module0_data_loading.md](60_module0_data_loading.md) | 数据加载与预处理 |
| [61_module1_curse_of_dimensionality.md](61_module1_curse_of_dimensionality.md) | 维度灾难演示 |
| [62_module2_dim_reduction_visualization.md](62_module2_dim_reduction_visualization.md) | PCA/t-SNE/UMAP 降维可视化 |
| [63_module3_clustering.md](63_module3_clustering.md) | K-Means 与层次聚类 |
| [64_module4_pca_for_ml.md](64_module4_pca_for_ml.md) | PCA 用于机器学习加速 |


### 教程 07：数据泄漏（模块 70-73）

| 模块 | 内容 |
|------|------|
| [70_module0_data_loading.md](70_module0_data_loading.md) | 数据加载与划分 |
| [71_module1_standardization_leakage.md](71_module1_standardization_leakage.md) | 标准化中的数据泄漏 |
| [72_module2_feature_selection_leakage.md](72_module2_feature_selection_leakage.md) | 特征选择中的数据泄漏 |
| [73_module3_comprehensive_pipeline.md](73_module3_comprehensive_pipeline.md) | 综合 Pipeline 防范 |



### 教程 08：交叉验证（模块 80-83）

| 模块 | 内容 |
|------|------|
| [80_module0_data_and_pipeline.md](80_module0_data_and_pipeline.md) | 数据准备与 Pipeline 构建 |
| [81_module1_split_and_kfold.md](81_module1_split_and_kfold.md) | 数据划分与 KFold |
| [82_module2_repeated_and_loocv.md](82_module2_repeated_and_loocv.md) | RepeatedKFold 与 LOOCV |
| [83_module3_nested_and_summary.md](83_module3_nested_and_summary.md) | 嵌套交叉验证与总结 |




### 教程 09：模型比较（模块 91-94）

| 模块 | 内容 |
|------|------|
| [91_module1_pipeline_cv_framework.md](91_module1_pipeline_cv_framework.md) | Pipeline 与 CV 框架 |
| [92_module2_seven_models.md](92_module2_seven_models.md) | 七种模型配置 |
| [93_module3_training_evaluation.md](93_module3_training_evaluation.md) | 模型训练与评估 |
| [94_module4_visualization_results.md](94_module4_visualization_results.md) | 结果可视化 |



### 教程 10：不平衡数据（模块 100-104）

| 模块 | 内容 |
|------|------|
| [100_module0_data_loading_and_eda.md](100_module0_data_loading_and_eda.md) | 数据加载与 EDA |
| [101_module1_accuracy_paradox.md](101_module1_accuracy_paradox.md) | 准确率悖论 |
| [102_module2_resampling_strategies.md](102_module2_resampling_strategies.md) | 重采样策略（SMOTE/ADASYN/欠采样） |
| [103_module3_smote_leakage.md](103_module3_smote_leakage.md) | SMOTE 数据泄漏问题 |
| [104_module4_cv_resampling.md](104_module4_cv_resampling.md) | 交叉验证中的重采样 |


### 教程 11：校准与 DCA（模块 110-114）

| 模块 | 内容 |
|------|------|
| [110_module0_data_loading.md](110_module0_data_loading.md) | 数据加载与基准模型 |
| [111_module1_calibration_curves.md](111_module1_calibration_curves.md) | 校准曲线绘制 |
| [112_module2_brier_decomposition.md](112_module2_brier_decomposition.md) | Brier 分数分解 |
| [113_module3_dca_analysis.md](113_module3_dca_analysis.md) | 决策曲线分析 |



### 教程 12：模型解释（SHAP & LIME）（模块 120-123）

| 模块 | 内容 |
|------|------|
| [120_module0_data_model_baseline.md](120_module0_data_model_baseline.md) | 数据加载与模型训练 |
| [121_module1_global_interpretation.md](121_module1_global_interpretation.md) | 全局解释（特征重要性、SHAP） |
| [122_module2_local_interpretation.md](122_module2_local_interpretation.md) | 局部解释（Waterfall、LIME） |
| [123_module3_shap_vs_lime_stability.md](123_module3_shap_vs_lime_stability.md) | SHAP vs LIME 稳定性比较 |

### 教程 12b：高级 SHAP（模块 125-127）

| 模块 | 内容 |
|------|------|
| [126_module1_advanced_dependence_3d_matrix_pie.md](126_module1_advanced_dependence_3d_matrix_pie.md) | 依赖图、3D 空间、矩阵热图、饼图 |
| [127_module2_trend_network_quantiles.md](127_module2_trend_network_quantiles.md) | 趋势分析、网络图、分位数分析 |

### 教程 13：SHAP 交互分析（模块 130-133）

| 模块 | 内容 |
|------|------|
| [131_module1_interaction_scan_dependence.md](131_module1_interaction_scan_dependence.md) | 交互值扫描与依赖分析 |
| [132_module2_matrix_ranking.md](132_module2_matrix_ranking.md) | 交互矩阵与排序 |
| [133_module3_deepdive_method_comparison.md](133_module3_deepdive_method_comparison.md) | 深度分析与方法对比 |


## 使用方法

**推荐顺序**（与 STUDY_GUIDE.md 的四步法一致）：

```
讲义前半 → 本笔记 → 运行代码 → 讲义后半
```

具体来说：

1. 先读 `lectures/` 中对应教程的**前半部分**（前序知识 + 实践目的）
2. 打开本目录中的笔记，从 `00`（数据加载）开始逐篇阅读
3. 同时打开 `jupyter/` 中的对应代码文件，对照查看
4. 理解后再运行代码












