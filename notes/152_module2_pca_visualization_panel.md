# 模块 2：PCA 降维与 2×3 综合可视化面板

> 本模块是案例教程 15「SHAP 聚类分析」的**可视化核心模块**。在模块 1 中，我们用 K-Means 在 6 维 SHAP 空间中聚类，得到最佳聚类数 k=2（Silhouette=0.6263），并用 PCA 把 SHAP 矩阵降到 2 维（PC1+PC2 解释 92.32% 方差）。本模块将把所有分析结果整合到一张 **2×3 综合可视化面板**中，从 6 个不同角度展示 SHAP 聚类的发现。 
>
> 本模块最核心的知识点有三个：**一是 2×3 面板的布局设计**——6 个子图分别回答"聚类结构、特征重要性、目标分布、聚类大小、特征均值、画像总结"6 个问题；**二是 matplotlib 子图坐标系的操作**——理解 `plt.subplot(2, 3, k)`、`ax.scatter`、`ax.bar`、`ax.boxplot`、`ax.pie`、`ax.imshow`、`ax.text` 各自的参数；**三是如何把聚类中心从 6 维 SHAP 空间投影到 2D PCA 空间**——用 `pca.transform(kmeans.cluster_centers_)` 实现坐标系转换。

***

## 学习目标

学完本模块后，你将能够：

1. **理解 2×3 面板的设计思路**：知道 6 个子图各自回答什么问题，以及为什么这样排列。
2. **掌握 matplotlib 子图布局**：会用 `plt.subplot(2, 3, k)` 创建 2 行 3 列的子图网格。
3. **掌握散点图绘制**：会用 `ax.scatter` 绘制 PCA 投影的聚类散点图，包括颜色映射、点大小、透明度、图例。
4. **掌握分组条形图绘制**：会用 `ax.bar` 绘制各聚类的特征重要性对比，包括分组偏移、宽度计算、标签旋转。
5. **掌握箱线图绘制**：会用 `ax.boxplot` 绘制各聚类的目标分布，并叠加文字标注。
6. **掌握饼图绘制**：会用 `ax.pie` 绘制聚类大小饼图，包括百分比标注、颜色填充。
7. **掌握热图绘制**：会用 `ax.imshow` 绘制聚类特征均值热图，包括颜色映射、数值标注、colorbar。
8. **掌握文字画像绘制**：会用 `ax.text` 在子图中绘制多行文字总结。
9. **理解 PCA 投影的坐标系转换**：知道如何用 `pca.transform` 把 6 维聚类中心投影到 2D。
10. **理解** **`plt.tight_layout`** **和** **`bbox_inches='tight'`**：知道它们如何避免子图重叠和标签被裁剪。

***

## 一、本模块在整体流程中的位置

```
模块 0：数据 → 模型 → SHAP 值矩阵 sv (3000, 6)
                    ↓
模块 1：sv → K-Means 聚类 → clusters (3000,)
       sv → PCA 降维 → sv_pca (3000, 2)
                    ↓
模块 2：sv + clusters + sv_pca → 2×3 可视化面板  ← 本模块
                    ↓
模块 3：clusters + sv → 相对重要性 → 临床画像
```

本模块的输入包括：

- `sv`：SHAP 值矩阵（用于特征重要性、相对重要性）
- `clusters`：聚类标签（用于分组着色）
- `sv_pca`：PCA 降维结果（用于散点图）
- `X_shap`：标准化后的特征值（用于特征均值热图）
- `y_shap`：真实标签（用于目标分布）
- `kmeans.cluster_centers_`：聚类中心（用于散点图标注）
- `pca`：PCA 模型（用于投影聚类中心）
- `feature_order`：特征排序（用于选择 Top N 特征）

输出是一张图片：`img/19a_shap_clustering_panel.png`。

***

## 二、面板整体设计

### 2.1 打印模块标题

```python
print("\n" + "=" * 70)
print("[3] 综合可视化面板 (2×3)")
print("=" * 70)
```

### 2.2 颜色配置

```python
cluster_colors = plt.cm.tab10(np.linspace(0, 0.8, best_n))
```

#### 逐行解释

- **`plt.cm.tab10`**：matplotlib 内置的离散颜色映射，有 10 种颜色。
- **`np.linspace(0, 0.8, best_n)`**：在 \[0, 0.8] 区间生成 `best_n=2` 个等距点，即 `[0.0, 0.8]`。
- **`plt.cm.tab10([0.0, 0.8])`**：从 tab10 颜色映射中取这两个位置的颜色，得到 2 个 RGB 颜色。

> 💡 **小贴士：为什么取 \[0, 0.8] 而不是 \[0, 1]？**
>
> tab10 的颜色在 0\~1 之间循环。取 \[0, 0.8] 是为了避开颜色映射末尾的颜色（如纯黄、纯白），这些颜色在白底图上不易识别。
>
> 对于 2 个簇，取 0.0 和 0.8 对应蓝色和红色，对比鲜明。

### 2.3 创建画布和总标题

```python
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'SHAP Clustering Analysis: {best_n} Clusters (Silhouette={silhouette_scores[best_n]:.4f})'
             f'\nClustering based on SHAP value patterns (not raw features)',
             fontsize=14, fontweight='bold')
```

#### 逐行解释

- **`plt.figure(figsize=(20, 12))`**：创建画布，大小 20×12 英寸。
  - 20 英寸宽 × 12 英寸高，比例 5:3，适合 2×3 布局。
  - 在 150 DPI 下，图片尺寸为 3000×1800 像素。
- **`fig.suptitle(...)`**：设置**总标题**（super title），位于整个画布顶部。
  - 第一行：`SHAP Clustering Analysis: 2 Clusters (Silhouette=0.6263)`
  - 第二行：`Clustering based on SHAP value patterns (not raw features)`
  - `fontsize=14`：字体大小 14。
  - `fontweight='bold'`：粗体。
  - 第二行特别强调"基于 SHAP 值模式（不是原始特征）"，提醒读者聚类的输入是 SHAP 值。

***

## 三、子图 1：PCA 投影散点图

### 3.1 代码

```python
# === 子图 1: PCA 投影 → SHAP 空间的聚类结构 ===
ax1 = plt.subplot(2, 3, 1)
for c in range(best_n):
    mask = clusters == c
    ax1.scatter(sv_pca[mask, 0], sv_pca[mask, 1],
                c=[cluster_colors[c]], s=30, alpha=0.5,
                edgecolors='gray', linewidth=0.3,
                label=f'Cluster {c+1} (n={mask.sum()})')

centers_pca = pca.transform(kmeans.cluster_centers_)
ax1.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='red', marker='*', s=400, edgecolors='black', linewidth=1,
            label='Centroids', zorder=5)
ax1.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
ax1.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
ax1.set_title('SHAP Value Space (PCA Projection)\nColor = Cluster',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=7, loc='best')
ax1.grid(True, alpha=0.2)
```

### 3.2 逐行解释

#### 创建子图

- **`ax1 = plt.subplot(2, 3, 1)`**：在 2 行 3 列的网格中创建第 1 个子图（左上角）。返回 `ax1` 是 Axes 对象，后续所有绘图操作都基于它。

#### 绘制每个簇的散点

- **`for c in range(best_n):`**：遍历每个簇（c=0, 1）。
- **`mask = clusters == c`**：布尔掩码，标记属于簇 c 的样本。
- **`ax1.scatter(sv_pca[mask, 0], sv_pca[mask, 1], ...)`**：绘制散点。
  - `sv_pca[mask, 0]`：簇 c 样本的 PC1 坐标。
  - `sv_pca[mask, 1]`：簇 c 样本的 PC2 坐标。
  - `c=[cluster_colors[c]]`：点颜色，用列表包裹确保所有点同色。
  - `s=30`：点大小 30（单位是点的平方）。
  - `alpha=0.5`：透明度 0.5，让重叠点可见。
  - `edgecolors='gray'`：点边缘颜色为灰色。
  - `linewidth=0.3`：点边缘线宽 0.3。
  - `label=f'Cluster {c+1} (n={mask.sum()})'`：图例标签，显示簇编号和样本数。

#### 投影聚类中心

- **`centers_pca = pca.transform(kmeans.cluster_centers_)`**：把 6 维聚类中心投影到 2D PCA 空间。
  - `kmeans.cluster_centers_`：形状 `(2, 6)`，两个簇中心的 6 维 SHAP 坐标。
  - `pca.transform(...)`：用模块 1 学习的 PCA 模型，把 6 维坐标转为 2 维。
  - `centers_pca`：形状 `(2, 2)`，两个簇中心的 (PC1, PC2) 坐标。

> 💡 **小贴士：为什么要** **`pca.transform`** **聚类中心？**
>
> K-Means 在 6 维 SHAP 空间聚类，簇中心也是 6 维的。但散点图是 2D 的（PC1, PC2）。
>
> 要在散点图上标注簇中心，必须把 6 维中心投影到 2D。`pca.transform` 就是做这个投影——用与原始数据相同的 PCA 变换，保证坐标系一致。

#### 绘制聚类中心

- **`ax1.scatter(centers_pca[:, 0], centers_pca[:, 1], ...)`**：绘制簇中心。
  - `centers_pca[:, 0]`：两个中心的 PC1 坐标。
  - `centers_pca[:, 1]`：两个中心的 PC2 坐标。
  - `c='red'`：红色。
  - `marker='*'`：星形标记。
  - `s=400`：点大小 400（比普通点大很多，醒目）。
  - `edgecolors='black'`：黑色边缘。
  - `linewidth=1`：边缘线宽 1。
  - `label='Centroids'`：图例标签。
  - `zorder=5`：图层顺序 5（越大越在上），保证中心点在散点之上。

#### 设置坐标轴和标题

- **`ax1.set_xlabel(f'PC 1 ({pca.explained_variance_ratio_[0]:.1%})')`**：x 轴标签，显示 PC1 及其方差解释率（如 `PC 1 (74.0%)`）。
- **`ax1.set_ylabel(f'PC 2 ({pca.explained_variance_ratio_[1]:.1%})')`**：y 轴标签，显示 PC2 及其方差解释率（如 `PC 2 (18.3%)`）。
- **`ax1.set_title(...)`**：子图标题，两行：`SHAP Value Space (PCA Projection)` 和 `Color = Cluster`。
- **`ax1.legend(fontsize=7, loc='best')`**：图例，字体 7，位置自动选择最佳。
- **`ax1.grid(True, alpha=0.2)`**：显示网格，透明度 0.2。

### 3.3 子图 1 解读

![SHAP 聚类分析面板](../img/19a_shap_clustering_panel.png)

从子图 1 可以看到：

- **Cluster 1**（蓝色）偏左紧凑，集中在 PC1 较小的区域。
- **Cluster 2**（橙色）偏右分散，PC1 较大。
- 两个红色星形是簇中心，明显分开。
- 两个簇在 PC1 方向上有清晰分界，这与 PC1 解释 74% 方差一致。

***

## 四、子图 2：聚类特征重要性对比

### 4.1 代码

```python
# === 子图 2: 聚类特征重要性对比 (分组条形图) ===
ax2 = plt.subplot(2, 3, 2)
cluster_mean_shap = []
for c in range(best_n):
    mask = clusters == c
    mean_shap = np.abs(sv[mask]).mean(axis=0)
    cluster_mean_shap.append(mean_shap)

top_n_feat = min(6, len(feature_names))
top_feat_idx = feature_order[:top_n_feat]
x = np.arange(len(top_feat_idx))
width = 0.8 / best_n

for c in range(best_n):
    values = [cluster_mean_shap[c][idx] for idx in top_feat_idx]
    ax2.bar(x + c * width, values, width,
            label=f'Cluster {c+1}', color=cluster_colors[c], alpha=0.85, edgecolor='white')

ax2.set_xlabel('Feature', fontsize=10)
ax2.set_ylabel('Mean |SHAP Value|', fontsize=10)
ax2.set_title('Feature Importance by Cluster',
              fontsize=11, fontweight='bold')
ax2.set_xticks(x + width * (best_n - 1) / 2)
ax2.set_xticklabels([feature_names[idx][:15] for idx in top_feat_idx], rotation=25, ha='right')
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.2, axis='y')
```

### 4.2 逐行解释

#### 计算每个簇的平均 SHAP 重要性

- **`cluster_mean_shap = []`**：存储每个簇的 6 维平均 SHAP 重要性。
- **`for c in range(best_n):`**：遍历每个簇。
- **`mask = clusters == c`**：簇 c 的样本掩码。
- **`mean_shap = np.abs(sv[mask]).mean(axis=0)`**：
  - `sv[mask]`：簇 c 样本的 SHAP 值，形状 `(n_c, 6)`。
  - `np.abs(...)`：取绝对值。
  - `.mean(axis=0)`：沿样本轴求均值，得到 `(6,)` 数组，是该簇每个特征的平均绝对 SHAP 值。
- **`cluster_mean_shap.append(mean_shap)`**：追加到列表。

#### 选择 Top N 特征

- **`top_n_feat = min(6, len(feature_names))`**：最多显示 6 个特征（本教程正好 6 个）。
- **`top_feat_idx = feature_order[:top_n_feat]`**：按全局重要性排序，取前 6 个特征的索引。
- **`x = np.arange(len(top_feat_idx))`**：x 轴位置，`[0, 1, 2, 3, 4, 5]`。
- **`width = 0.8 / best_n`**：每个条形的宽度。`0.8` 是一组特征占的总宽度，除以簇数得到每个条形的宽度。对于 2 个簇，width=0.4。

#### 绘制分组条形图

- **`for c in range(best_n):`**：遍历每个簇。
- **`values = [cluster_mean_shap[c][idx] for idx in top_feat_idx]`**：取簇 c 在 Top 6 特征上的重要性值。
- **`ax2.bar(x + c * width, values, width, ...)`**：绘制条形。
  - `x + c * width`：x 轴位置偏移，让不同簇的条形并排显示。
  - `values`：条形高度。
  - `width`：条形宽度。
  - `label=f'Cluster {c+1}'`：图例标签。
  - `color=cluster_colors[c]`：条形颜色。
  - `alpha=0.85`：透明度 0.85。
  - `edgecolor='white'`：白色边缘，让相邻条形更易区分。

#### 设置坐标轴

- **`ax2.set_xlabel('Feature')`**：x 轴标签"Feature"。
- **`ax2.set_ylabel('Mean |SHAP Value|')`**：y 轴标签"Mean |SHAP Value|"。
- **`ax2.set_xticks(x + width * (best_n - 1) / 2)`**：x 轴刻度位置，居中于一组条形。
- **`ax2.set_xticklabels([feature_names[idx][:15] for idx in top_feat_idx], rotation=25, ha='right')`**：
  - `feature_names[idx][:15]`：特征名截断到 15 个字符（避免过长）。
  - `rotation=25`：旋转 25 度，避免重叠。
  - `ha='right'`：右对齐，让旋转后的标签贴近刻度。
- **`ax2.grid(True, alpha=0.2, axis='y')`**：只显示 y 轴网格。

### 4.3 子图 2 解读

从子图 2 可以看到：

- **year** 在两个簇中都是最重要特征，但 Cluster 2 的 year 重要性更高（条形更高）。
- **Diagnostic.means** 在 Cluster 2 中明显比 Cluster 1 高——这是 Cluster 2 的"旗帜"特征。
- **Extension** 在 Cluster 2 中也明显更高。
- **Age** 在 Cluster 1 中略高。
- 这预示着 Cluster 2 由"诊断方式和肿瘤扩展"驱动，Cluster 1 由"年份和年龄"驱动。

***

## 五、子图 3：目标分布箱线图

### 5.1 代码

```python
# === 子图 3: 目标分布 (VIVO 比例) per cluster ===
ax3 = plt.subplot(2, 3, 3)
cluster_target_data = []
cluster_vivo_pct = []
for c in range(best_n):
    mask = clusters == c
    cluster_target_data.append(y_shap[mask])
    vivo_pct = y_shap[mask].mean() * 100
    cluster_vivo_pct.append(vivo_pct)

bp = ax3.boxplot(cluster_target_data, labels=[f'Cluster {c+1}' for c in range(best_n)],
                 patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], cluster_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# 叠加 VIVO 比例标注
for c in range(best_n):
    ax3.text(c + 1, ax3.get_ylim()[1] * 0.9,
             f'VIVO={cluster_vivo_pct[c]:.1f}%',
             ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax3.set_xlabel('Cluster', fontsize=10)
ax3.set_ylabel('Target (VIVO=1 / MORTO=0)', fontsize=10)
ax3.set_title(f'Target Distribution by Cluster\n(Overall VIVO={y.mean()*100:.1f}%)',
              fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.2, axis='y')
```

### 5.2 逐行解释

#### 计算每个簇的 VIVO 比例

- **`cluster_target_data = []`**：存储每个簇的目标值数组。
- **`cluster_vivo_pct = []`**：存储每个簇的 VIVO 百分比。
- **`for c in range(best_n):`**：遍历每个簇。
- **`mask = clusters == c`**：簇 c 的样本掩码。
- **`cluster_target_data.append(y_shap[mask])`**：追加簇 c 的目标值数组（0/1）。
- **`vivo_pct = y_shap[mask].mean() * 100`**：VIVO 比例 = 目标值均值 × 100（因为 VIVO=1, MORTO=0）。
- **`cluster_vivo_pct.append(vivo_pct)`**：追加百分比。

#### 绘制箱线图

- **`bp = ax3.boxplot(cluster_target_data, ...)`**：绘制箱线图。
  - `cluster_target_data`：每个簇的目标值列表。
  - `labels=[f'Cluster {c+1}' for c in range(best_n)]`：x 轴标签。
  - `patch_artist=True`：允许填充箱体颜色（默认是白色）。
  - `widths=0.6`：箱体宽度 0.6。
- **`bp`**：返回的字典，包含 `boxes`（箱体）、`medians`（中位数线）、`whiskers`（须线）、`caps`（端帽）、`fliers`（异常点）。

#### 填充箱体颜色

- **`for patch, color in zip(bp['boxes'], cluster_colors):`**：遍历每个箱体和对应颜色。
- **`patch.set_facecolor(color)`**：设置箱体填充色。
- **`patch.set_alpha(0.7)`**：透明度 0.7。

#### 叠加 VIVO 比例标注

- **`for c in range(best_n):`**：遍历每个簇。
- **`ax3.text(c + 1, ax3.get_ylim()[1] * 0.9, ...)`**：在箱体上方添加文字。
  - `c + 1`：x 坐标（箱线图的 x 轴从 1 开始）。
  - `ax3.get_ylim()[1] * 0.9`：y 坐标，取 y 轴上限的 90%。
  - `f'VIVO={cluster_vivo_pct[c]:.1f}%'`：文字内容，如 `VIVO=18.4%`。
  - `ha='center'`：水平居中。
  - `fontsize=9, fontweight='bold'`：字体 9，粗体。
  - `bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)`：文字背景框，圆角，浅黄色，透明度 0.8。

#### 设置坐标轴

- **`ax3.set_xlabel('Cluster')`**：x 轴标签。
- **`ax3.set_ylabel('Target (VIVO=1 / MORTO=0)')`**：y 轴标签，说明 0=MORTO, 1=VIVO。
- **`ax3.set_title(f'Target Distribution by Cluster\n(Overall VIVO={y.mean()*100:.1f}%)')`**：标题，两行，第二行显示总体 VIVO 比例（17.1%）。
- **`ax3.grid(True, alpha=0.2, axis='y')`**：y 轴网格。

### 5.3 子图 3 解读

从子图 3 可以看到：

- **Cluster 1**：VIVO 比例 = 1.4%（极低），几乎所有样本都是 MORTO。
- **Cluster 2**：VIVO 比例 = 18.4%（较高），是 Cluster 1 的 13 倍！
- 总体 VIVO = 17.1%，Cluster 2 的 VIVO 比例甚至略高于总体。
- 这说明 Cluster 2 是"高存活率"亚群，Cluster 1 是"低存活率"亚群。

> 💡 **小贴士：箱线图在这里的含义**
>
> 由于目标是 0/1 二值变量，箱线图看起来有点奇怪：
>
> - 中位数 = 0 或 1（取决于多数类）。
> - 箱体高度 = 0 或 1（取决于分布）。
>
> 但 VIVO 比例标注才是关键信息——它直接告诉我们每个簇的存活率。

***

## 六、子图 4：聚类大小饼图

### 6.1 代码

```python
# === 子图 4: 聚类大小饼图 ===
ax4 = plt.subplot(2, 3, 4)
cluster_sizes = [np.sum(clusters == c) for c in range(best_n)]
wedges, texts, autotexts = ax4.pie(
    cluster_sizes,
    labels=[f'Cluster {c+1}\n(n={cluster_sizes[c]})' for c in range(best_n)],
    colors=cluster_colors, autopct='%1.1f%%',
    startangle=90, pctdistance=0.6)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_fontweight('bold')
ax4.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
```

### 6.2 逐行解释

#### 计算聚类大小

- **`cluster_sizes = [np.sum(clusters == c) for c in range(best_n)]`**：列表推导，计算每个簇的样本数。对于本教程，`[2381, 619]`。

#### 绘制饼图

- **`wedges, texts, autotexts = ax4.pie(...)`**：绘制饼图，返回三个对象：
  - `wedges`：饼图扇区（Wedge 对象列表）。
  - `texts`：外部标签文本。
  - `autotexts`：内部百分比文本。
- **参数**：
  - `cluster_sizes`：每个扇区的大小。
  - `labels=[f'Cluster {c+1}\n(n={cluster_sizes[c]})' for c in range(best_n)]`：外部标签，显示簇编号和样本数，如 `Cluster 1\n(n=2381)`。
  - `colors=cluster_colors`：扇区颜色。
  - `autopct='%1.1f%%'`：自动显示百分比，格式 `79.4%`。
  - `startangle=90`：从 12 点钟方向开始画。
  - `pctdistance=0.6`：百分比文本距离圆心的比例（0=圆心，1=边缘）。

#### 设置百分比文本样式

- **`for autotext in autotexts:`**：遍历每个百分比文本。
- **`autotext.set_color('white')`**：白色文字（在彩色扇区上易读）。
- **`autotext.set_fontsize(10)`**：字体 10。
- **`autotext.set_fontweight('bold')`**：粗体。

#### 设置标题

- **`ax4.set_title('Cluster Size Distribution', ...)`**：标题"Cluster Size Distribution"。

### 6.3 子图 4 解读

从子图 4 可以看到：

- **Cluster 1**：79.4%（n=2381）——主流簇。
- **Cluster 2**：20.6%（n=619）——少数簇。
- 这种"一大一小"的分布很常见——大多数患者遵循常规决策模式，少数患者有特殊模式。

***

## 七、子图 5：聚类特征均值热图

### 7.1 代码

```python
# === 子图 5: 聚类特征均值热图 (竖版) ===
ax5 = plt.subplot(2, 3, 5)
n_feat_heat = min(6, len(feature_names))
top_heat_idx = feature_order[:n_feat_heat]
cluster_mean_feat = np.zeros((best_n, n_feat_heat))
for c in range(best_n):
    mask = clusters == c
    cluster_mean_feat[c] = X_shap[mask][:, top_heat_idx].mean(axis=0)

vmax = max(abs(cluster_mean_feat.max()), abs(cluster_mean_feat.min()))
im = ax5.imshow(cluster_mean_feat.T, cmap='coolwarm', aspect='auto', vmin=-vmax, vmax=vmax)
ax5.set_yticks(range(n_feat_heat))
ax5.set_yticklabels([feature_names[idx][:15] for idx in top_heat_idx])
ax5.set_xticks(range(best_n))
ax5.set_xticklabels([f'Cluster {c+1}' for c in range(best_n)])
ax5.set_title('Cluster Feature Mean (Standardized)',
              fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax5, label='Mean Std Value', shrink=0.8)
for i in range(n_feat_heat):
    for j in range(best_n):
        val = cluster_mean_feat[j, i]
        text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
        ax5.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color=text_color)
```

### 7.2 逐行解释

#### 计算每个簇的特征均值

- **`n_feat_heat = min(6, len(feature_names))`**：热图显示的特征数，最多 6 个。
- **`top_heat_idx = feature_order[:n_feat_heat]`**：按全局重要性排序，取前 6 个特征的索引。
- **`cluster_mean_feat = np.zeros((best_n, n_feat_heat))`**：初始化 `(2, 6)` 数组，存储每个簇每个特征的平均值。
- **`for c in range(best_n):`**：遍历每个簇。
- **`mask = clusters == c`**：簇 c 的样本掩码。
- **`cluster_mean_feat[c] = X_shap[mask][:, top_heat_idx].mean(axis=0)`**：
  - `X_shap[mask]`：簇 c 样本的标准化特征值，形状 `(n_c, 6)`。
  - `[:, top_heat_idx]`：选取 Top 6 特征列。
  - `.mean(axis=0)`：沿样本轴求均值，得到 `(6,)` 数组。
  - 赋值给 `cluster_mean_feat[c]`。

#### 计算颜色范围

- **`vmax = max(abs(cluster_mean_feat.max()), abs(cluster_mean_feat.min()))`**：取最大绝对值，保证颜色对称（如 -0.5 和 +0.5 颜色深度相同）。

#### 绘制热图

- **`im = ax5.imshow(cluster_mean_feat.T, ...)`**：绘制热图。
  - `cluster_mean_feat.T`：转置，形状 `(6, 2)`，让特征在 y 轴、簇在 x 轴。
  - `cmap='coolwarm'`：颜色映射，蓝色=负值，红色=正值，白色=0。
  - `aspect='auto'`：自动调整宽高比，让热图填满子图。
  - `vmin=-vmax, vmax=vmax`：颜色范围对称。

#### 设置坐标轴

- **`ax5.set_yticks(range(n_feat_heat))`**：y 轴刻度位置。
- **`ax5.set_yticklabels([feature_names[idx][:15] for idx in top_heat_idx])`**：y 轴标签，特征名（截断到 15 字符）。
- **`ax5.set_xticks(range(best_n))`**：x 轴刻度位置。
- **`ax5.set_xticklabels([f'Cluster {c+1}' for c in range(best_n)])`**：x 轴标签，簇编号。
- **`ax5.set_title('Cluster Feature Mean (Standardized)', ...)`**：标题，强调"标准化"后的值。

#### 添加 colorbar

- **`plt.colorbar(im, ax=ax5, label='Mean Std Value', shrink=0.8)`**：添加颜色条。
  - `im`：热图对象。
  - `ax=ax5`：关联的子图。
  - `label='Mean Std Value'`：颜色条标签。
  - `shrink=0.8`：颜色条高度缩小到 80%，避免过长。

#### 添加数值标注

- **`for i in range(n_feat_heat):`**：遍历每个特征（行）。
- **`for j in range(best_n):`**：遍历每个簇（列）。
- **`val = cluster_mean_feat[j, i]`**：第 j 簇第 i 特征的均值。
- **`text_color = 'white' if abs(val) > vmax * 0.5 else 'black'`**：如果值很大（颜色深），用白字；否则用黑字。
- **`ax5.text(j, i, f'{val:.2f}', ...)`**：在 (j, i) 位置添加数值标注。
  - `ha='center', va='center'`：水平垂直居中。
  - `fontsize=9`：字体 9。
  - `color=text_color`：文字颜色。

### 7.3 子图 5 解读

从子图 5 可以看到（实际运行数据）：

| 特征               | Cluster 1 | Cluster 2 |
| ---------------- | --------- | --------- |
| year             | -0.35     | +1.32     |
| Diagnostic.means | -0.04     | -0.00     |
| Code.Profession  | -0.02     | +0.15     |
| Age              | -0.03     | +0.15     |
| Raca.Color       | -0.01     | +0.07     |
| Extension        | -0.01     | -0.03     |

**关键发现**：

- **year**：Cluster 1 = -0.35（年份较早），Cluster 2 = +1.32（年份较晚）。这是最显著的差异！
- **Code.Profession**：Cluster 2 略高（+0.15）。
- **Age**：Cluster 2 略高（+0.15）。
- 其他特征差异不大。

> 💡 **小贴士：为什么 Cluster 2 的 year 这么高？**
>
> year = +1.32 表示 Cluster 2 患者的诊断年份比平均水平高 1.32 个标准差——也就是诊断年份较晚（近年）。
>
> 这可能反映了医疗技术的进步：近年诊断的患者，模型更多依赖"诊断方式"和"肿瘤扩展"来判断预后，而不是单纯看年份。

***

## 八、子图 6：聚类画像文字总结

### 8.1 代码

```python
# === 子图 6: 聚类画像表 (文字) ===
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
profile_lines = ["Cluster Profile Summary", "=" * 30]
for c in range(best_n):
    mask = clusters == c
    n = mask.sum()
    vivo_pct = y_shap[mask].mean() * 100
    # 找出该聚类中最重要的特征 (区别于其他聚类)
    this_imp = np.abs(sv[mask]).mean(axis=0)  # (n_features,)
    global_imp = np.abs(sv).mean(axis=0)
    # 相对重要性: 该聚类 vs 全局
    rel_imp = this_imp / (global_imp + 1e-8)
    # 相对重要性最高的特征
    top_rel_idx = np.argsort(rel_imp)[::-1][:3]
    top_rel_feats = [f"{feature_names[i]}(x{rel_imp[i]:.2f})" for i in top_rel_idx]

    # 显著高于全局的特征
    profile_lines.append(f"\nCluster {c+1} (n={n}, {vivo_pct:.1f}% VIVO):")
    profile_lines.append(f"  Dominant features: {', '.join(top_rel_feats)}")
    # 特征均值倾向
    feat_tend = []
    for idx in top_heat_idx:
        mean_val = cluster_mean_feat[c, list(top_heat_idx).index(idx)]
        feat_tend.append(f"{feature_names[idx]}={mean_val:+.2f}")
    profile_lines.append(f"  Feature values: {' | '.join(feat_tend[:4])}")

ax6.text(0, 1, '\n'.join(profile_lines), transform=ax6.transAxes,
         fontsize=9, fontfamily='monospace', va='top', ha='left')
ax6.set_title('Cluster Profiling', fontsize=11, fontweight='bold')
```

### 8.2 逐行解释

#### 关闭坐标轴

- **`ax6.axis('off')`**：关闭坐标轴，因为这个子图只显示文字。

#### 初始化文字行

- **`profile_lines = ["Cluster Profile Summary", "=" * 30]`**：第一行是标题，第二行是分隔线。

#### 遍历每个簇生成画像

- **`for c in range(best_n):`**：遍历每个簇。
- **`mask = clusters == c`**：簇 c 的样本掩码。
- **`n = mask.sum()`**：簇 c 的样本数。
- **`vivo_pct = y_shap[mask].mean() * 100`**：簇 c 的 VIVO 百分比。

#### 计算相对重要性

- **`this_imp = np.abs(sv[mask]).mean(axis=0)`**：簇 c 的平均绝对 SHAP 值，形状 `(6,)`。
- **`global_imp = np.abs(sv).mean(axis=0)`**：全局平均绝对 SHAP 值，形状 `(6,)`。
- **`rel_imp = this_imp / (global_imp + 1e-8)`**：相对重要性 = 簇 c 重要性 / 全局重要性。
  - `+ 1e-8`：避免除以 0（虽然 SHAP 重要性通常不为 0）。
  - `rel_imp > 1`：该特征在簇 c 中比全局更重要。
  - `rel_imp < 1`：该特征在簇 c 中比全局更不重要。

#### 找出 Top 3 相对重要性特征

- **`top_rel_idx = np.argsort(rel_imp)[::-1][:3]`**：按相对重要性降序排序，取前 3 个特征的索引。
- **`top_rel_feats = [f"{feature_names[i]}(x{rel_imp[i]:.2f})" for i in top_rel_idx]`**：格式化为字符串，如 `Diagnostic.means(x1.83)`。

#### 添加画像文字

- **`profile_lines.append(f"\nCluster {c+1} (n={n}, {vivo_pct:.1f}% VIVO):")`**：添加簇标题行。
- **`profile_lines.append(f"  Dominant features: {', '.join(top_rel_feats)}")`**：添加主导特征行。
- **`feat_tend = []`**：存储特征均值倾向。
- **`for idx in top_heat_idx:`**：遍历 Top 6 特征。
- **`mean_val = cluster_mean_feat[c, list(top_heat_idx).index(idx)]`**：取该簇该特征的均值。
- **`feat_tend.append(f"{feature_names[idx]}={mean_val:+.2f}")`**：格式化为字符串，如 `year=+1.32`。
- **`profile_lines.append(f"  Feature values: {' | '.join(feat_tend[:4])}")`**：添加特征值行，只显示前 4 个。

#### 绘制文字

- **`ax6.text(0, 1, '\n'.join(profile_lines), transform=ax6.transAxes, ...)`**：在子图中绘制文字。
  - `0, 1`：x=0（左），y=1（上），即左上角。
  - `'\n'.join(profile_lines)`：用换行符连接所有行。
  - `transform=ax6.transAxes`：坐标系是子图相对坐标（0\~1），不是数据坐标。
  - `fontsize=9`：字体 9。
  - `fontfamily='monospace'`：等宽字体，让对齐更整齐。
  - `va='top', ha='left'`：垂直顶对齐，水平左对齐。
- **`ax6.set_title('Cluster Profiling', ...)`**：标题"Cluster Profiling"。

### 8.3 子图 6 解读

从子图 6 可以看到（实际运行数据）：

```
Cluster Profile Summary
==============================

Cluster 1 (n=2381, 1.4% VIVO):
  Dominant features: Age(x1.10), Raca.Color(x1.06), year(x1.04)
  Feature values: year=-0.35 | Diagnostic.means=-0.04 | Code.Profession=-0.02 | Age=-0.03

Cluster 2 (n=619, 18.4% VIVO):
  Dominant features: Diagnostic.means(x1.83), Extension(x1.55), Code.Profession(x1.14)
  Feature values: year=+1.32 | Diagnostic.means=-0.00 | Code.Profession=+0.15 | Age=+0.15
```

**关键发现**：

- **Cluster 1**：相对重要性最高的特征是 Age (1.10x)、Raca.Color (1.06x)、year (1.04x)——都接近 1.0，说明这个簇"没有特别突出的特征"，是"主流模式"。
- **Cluster 2**：相对重要性最高的是 Diagnostic.means (1.83x)、Extension (1.55x)、Code.Profession (1.14x)——Diagnostic.means 和 Extension 明显高于 1.0，说明这个簇由"诊断方式"和"肿瘤扩展"驱动。

> 💡 **小贴士：相对重要性是本教程的核心创新**
>
> 传统分析只看"全局重要性"或"簇内重要性"，无法回答"这个特征在该簇中是否异常突出"。
>
> 相对重要性 = 簇内重要性 / 全局重要性，能直接回答这个问题：
>
> - `> 1.0`：该特征在该簇中比全局更重要（异常突出）。
> - `< 1.0`：该特征在该簇中比全局更不重要（被弱化）。
>
> 这个指标将在模块 3 详细讨论。

***

## 九、保存图片

### 9.1 代码

```python
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "19a_shap_clustering_panel.png"), dpi=150, bbox_inches='tight')
plt.close()
print("  [图] 19a_shap_clustering_panel.png 已保存")
```

### 9.2 逐行解释

- **`plt.tight_layout()`**：自动调整子图间距，避免标题、标签、图例重叠。这是 matplotlib 中最常用的布局调整函数。
- **`plt.savefig(...)`**：保存图片。
  - `os.path.join(IMG_DIR, "19a_shap_clustering_panel.png")`：保存路径。
  - `dpi=150`：分辨率 150 DPI（每英寸 150 像素）。对于 20×12 英寸的画布，图片尺寸为 3000×1800 像素。
  - `bbox_inches='tight'`：自动裁剪空白边缘，避免标签被裁剪。
- **`plt.close()`**：关闭画布，释放内存。在循环绘图时尤其重要，避免内存泄漏。
- **`print("  [图] 19a_shap_clustering_panel.png 已保存")`**：打印保存信息。

### 9.3 实际运行结果

```
[3] 综合可视化面板 (2×3)
==============================================================================
  [图] 19a_shap_clustering_panel.png 已保存
```

 

![SHAP 聚类分析面板](../img/19a_shap_clustering_panel.png)

***

## 十、面板整体解读

把 6 个子图综合起来看，可以得到以下洞察：

### 10.1 聚类结构（子图 1）

- SHAP 空间有两个清晰的簇，PC1 方向分离明显。
- Cluster 1 紧凑（主流模式），Cluster 2 分散（特殊模式）。

### 10.2 特征重要性差异（子图 2）

- Cluster 2 的 Diagnostic.means 和 Extension 重要性明显更高。
- Cluster 1 的特征重要性较均匀，没有特别突出的特征。

### 10.3 目标分布差异（子图 3）

- Cluster 1 的 VIVO 比例 = 1.4%（几乎都预测死亡）。
- Cluster 2 的 VIVO 比例 = 18.4%（存活率是 Cluster 1 的 13 倍）。
- 这说明 Cluster 2 是"高存活率"亚群。

### 10.4 聚类大小（子图 4）

- Cluster 1 占 79.4%（主流）。
- Cluster 2 占 20.6%（少数）。

### 10.5 特征均值差异（子图 5）

- Cluster 2 的 year 明显更高（+1.32 vs -0.35）——近年诊断的患者。
- Cluster 2 的 Code.Profession 和 Age 略高。
- 其他特征差异不大。

### 10.6 画像总结（子图 6）

- Cluster 1：Age、Raca.Color、year 略突出（都接近 1.0x），是"主流模式"。
- Cluster 2：Diagnostic.means (1.83x)、Extension (1.55x) 异常突出，是"诊断方式驱动"的特殊模式。

### 10.7 综合结论

```
Cluster 1 — "主流模式" (79.4% 患者)
  画像: 大多数患者, 年份较早
  特征: 模型使用常规特征 (year, Age, Raca.Color) 进行预测
  VIVO: 极低 (1.4%) — 几乎都预测死亡
  解读: 这部分患者的预测"比较常规"

Cluster 2 — "特殊模式" (20.6% 患者)
  画像: 少数患者, 年份较晚, 存活率较高
  特征: Diagnostic.means (1.83x) — 最重要的区分
        Extension (1.55x)
  VIVO: 较高 (18.4%) — 存活率远高于总体
  解读: 这部分患者的预测主要由"诊断方式"和"肿瘤扩展"驱动
```

***

## 小贴士

### 1. 2×3 面板的设计原则

6 个子图按"从结构到细节"的顺序排列：

- 第 1 行（子图 1-3）：聚类结构、特征重要性、目标分布——回答"聚类是什么样的"。
- 第 2 行（子图 4-6）：聚类大小、特征均值、画像总结——回答"聚类有多大、特征差异、总结"。

这种布局让读者从左到右、从上到下逐步深入理解聚类结果。

### 2. `plt.cm.tab10` 的颜色选择

`tab10` 是 matplotlib 默认的离散颜色映射，有 10 种颜色。本教程用 `np.linspace(0, 0.8, best_n)` 取前 2 种颜色（蓝色和橙色），对比鲜明。

如果你有 5+ 个簇，可以考虑用 `tab20`（20 种颜色）或自定义颜色列表。

### 3. `zorder` 的作用

`zorder` 控制图层顺序，越大越在上。本教程中聚类中心的 `zorder=5`，保证它在散点之上。如果不设 zorder，后绘制的元素会覆盖先绘制的，但显式设 zorder 更可靠。

### &#x20;

### &#x20;

***

## 本模块小结

本模块完成了 SHAP 聚类分析的**可视化核心**：

1. **面板设计**：创建 20×12 英寸的画布，2×3 布局，6 个子图从不同角度展示聚类结果。
2. **子图 1（PCA 散点图）**：用 `ax.scatter` 绘制 PCA 投影的聚类散点图，叠加红色星形聚类中心。Cluster 1 偏左紧凑，Cluster 2 偏右分散。
3. **子图 2（特征重要性对比）**：用 `ax.bar` 绘制分组条形图，对比两个簇在 Top 6 特征上的平均绝对 SHAP 值。Cluster 2 的 Diagnostic.means 和 Extension 明显更高。
4. **子图 3（目标分布箱线图）**：用 `ax.boxplot` 绘制各簇的目标分布，叠加 VIVO 比例标注。Cluster 1 的 VIVO=1.4%，Cluster 2 的 VIVO=18.4%。
5. **子图 4（聚类大小饼图）**：用 `ax.pie` 绘制饼图，显示 Cluster 1 占 79.4%、Cluster 2 占 20.6%。
6. **子图 5（特征均值热图）**：用 `ax.imshow` 绘制热图，显示各簇在 Top 6 特征上的标准化均值。Cluster 2 的 year=+1.32（近年诊断）是最显著的差异。
7. **子图 6（画像文字总结）**：用 `ax.text` 绘制多行文字，总结每个簇的样本数、VIVO 比例、主导特征（相对重要性）、特征均值。
8. **保存图片**：用 `plt.savefig` 保存为 `img/19a_shap_clustering_panel.png`，分辨率 150 DPI，`bbox_inches='tight'` 避免标签被裁剪。

**关键产物**：

- `img/19a_shap_clustering_panel.png`：2×3 综合可视化面板，包含 6 个子图。
- 这张图是本教程的"招牌图"，可以直接用于论文或报告。

**下一模块预告**：模块 3 将深入分析每个簇的"相对重要性"，计算 `rel_imp = this_imp / global_imp`，并讨论 Cluster 2 的 Diagnostic.means (1.83x) 和 Extension (1.55x) 的临床含义。
