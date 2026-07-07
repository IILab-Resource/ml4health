# 小白运行第一个 Notebook：`01_eda_detail_tutorial.ipynb` 操作流程

目标：从零安装工具，用 Git 下载项目，用 Anaconda 管理 Python 环境，用 VS Code 打开并运行本项目第一个 Notebook：

```text
ml4health/jupyter/01_eda_detail_tutorial.ipynb
```

本流程默认系统为 Windows。

---

## 1. 总体主线

你要完成的主线只有一条：

```text
安装 Git
  ↓
安装 Anaconda
  ↓
安装 VS Code
  ↓
用 Git 下载项目
  ↓
用 conda 创建项目环境
  ↓
安装 Python 包
  ↓
用 VS Code 打开项目
  ↓
选择 conda 环境作为 Notebook 内核
  ↓
运行 01_eda_detail_tutorial.ipynb
```

---

## 2. 安装 Git

Git 用来下载 GitHub 上的代码项目。

安装可参考菜鸟教程：

```text
https://www.runoob.com/git/git-install-setup.html
```

### 2.1 下载 Git

打开：

```text
https://git-scm.com/downloads
```

选择 Windows 版本下载安装。

### 2.2 安装 Git

安装时大部分选项保持默认即可。

建议注意三点：

- 编辑器可保持默认。
- 终端选项选择支持在命令行中使用 `git`。
- 到 `Adjusting your PATH environment` 这一步时，建议选择 `Git from the command line and also from 3rd-party software`。这样会把 Git 加入 PATH，后续在 `Anaconda Prompt`、PowerShell、VS Code 终端里都可以直接使用 `git`。

### 2.3 检查 Git 是否安装成功

打开 Windows 开始菜单，搜索：

```text
Anaconda Prompt
```

打开后输入：

```bash
git --version
```

如果看到类似结果，说明安装成功：

```text
git version 2.xx.x
```

---

## 3. 安装 Anaconda

Anaconda 用来管理 Python 和各种数据分析包，适合初学者。

安装可参考菜鸟教程：

```text
https://www.runoob.com/python-qt/anaconda-tutorial.html
```

### 3.1 下载 Anaconda

打开：

```text
https://www.anaconda.com/download
```

下载 Windows 版本。

### 3.2 安装 Anaconda

安装时建议：

- 安装给当前用户即可。
- 如果这台电脑主要只用 Anaconda 这一套 Python，建议勾选或配置 `Add Anaconda to my PATH environment variable`，减少后续 `conda` 命令找不到的问题。
- 如果电脑里已经安装过多套 Python，勾选 Anaconda PATH 可能会影响其他 Python；这种情况下可以不勾选，但以后要固定使用 `Anaconda Prompt`。
- 建议勾选 `Register Anaconda as my default Python`，这样 VS Code 更容易识别 Anaconda 的 Python。

### 3.3 关于 Anaconda PATH 的实用建议

为了减少后续 `conda` 命令找不到的问题，初学者可以优先采用下面策略：

```text
只有一套 Python / 新电脑 / 专门用于本课程
→ 安装时添加 Anaconda 到 PATH

电脑上已有 Python、Miniconda、公司软件自带 Python
→ 不添加 Anaconda 到 PATH，使用 Anaconda Prompt
```

如果安装后在普通 PowerShell 或 VS Code 终端中输入 `conda --version` 提示找不到 `conda`，不是安装失败，通常只是 PATH 没配置。此时可以使用 `Anaconda Prompt`，或重新安装 Anaconda 时勾选添加 PATH。

常见 Anaconda 路径类似：

```text
C:\Users\你的用户名\anaconda3
C:\Users\你的用户名\anaconda3\Scripts
C:\Users\你的用户名\anaconda3\Library\bin
```

### 3.4 检查 conda 是否安装成功

打开 `Anaconda Prompt`，输入：

```bash
conda --version
```

如果看到类似结果，说明安装成功：

```text
conda 24.x.x
```

---

## 4. 安装 VS Code

VS Code 用来写代码、打开 GitHub 项目、运行 Notebook。

安装可参考菜鸟教程：

```text
https://www.runoob.com/vscode/vscode-windows-install.html
```

### 4.1 下载 VS Code

打开：

```text
https://code.visualstudio.com/
```

下载并安装 Windows 版本。

安装时建议勾选：

```text
Add to PATH
```

如果安装界面有下面选项，也建议勾选：

```text
Open with Code
Register Code as an editor for supported file types
```

这样后续可以在项目目录中直接运行：

```bash
code .
```

### 4.2 安装 VS Code 插件

打开 VS Code，点击左侧 Extensions，搜索并安装：

```text
Python
Jupyter
```

这两个插件都建议安装 Microsoft 发布的版本。

安装完成后，建议重启一次 VS Code。Notebook 能否顺利运行，主要依赖这两个插件和后面选择正确的 Python Kernel。

---

## 5. 用 Git 下载项目

选择一个专门放代码的文件夹，例如：

```text
D:\NJMU\works\AI Learning
```

打开 `Anaconda Prompt`，进入该目录：

```bash
D:
cd "D:\NJMU\works\AI Learning"
```

用 Git 下载项目：

```bash
git clone https://github.com/IILab-Resource/ml4health.git ml4health
```

例如：

```bash
git clone https://github.com/IILab-Resource/ml4health.git ml4health
```

下载完成后，进入项目：

```bash
cd ml4health
```

检查项目结构：

```bash
dir
```

你应该能看到类似：

```text
data
jupyter
lectures
notes
README.md
STUDY_GUIDE.md
```

本教程要运行的第一个程序在：

```text
ml4health\jupyter\01_eda_detail_tutorial.ipynb
```

如果输入 `git clone` 时提示找不到 `git`，说明 Git 没有正确加入 PATH。解决方法：

1. 重新打开终端再试一次。
2. 重新安装 Git，并在 PATH 选项中选择 `Git from the command line and also from 3rd-party software`。
3. 或打开 Git Bash 执行 `git clone`。

---

## 6. 确认数据文件是否存在

第一个 Notebook 会读取这个数据文件：

```text
ml4health\data\cancer_data_eng.csv
```

在项目根目录运行：

```bash
dir data
```

确认里面有：

```text
cancer_data_eng.csv
```

如果没有，需要根据项目 README 中的数据集链接下载数据，并放到：

```text
ml4health\data\
```

注意：文件名必须保持为：

```text
cancer_data_eng.csv
```

---

## 7. 创建 conda 环境

建议为本项目单独创建一个 Python 环境，避免和其他项目冲突。

在 `Anaconda Prompt` 中进入项目根目录：

```bash
cd "D:\NJMU\works\AI Learning\ml4health"
```

创建环境：

```bash
conda create -n ml4health python=3.10 -y
```

激活环境：

```bash
conda activate ml4health
```

以后每次运行这个项目，都先执行：

```bash
conda activate ml4health
```

---

## 8. 安装第一个 Notebook 需要的包

`01_eda_detail_tutorial.ipynb` 主要需要这些包：

```text
pandas
numpy
matplotlib
seaborn
scipy
ipykernel
jupyter
```

安装命令：

```bash
conda install pandas numpy matplotlib seaborn scipy ipykernel jupyter -y
```

如果你准备继续运行后续机器学习、统计建模和模型解释类 Notebook，可以一次性补充安装常用依赖：

```bash
pip install scikit-learn statsmodels imbalanced-learn shap lime
```

说明：第一个 EDA Notebook 不一定用到这些扩展包，但后续模块会逐渐用到。第一次只想跑通 `01_eda_detail_tutorial.ipynb`，先安装前面的 conda 命令即可。

把当前 conda 环境注册成 Jupyter 内核：

```bash
python -m ipykernel install --user --name ml4health --display-name "Python (ml4health)"
```

以后在 VS Code 里就可以选择：

```text
Python (ml4health)
```

作为 Notebook 内核。

---

## 9. 用 VS Code 打开项目

推荐打开项目根目录，而不是只打开单个 `.ipynb` 文件。项目根目录是：

```text
D:\NJMU\works\AI Learning\ml4health
```

也就是包含这些文件夹的目录：

```text
ml4health/
├── jupyter/
├── lectures/
├── notes/
├── img/
└── data/
```

打开整个项目文件夹的好处是：VS Code 能识别项目结构，Notebook 里的相对路径也更不容易出错。

在 `Anaconda Prompt` 中运行：

```bash
cd "D:\NJMU\works\AI Learning\ml4health"
code .
```

如果 `code .` 不可用，可以手动打开 VS Code：

```text
File → Open Folder → 选择 D:\NJMU\works\AI Learning\ml4health
```

---

## 10. 打开第一个 Notebook

在 VS Code 左侧文件栏中打开：

```text
jupyter/01_eda_detail_tutorial.ipynb
```

打开后，右上角会显示当前 Notebook 使用的 Python 内核。如果还没有选择内核，通常会显示：

```text
Select Kernel
```

点击它，按下面路径选择：

```text
Python Environments → Python (ml4health)
```

有些 VS Code 版本里显示为环境名 `ml4health`，有些会显示为 `Python (ml4health)`，选择这个 conda 环境即可。

如果没看到这个内核：

1. 关闭 VS Code。
2. 打开 `Anaconda Prompt`。
3. 执行：

```bash
conda activate ml4health
python -m ipykernel install --user --name ml4health --display-name "Python (ml4health)"
```

4. 重新打开 VS Code。

---

## 11. 运行前检查路径

这个 Notebook 里有类似代码：

```python
BASE_DIR = "../"
DATA_PATH = os.path.join(BASE_DIR, "data", "cancer_data_eng.csv")
IMG_DIR = os.path.join(BASE_DIR, "img")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
```

它默认当前运行位置是：

```text
ml4health/jupyter
```

所以 `../data/cancer_data_eng.csv` 才能正确指向：

```text
ml4health/data/cancer_data_eng.csv
```

同理，如果代码里出现：

```python
pd.read_csv("../data/xxx.csv")
```

那么数据文件就应该放在：

```text
ml4health/data/xxx.csv
```

这里的 `..` 表示“上一级目录”：Notebook 在 `ml4health/jupyter/`，上一级就是 `ml4health/`，所以 `../data/` 就是项目根目录下的 `data/`。

### 建议先运行这个检查单元

在 Notebook 最前面新建一个代码单元，运行：

```python
import os
print(os.getcwd())
```

如果输出结尾是：

```text
ml4health\jupyter
```

说明路径正确。

如果输出是：

```text
ml4health
```

可以在 Notebook 最前面先运行：

```python
%cd jupyter
```

或者把代码中的：

```python
BASE_DIR = "../"
```

改成：

```python
BASE_DIR = "."
```

但对初学者更推荐使用：

```python
%cd jupyter
```

这样不用改原始代码。

---

## 12. 正式运行 Notebook

在 VS Code Notebook 顶部点击：

```text
Run All
```

或者从第一个代码单元开始，一个一个点击左侧运行按钮。

推荐初学者第一次不要直接 Run All，而是逐格运行：

```text
先运行导入包
再运行路径和数据读取
再运行数据概况分析
再运行缺失值分析
再运行分布分析和画图
```

也可以理解为：

```text
第 1 个代码块 → 看输出
第 2 个代码块 → 看输出
第 3 个代码块 → 看输出
...
```

这样如果报错，容易知道是哪一步的问题。例如导入包报错，多半是环境依赖没装好；读取数据报错，多半是数据路径或文件名不对。

---

## 13. 运行成功的标志

如果运行成功，你会看到：

1. 代码单元没有红色报错。
2. 输出里显示数据集形状，例如样本数和变量数。
3. 项目目录下生成或更新图片：

```text
ml4health/img/
```

例如：

```text
01_target_distribution.png
02a_missing_percentage.png
02b_missing_pattern.png
03a_distribution_hist_kde.png
04a_outlier_iqr_vs_zscore.png
```

4. 项目目录下可能生成结果文件夹：

```text
ml4health/results/
```

---

## 14. 常见问题

### 问题 1：`ModuleNotFoundError: No module named 'pandas'`

原因：当前 Notebook 没有使用 `ml4health` 环境，或者环境里没安装包。

解决：

```bash
conda activate ml4health
conda install pandas numpy matplotlib seaborn scipy ipykernel jupyter -y
```

然后在 VS Code 右上角重新选择：

```text
Python (ml4health)
```

---

### 问题 2：`FileNotFoundError: cancer_data_eng.csv`

原因通常有两个：

第一，数据文件不存在。

检查：

```text
ml4health/data/cancer_data_eng.csv
```

第二，Notebook 当前运行目录不对。

在 Notebook 中运行：

```python
import os
print(os.getcwd())
```

如果不是 `ml4health/jupyter`，先运行：

```python
%cd jupyter
```

---

### 问题 3：VS Code 找不到 conda 环境

先在 `Anaconda Prompt` 中执行：

```bash
conda activate ml4health
python -m ipykernel install --user --name ml4health --display-name "Python (ml4health)"
```

然后重启 VS Code。

---

### 问题 4：`git`、`conda` 或 `code` 命令找不到

这通常是 PATH 没配置好。分别检查：

```bash
git --version
conda --version
code --version
```

如果 `git` 找不到：重新安装 Git，安装时选择 `Git from the command line and also from 3rd-party software`。

如果 `conda` 找不到：优先使用 `Anaconda Prompt`，或重新安装 Anaconda 并添加到 PATH。

如果 `code` 找不到：重新安装 VS Code，安装时勾选 `Add to PATH`。

---

### 问题 5：运行很慢

本项目的 `cancer_data_eng.csv` 比较大，约数百 MB。第一次读取可能较慢。

建议：

- 不要同时开很多软件。
- 不要把项目放在同步很慢的网盘目录中。
- 第一次运行耐心等待。
- 如果电脑内存较小，先只运行前几个单元。

---

## 15. 推荐学习顺序

第一个 Notebook 对应 EDA，即探索性数据分析。

建议按下面顺序学习：

```text
1. 先读 lectures/01_eda_teaching_doc.md 前半部分
2. 再看 notes/00_module0_data_loading.md
3. 打开 jupyter/01_eda_detail_tutorial.ipynb
4. 一格一格运行代码
5. 每运行一段，就回头看 notes/ 中对应说明
6. 跑通后再读 lectures/01_eda_teaching_doc.md 后半部分
```

本项目推荐的学习逻辑是：

```text
lectures/ 先建立概念
notes/    再理解代码为什么这样写
jupyter/  最后动手运行
```

---

## 16. 每次重新打开项目时怎么做

以后每次学习这个项目，只需要：

1. 打开 `Anaconda Prompt`。
2. 激活环境：

```bash
conda activate ml4health
```

3. 进入项目：

```bash
cd "D:\NJMU\works\AI Learning\ml4health"
```

4. 打开 VS Code：

```bash
code .
```

5. 打开：

```text
jupyter/01_eda_detail_tutorial.ipynb
```

6. 选择内核：

```text
Python (ml4health)
```

7. 逐格运行。

---

## 17. 初学者使用技巧

### 技巧 1：不要急着改原始 Notebook

第一次目标不是创新，而是：

```text
完整跑通
看懂输出
知道每段代码大概在做什么
```

等第一次完整跑通后，再尝试修改参数。

推荐保留原始文件，复制一份练习副本。例如把：

```text
jupyter/01_eda_detail_tutorial.ipynb
```

复制为：

```text
jupyter/01_eda_mycopy.ipynb
```

然后在副本里运行、改参数、写自己的笔记。这样原始教程文件仍然保留，方便以后对照。

如果你想单独建 `my_practice/` 文件夹保存练习副本，也可以，但要先运行 `os.getcwd()` 检查当前工作目录，再确认 `../data/` 是否仍然指向 `ml4health/data/`。如果路径不对，需要相应调整 `BASE_DIR`。

### 技巧 2：每次报错先看最后一行

Python 报错很长时，优先看最后一行，例如：

```text
ModuleNotFoundError
FileNotFoundError
KeyError
NameError
```

最后一行通常最关键。

### 技巧 3：Notebook 可以分块运行

Notebook 的好处是可以一格一格运行。

不要把它当成普通脚本直接全部运行。初学时建议：

```text
运行一格 → 看输出 → 理解 → 再运行下一格
```

### 技巧 4：确认当前环境

在 Notebook 中可以运行：

```python
import sys
print(sys.executable)
```

如果路径中包含：

```text
anaconda
envs
ml4health
```

说明正在使用正确环境。

### 技巧 5：确认当前路径

在 Notebook 中可以运行：

```python
import os
print(os.getcwd())
```

对于本 Notebook，理想情况是当前目录为：

```text
...\ml4health\jupyter
```

---

## 18. 最小成功标准

只要你完成下面 4 件事，就算第一个程序跑通：

- VS Code 成功打开 `01_eda_detail_tutorial.ipynb`
- Notebook 内核选择为 `Python (ml4health)`
- 能成功读取 `data/cancer_data_eng.csv`
- 能生成 EDA 图表到 `img/` 文件夹

完成后，就可以继续学习第二个 Notebook：

```text
jupyter/02_statistical_analysis.ipynb
```



