# pyradiomics 安装指南 (Linux aarch64 / Python 3.11+)

> 背景: pyradiomics 在 aarch64 架构上通过 pip 直接安装会失败,需要从源码编译。
> 本文档记录了在 Linux aarch64 (ARM 64位) 上从零编译安装 pyradiomics 的完整过程。
> 该流程也适用于其他 pip 安装失败的情况。

***

## 环境信息

| 项        | 版本/路径                   |
| -------- | ----------------------- |
| 系统       | Linux aarch64 (ARM64)   |
| Python   | 3.11.x (通过 conda 管理)    |
| conda 路径 | `/home/wjj/miniconda3/` |
| gcc      | `/usr/bin/gcc`          |

***

## 完整安装流程

### 第一步: 创建 conda 环境

```bash
# 用 conda-forge 创建 Python 3.11 环境 (自带 C 头文件,避免找不到 Python.h)
conda create -n radiomics -c conda-forge python=3.11 \
    scikit-learn pandas numpy scipy matplotlib seaborn tqdm \
    pydicom simpleitk -y

# 激活环境
conda activate radiomics
```

> **为什么要 Python 3.11?**
>
> - Python 3.12 的 `configparser.SafeConfigParser` 被移除,导致 pyradiomics 的
>   versioneer 构建脚本报错
> - Python 3.11 完全兼容且 conda-forge 有完整的预编译包

### 第二步: 安装 pip (conda 环境默认不带 pip)

```bash
# 用 ensurepip 引导安装 pip (无需 sudo)
python -m ensurepip --upgrade

# 验证
python -m pip --version
# 输出: pip 24.0 from .../site-packages/pip-24.0-py3-none-any.whl
```

### 第三步: 安装编译依赖

```bash
# 安装 versioneer (setup.py 需要它来生成版本号)
python -m pip install versioneer

# 如果后续编译失败提示找不到 Python.h,确认 conda 环境自带头文件:
ls $CONDA_PREFIX/include/python3.11/Python.h
# 预期: 存在该文件
```

### 第四步: 下载 pyradiomics git 源码

> **为什么要从 git 克隆而不直接 pip download?**
>
> - PyPI sdist 包 (`pyradiomics-3.1.0.tar.gz`) 缺少 Cython 生成的 `.h` 头文件
>   (`cmatrices.h`, `cshape.h`),导致 gcc 编译失败
> - git 仓库包含完整的 Cython 源文件,可以直接编译

```bash
# 在临时目录克隆 (--depth 1 快速拉取)
cd /tmp
git clone --depth 1 --branch v3.1.0 \
    https://github.com/AIM-Harvard/pyradiomics.git pyradiomics_git

# 确认头文件存在
ls pyradiomics_git/radiomics/src/*.h
# 预期: cmatrices.c  cmatrices.h  cshape.c  cshape.h  _cmatrices.c  _cshape.c
```

### 第五步: 给 setup.py 打补丁 (修复 include\_dirs 路径)

> **问题**: 编译命令中 `-I` 路径缺少 `radiomics/src/`,导致 `#include "cmatrices.h"` 找不到。

```bash
cd /tmp/pyradiomics_git

# 用 sed 或编辑器修改 setup.py
# 找到第 13-14 行:
#   commands = versioneer.get_cmdclass()
#   incDirs = [sysconfig.get_python_inc(), numpy.get_include()]
# 替换为:
```

```python
# 修改后的 setup.py 开头:
from distutils import sysconfig
import platform
import os

import numpy
from setuptools import Extension, setup
import versioneer

if platform.architecture()[0].startswith('32'):
  raise Exception('PyRadiomics requires 64 bits python')

commands = versioneer.get_cmdclass()
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "radiomics", "src"))
incDirs = [sysconfig.get_python_inc(), numpy.get_include(), _src_dir]

ext = [Extension("radiomics._cmatrices", ["radiomics/src/_cmatrices.c", "radiomics/src/cmatrices.c"],
                 include_dirs=incDirs),
       Extension("radiomics._cshape", ["radiomics/src/_cshape.c", "radiomics/src/cshape.c"],
                 include_dirs=incDirs)]
```

> **修改要点**: `incDirs` 末尾添加 `_src_dir`,让 gcc 能找到 `cmatrices.h` 和 `cshape.h`。

### 第六步: 编译安装

```bash
cd /tmp/pyradiomics_git

# --no-build-isolation: 使用当前环境的 numpy (避免重新 build numpy)
# 如果报错 SafeConfigParser,需先修复 versioneer.py:
#   sed -i 's/configparser.SafeConfigParser()/configparser.ConfigParser()/g' versioneer.py
# (Python 3.11 无此问题,可跳过)

python -m pip install --no-build-isolation .

# 成功标志:
#   Created wheel for pyradiomics: filename=pyradiomics-3.0.1a1-cp311-linux_aarch64.whl
#   Successfully built pyradiomics
#   Successfully installed pyradiomics
```

### 第七步: 安装其余 ML 依赖

```bash
python -m pip install xgboost lightgbm imbalanced-learn boruta shap lime
```

### 第八步: 验证安装

```bash
python -c "
from radiomics import featureextractor
import pydicom, SimpleITK as sitk
import radiomics, shap
print('pydicom:', pydicom.__version__)
print('SimpleITK:', sitk.Version_VersionString())
print('pyradiomics:', radiomics.__version__)
print('shap:', shap.__version__)
print('All OK!')
"
```

***

## 常见问题排查

| 错误信息                                                                        | 原因                  | 解决方案                                                                            |
| --------------------------------------------------------------------------- | ------------------- | ------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'versioneer'`                         | 缺少 versioneer       | `pip install versioneer`                                                        |
| `AttributeError: module 'configparser' has no attribute 'SafeConfigParser'` | Python 3.12+ 兼容问题   | 换 Python 3.11; 或 `sed -i 's/SafeConfigParser()/ConfigParser()/g' versioneer.py` |
| `fatal error: Python.h: 没有那个文件或目录`                                          | 缺少 Python 开发头文件     | 用 conda 环境 (自带) 或 `apt-get install python3-dev`                                 |
| `fatal error: cmatrices.h: 没有那个文件或目录`                                       | sdist 缺少 Cython 头文件 | 从 git 克隆源码而非 pip download; 打补丁加 `_src_dir`                                      |
| `error: command 'gcc' failed with exit code 1`                              | gcc 编译错误            | 检查是否有 C 语法错误,或加 `-v` 看详细错误                                                      |

***

## 快速一键脚本

```bash
#!/bin/bash
set -e

# 1. 创建 conda 环境
conda create -n radiomics -c conda-forge python=3.11 \
    scikit-learn pandas numpy scipy matplotlib seaborn tqdm \
    pydicom simpleitk -y

conda activate radiomics

# 2. 安装 pip 和编译依赖
python -m ensurepip --upgrade
python -m pip install versioneer

# 3. 克隆 git 源码并打补丁
cd /tmp
rm -rf pyradiomics_git
git clone --depth 1 --branch v3.1.0 https://github.com/AIM-Harvard/pyradiomics.git pyradiomics_git

# 修复 setup.py 的 include_dirs
cd pyradiomics_git
sed -i 's|incDirs = \[sysconfig.get_python_inc(), numpy.get_include()\]|import os as _os\n_src_dir = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "radiomics", "src"))\nincDirs = [sysconfig.get_python_inc(), numpy.get_include(), _src_dir]|' setup.py

# 4. 编译安装
python -m pip install --no-build-isolation .

# 5. 安装其余依赖
python -m pip install xgboost lightgbm imbalanced-learn boruta shap lime

echo "安装完成! 运行: conda activate radiomics"
```

***

## 环境切换

```bash
# 激活
conda activate radiomics

# 运行教程 19-20
cd /path/to/ml_template
python src/19_radiomics_feature_extraction.py
python src/20_radiomics_ml_pipeline.py

# 切回主环境
conda deactivate
conda activate ml_template  # 或 source .venv/bin/activate
```

