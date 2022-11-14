---
layout: ../layouts/Content.astro
---

# Installation
**Prerequisites:**
- Install Pytorch: https://pytorch.org/TensorRT/tutorials/installation.html
- Install Pytorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

**Install Command:**
```bash
pip install olorenchemengine[full]
```

## Installation FAQ
1. **subprocess.CalledProcessError: Command '['which', 'g++']' returned non-zero exit status 1**
= apt-get install g++
2. “**ImportError: libXrender.so.1: cannot open shared object file: No such file or directory” and/or “
ImportError: libXrender.so.1: cannot open shared object file: No such file or directory**
= apt-get install libxrender1 libxtst6 -y
3. **pytorch==1.11.0 not found**
OCE was built back when PyTorch Geometric installation was significantly jankier, so this propagated into the official OCE installation having very stringent requirements. Nowadays, this isn’t as much of an issue. So try: “conda install pytorch -c pytorch; conda install pyg -c pyg”. PyTorch Geometric conda installation isn’t bulletproof yet so you may have to consult [https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for alternative PyTorch Geometric installation
4. **error: [Errno 2] No such file or directory: 'cmake’**
pip install cmake
5. **distutils.errors.CompileError: command '/usr/bin/gcc' failed with exit code 1**
You will need to install gcc, this is system dependent: some possibilities are `apt-get install gcc` or with Home-brew `brew install gcc`.
6. **M1 Macs: Illegal hardware instruction**

    Basically this boils down to M1/intel hardware/software compatibility. The best way to fix this is to completely reinstall conda ([https://docs.anaconda.com/anaconda/install/uninstall/](https://docs.anaconda.com/anaconda/install/uninstall/), commands different for miniconda, just rm -rf miniconda3, wherever it is). And to reinstall conda, taking special care to install the M1 mac version.

7. ****M1 Macs: lots of xgboost install errors****
Issue #50: Update to the latest version of Oloren ChemEngine, and then run `pip uninstall xgboost` and then `pip install xgboost`. If that isn’t working, follow the instructions here: [https://towardsdatascience.com/install-xgboost-and-lightgbm-on-apple-m1-macs-cb75180a2dda](https://towardsdatascience.com/install-xgboost-and-lightgbm-on-apple-m1-macs-cb75180a2dda).

8. **OSError: libtorch_cuda_cu.so: cannot open shared object file: No such file or directory**:
```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f [https://data.pyg.org/whl/torch-1.11.0+cu113.html](https://data.pyg.org/whl/torch-1.11.0+cu113.html)
```