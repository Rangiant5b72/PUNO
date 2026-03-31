````markdown
# PUNO

Official implementation of **PUNO: A Neural Operator Framework for Point Cloud Upsampling**  
(AAAI 2026)

PUNO is a simple and effective neural operator based framework for point cloud upsampling.  
This repository provides the model implementation, pretrained checkpoints, and runnable examples for quick testing and reproduction.

## Highlights

- Neural operator framework for point cloud upsampling
- Clean and lightweight codebase
- Pretrained checkpoints are included in `ckpt/`
- An end-to-end example is provided in `main.py`
- Additional testing examples are provided in `examples/`

Overall, this project is intentionally kept **easy to configure and easy to use**.  
You should be able to get it running with only a few installation steps.

---

## Installation

We recommend creating a clean conda environment with Python 3.8.

```bash
conda create -n puno python=3.8
conda activate puno
````

Install PyTorch and other Python dependencies:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

pip install pytorch3d==0.7.4
pip install open3d
pip install trimesh
pip install point-cloud-utils
```

Then install the PointNet++ operators:

```bash
cd pointnet2_ops_lib
python setup.py install
cd ..
```

---

## Requirements

The main dependencies are:

* Python 3.8
* PyTorch 1.10.0+cu111
* PyTorch3D 0.7.4
* Open3D
* trimesh
* point-cloud-utils

---

## Quick Start

### End-to-end example

A complete end-to-end example is provided in:

```bash
python main.py
```

### More examples

Additional test examples are available in:

```bash
examples/
```

### Pretrained checkpoints

Pretrained model weights are provided in:

```bash
ckpt/
```

With the provided checkpoints and examples, the whole pipeline is **very easy to run and reproduce**.

---

## Repository Structure

```text
PUNO/
├── ckpt/               # pretrained model weights
├── examples/           # testing examples
├── model/              # network definitions
├── pointnet2_ops_lib/  # PointNet++ custom operators
├── main.py             # end-to-end demo
└── README.md
```

---

## Notes

This codebase is designed to be lightweight and practical.
In our experience, the environment is straightforward to configure, and the project is easy to get running once the dependencies are installed.

---

## TODO

* Release the processed ShapeNet upsampling dataset

---

## Acknowledgements

This implementation partially references code from the following excellent projects:

* [GeoUDF](https://github.com/rsy6318/GeoUDF)
* [Super-Resolution-Neural-Operator](https://github.com/2y7c3/Super-Resolution-Neural-Operator)

We sincerely thank the authors for sharing their code.

---

## Citation

If you find this project useful, please consider citing our AAAI 2026 paper:

```bibtex
@inproceedings{xiao2026puno,
  title={PUNO: A Neural Operator Framework for Point Cloud Upsampling},
  author={Xiao, Zijian and Xu, Yining and Huang, Yingjie and Yao, Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={13},
  pages={11042--11050},
  year={2026}
}
```

> Please update the BibTeX entry with the final author list and publication information.