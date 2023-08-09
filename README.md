# JSP-GFN

## Installation
To avoid any conflict with your existing Python setup, we suggest to work in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```
Follow these [instructions](https://github.com/google/jax#installation) to install the version of JAX corresponding to your versions of CUDA and CuDNN.
```bash
pip install -r requirements.txt
```

## Example
```bash
python train.py --batch_size 256 --lr 1e-4 --params_num_samples 64 --model lingauss_diag --artifact ...
```
