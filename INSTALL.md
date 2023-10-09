# Tubelet-contrast Installation

The codebase is mainly built with following libraries:

- Python 3.6 or higher

- [PyTorch](https://pytorch.org/) <br>
  We can successfully reproduce the main results under two settings below:<br>
   CUDA 11.1 + PyTorch 1.8.0 <br>
   CUDA 10.1 + PyTorch 1.6.0 

```bash
# conda env for  pytorch 1.6.0 and cuda10
conda create -n tubelet_torch160_cuda10 python=3.6
conda activate tubelet_torch160_cuda10 
#
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html
python -c "import torch; print('Torch Version: ', torch.__version__)"
python -c "import torch; x = torch.randn(3, 4).cuda()"
#
pip install opencv-python==4.6.0.66
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
python -c "import mmcv; print('MMCV version: ', mmcv.__version__)"
pip install imutils
pip install scipy
#
pip install tensorboard
```



### Note:
- We recommend you to use **`PyTorch > 1.8.0`**.


