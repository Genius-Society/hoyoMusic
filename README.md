# hoyoMusic
[![license](https://img.shields.io/github/license/Genius-Society/hoyoMusic.svg)](./LICENSE)
[![Python application](https://github.com/Genius-Society/hoyoMusic/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/Genius-Society/hoyoMusic/actions/workflows/python-app.yml)
[![hf](https://img.shields.io/badge/modelscope-hoyoMusic-624aff.svg)](https://www.modelscope.cn/collections/hoyoMusic-6f952dae15c04e)
[![ms](https://img.shields.io/badge/huggingface-hoyoMusic-ffd21e.svg)](https://huggingface.co/collections/Genius-Society/hoyomusic-67e5e73b886f80b6f54d7d24)

miHoYo game style music generation

## Environment
```bash
conda create -n hoyo python=3.11 -y
conda activate hoyo
pip install -r requirements.txt
```

## Code download
```bash
git clone git@github.com:Genius-Society/hoyoMusic.git
cd hoyoMusic
```

## Train
```bash
python train.py # fine-tune model
python plot.py # plot training result
```

## Thanks
- [HoYoverse](https://www.hoyoverse.com/en-us/about-us)
- [Tunesformer](https://github.com/sander-wood/tunesformer)