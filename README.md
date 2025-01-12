# Motion Hologram
### [Paper](https://arxiv.org/pdf/2401.12537v2) | [Project Page](https://zhenxing-dong.github.io/Motion-Hologram/)

Source codes for the Science Advances (Accepted) paper titled "Motion Hologram: Jointly optimized hologram generation and motion planning for photorealistic 3D displays via reinforcement learning"

[Zhenxing Dong](https://zhenxing-dong.github.io/),
[Yuye Ling](http://www.yuyeling.com/),
[Yan Li](),
and [Yikai Su](https://otip.sjtu.edu.cn/en/member/YikaiSu) 

## Focal stack generation
We generate the focal stack from RGB in the paper using the [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2) and the [DeepFocus](https://github.com/facebookresearch/DeepFocus). We currently also find the mathematical model to synthesize the focal stack in the [Holographic Parallax](https://github.com/dongyeon93/holographic-parallax). We sincerely appreciate the authors for sharing their codes.

## CGH optimization
Replace parameters based on your holographic setup.
```
python ./codes/test_stage2/main.py --channel=0 --data_dir=./data/example_input --hologram_dir=./hologram
```

## Jointly optimization
You can jointly optimize the motion hologram generation (or other environment) and motion planning (or other values) based on your task.
```
python ./codes/train_stage1/train.py --channel=0 --image_dir=./data/focal_stack --depth_dir=./data/depth
```
## Camera-in-the-loop
The origin citl codes can be found in [neural-3d-holography](https://github.com/computational-imaging/neural-3d-holography). Here, we also provide citl codes in our paper to calibrate the holographic system. You should replace the system parameters based on your holographic setup. 

### Capturing
```
python ./codes/citl/system_captured/main.py
```
### Calibration
```
python ./codes/citl/system_captured/cali.py
```
### Citl model
```
python ./codes/citl/train.py --arch Multi_CNNpropCNN --train_dir ./data/citl/train_data/red --val_dir ./data/citl/val_data/red
```

## Acknowledgements
The codes are built on [neural-3d-holography](https://github.com/computational-imaging/neural-3d-holography) and [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch). We sincerely appreciate the authors for sharing their codes.
## Contact
If you have any questions, please do not hesitate to contact [d_zhenxing@sjtu.edu.cn](d_zhenxing@sjtu.edu.cn).
