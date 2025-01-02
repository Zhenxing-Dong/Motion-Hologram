# Motion Hologram
### [Paper](https://arxiv.org/pdf/2401.12537v2) | [arXiv](https://arxiv.org/abs/2401.12537v2) | [Project Page](https://zhenxing-dong.github.io/Motion-Hologram/)

Source code for the Science Advances paper titled "Motion Hologram: Jointly optimized hologram generation and motion planning for photorealistic 3D displays via reinforcement learning"

[Zhenxing Dong](https://zhenxing-dong.github.io/),
[Yuye Ling](http://www.yuyeling.com/),
[Yan Li](),
and [Yikai Su](https://otip.sjtu.edu.cn/en/member/YikaiSu) 

## Focal stack generation
We generate the focal stack from RGB using the [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2) and the [DeepFocus](https://github.com/facebookresearch/DeepFocus) in the paper. We currently also find the mathematical model to synthesize the focal stack in the [Holographic Parallax](https://github.com/dongyeon93/holographic-parallax). We sincerely appreciate the authors for sharing their codes.

## Quick testing: CGH optimization
Replace parameters based on your holographic setup.
```
python ./codes/test_stage2/main.py --channel=0 --data_dir=./data/example_input --hologram_dir=./hologram
```

## Jointly optimization
You can jointly optimize the hologram generation (Other environment) and motion planning (Other values) based on your task.
```
python ./codes/test_stage2/train.py --channel=0 --image_dir=./data/focal_stack --depth_dir=./data/depth
```

## Updating...


## Acknowledgements
The codes are built on [neural-3d-holography](https://github.com/computational-imaging/neural-3d-holography) and [PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch). We sincerely appreciate the authors for sharing their codes.
## Contact
If you have any questions, please do not hesitate to contact [d_zhenxing@sjtu.edu.cn](d_zhenxing@sjtu.edu.cn).
