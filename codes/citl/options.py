import os
import torch
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        # global settings
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--nepoch', type=int, default=350, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=16, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--pretrain_weights',type=str, 
                            default='./log/Multi_CNNpropCNN/green/models/model_best.pth', 
                            help='path of pretrained_weights')
        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=3e-4, help='initial learning rate')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='1', help='GPUs')
        parser.add_argument('--arch', type=str, default ='CNNpropCNN',  help='archtechture')
        parser.add_argument('--channel', type=str, default ='green',  help='color')
     
        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./log',  help='save dir')
        parser.add_argument('--save_images', action='store_true',default=False)
        parser.add_argument('--env', type=str, default ='_',  help='env')
        parser.add_argument('--checkpoint', type=int, default=100, help='checkpoint')
  
        # args for training
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--train_dir', type=str, default ='/mnt/data/zhenxing/citl/train/red',  help='dir of train data')
        parser.add_argument('--val_dir', type=str, default ='/mnt/data/zhenxing/citl/val/red',  help='dir of train data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        # args for CNNpropCNN
        parser.add_argument('--F_aperture', type=float, default=1, help='Fourier filter size')
        parser.add_argument('--num_downs_slm', type=int, default=4, help='')
        parser.add_argument('--num_feats_slm_min', type=int, default=32, help='')
        parser.add_argument('--shift_direction', type=str, default='0', help='')
        parser.add_argument('--num_feats_slm_max', type=int, default=256, help='')
        parser.add_argument('--num_downs_target', type=int, default=3, help='')
        parser.add_argument('--num_feats_target_min', type=int, default=16, help='')
        parser.add_argument('--num_feats_target_max', type=int, default=128, help='')
        parser.add_argument('--norm', type=str, default='instance', help='normalization layer')
        parser.add_argument('--slm_latent_amp', action='store_true', default=False, help='If True, '
                                                                              'param amplitdues multiplied at SLM')
        parser.add_argument('--slm_latent_phase', action='store_true', default=False, help='If True, '
                                                                                'parameterize phase added at SLM')

        return parser
