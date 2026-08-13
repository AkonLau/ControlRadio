from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from controlRadio_dataset import RadioMapDataset, RadioMapSeerDataset_RadioDiff, RadioMapSeerDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from glob import glob
import torch.nn as nn

import torch.nn.functional as F
from tqdm import tqdm
import torch
from PIL import Image

import time
import os
import numpy as np
import einops
from einops import rearrange, repeat
import math
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.functional import peak_signal_noise_ratio

def compute_nmse(predicted, ground_truth):
    assert predicted.shape == ground_truth.shape, "Predicted and ground truth must have the same shape."
    numerator = torch.sum((predicted - ground_truth) ** 2, dim=[1, 2, 3])
    denominator = torch.sum(ground_truth ** 2, dim=[1, 2, 3])
    denominator = torch.clamp(denominator, min=1e-6)

    nmse = numerator / denominator
    return nmse.mean()



def main(args):
    # Misc
    if args.dataset == 'RadioMapPCL':
        args.save_dir = f"{args.save_dir}/{args.dataset}/{args.exp_name}"
        dataset = RadioMapDataset
    elif args.dataset == 'RadioMapSeer_RadioDiff':
        if args.carsInput == "no":
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.train_simulation}-{args.carsInput}-carsInput/{args.exp_name}"
        else:
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.train_simulation}-carsInput/{args.exp_name}"
        dataset = RadioMapSeerDataset_RadioDiff
    elif args.dataset == 'RadioMapSeer':
        if args.carsInput == "no":
            args.save_dir = f"{args.save_dir}/ControlRadio_{args.dataset}-{args.train_simulation}-{args.carsInput}-carsInput/{args.exp_name}"
        else:
            args.save_dir = f"{args.save_dir}/ControlRadio_{args.dataset}-{args.train_simulation}-carsInput/{args.exp_name}"
        dataset = RadioMapSeerDataset
    else:
        raise NotImplementedError
    # set seed
    pl.seed_everything(args.seed)

    # 2025-03-05 for two-step fine-tuning
    if args.resume_epochs > 0:
        if args.trained_sd_locked is not None:
            if args.trained_sd_locked is True and args.sd_locked is False:
                args.save_dir = args.save_dir.replace('sd_tune', 'sd_lock->sd_tune')
            elif args.trained_sd_locked is False and args.sd_locked is True:
                args.save_dir = args.save_dir.replace('sd_lock', 'sd_tune->sd_lock')
        if args.trained_vae_locked is not None:
            if args.trained_vae_locked is True and args.vae_locked is False:
                args.save_dir = args.save_dir.replace(f'_ft_vae{args.vae_weight}',
                                                      f'_no_ft_vae->ft_vae{args.vae_weight}')
            elif args.trained_vae_locked is False and args.vae_locked is True:
                args.save_dir = args.save_dir + f'_ft_vae{args.vae_weight}->no_ft_vae'

    # dynamic find ckpt files
    ckpt_path = f'{args.save_dir}/lightning_logs/version_0/checkpoints/*.ckpt'
    ckpt_files = glob(ckpt_path)
    if ckpt_files:
        latest_ckpt = ckpt_files[0]
    else:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_path}")
    args.resume_path = latest_ckpt
    model_state_dict = load_state_dict(args.resume_path, location='cpu')

    if args.vae_resume_path is not None:
        print(f"Loading VAE weight")
        vae_state_dict = load_state_dict(args.vae_resume_path, location='cpu')
        for k in vae_state_dict.keys():
            copy_k = 'first_stage_model.' + k
            if copy_k in model_state_dict.keys():
                model_state_dict[copy_k] = vae_state_dict[k]
            else:
                print(f"Skip {copy_k} in vae weight")

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_path).cpu()
    model.load_state_dict(model_state_dict, strict=False)
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.vae_locked = args.vae_locked
    model.vae_weight = args.vae_weight
    model.physical_loss = args.physical_loss
    model.physical_loss_weight = args.physical_loss_weight
    model.noise_control = args.noise_control
    model.means = args.means
    model.vars = args.vars

    train_dataset = dataset(args, partition='train')
    val_dataset = dataset(args, partition='val')
    test_dataset = dataset(args, partition='test')
    print(len(train_dataset), len(val_dataset), len(test_dataset))
    # if len(args.gpus) > 1:
    #     model = nn.DataParallel(model, device_ids=args.gpus)
    # else:
    #     model = model.cuda(args.gpus[0])

    model = model.cuda(args.local_rank)  # 将模型拷贝到每个gpu上.直接.cuda()也行，因为多进程时每个进程的device号是不一样的
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 设置多个gpu的BN同步
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                         device_ids=[args.local_rank],
                                                         output_device=args.local_rank,
                                                         find_unused_parameters=False,
                                                         broadcast_buffers=False)
    model.eval()    

    print(f"Using {args.val} dataset with {args.ddim_steps} steps \n")

    if args.val == 'val':
        print("Using validation set")
        pred_dataset = val_dataset
    
    elif args.val == 'train':
        print("Using training set")
        pred_dataset = train_dataset
    else:
        print("Using test set")
        pred_dataset = test_dataset
    
    sampler = torch.utils.data.distributed.DistributedSampler(pred_dataset, rank=args.local_rank, shuffle=False, drop_last=False)
    
    
    pred_dataloader = DataLoader(pred_dataset, 
                                 num_workers=16, 
                                 batch_size=args.test_batch_size, 
                                 shuffle=False,
                                         pin_memory=True,
                                         drop_last=False,
                                         sampler=sampler)                
    total_losses = 0
    total_nmse_losses = 0
    total_ssim = 0
    total_psnr = 0
    
    total_num = 0
    
    start_time = time.time()
    for batch in tqdm(pred_dataloader, desc="Predicting"):
    # for i, batch in enumerate(pred_dataloader):
        with torch.no_grad():
            N = len(batch['jpg'])
    
            prompt_embed = model.module.get_learned_conditioning(batch["txt"]).cuda(args.local_rank, non_blocking=True)
            target = batch['jpg'].permute(0, 3, 1, 2).cuda(args.local_rank, non_blocking=True)
            layout_cond = batch["hint"].cuda(args.local_rank, non_blocking=True)
            layout_cond = einops.rearrange(layout_cond, 'b h w c -> b c h w')
            layout_cond = layout_cond.to(memory_format=torch.contiguous_format).float()
    
            print(f"---using random noise with {args.means} means, {args.vars} vars---")
            controlled_noise = torch.randn((N, 4, 32, 32)) * (args.vars ** 0.5) + args.means
            controlled_noise = controlled_noise.cuda(args.local_rank, non_blocking=True)
            # import pdb; pdb.set_trace
            # get denoise row
            samples, z_denoise_row = model.module.sample_log(cond={"c_concat": [layout_cond], "c_crossattn": [prompt_embed]},
                                                     batch_size=N, ddim=True,
                                                     ddim_steps=args.ddim_steps, eta=0.0, log_every_t=5,
                                                     x_T=controlled_noise
                                                     )
            pred = model.module.decode_first_stage(samples) # B x 3 x H x W
            pred = torch.clamp(pred, -1., 1.)
    
            # normalize
            pred_norm = pred * 0.5 + 0.5 # scale to [0, 1]
            target_norm = target * 0.5 + 0.5 # scale to [0, 1]
    
            # building_mask
            if args.building_mask:
                building_mask = batch["mask"].cuda(args.local_rank, non_blocking=True)
                building_mask = einops.rearrange(building_mask, 'b h w c -> b c h w')
                building_mask = building_mask.to(memory_format=torch.contiguous_format).float()
                pred_norm *= building_mask
                target_norm *= building_mask
    
            mse_loss = nn.MSELoss()(pred_norm, target_norm)
            nmse_loss = compute_nmse(pred_norm, target_norm)
            ssim = structural_similarity_index_measure(pred_norm, target_norm)
            psnr = peak_signal_noise_ratio(pred_norm, target_norm)
            
            # reduce mean across all GPUs
            mse_loss = reduce_mean(mse_loss, dist.get_world_size())
            nmse_loss = reduce_mean(nmse_loss, dist.get_world_size())
            ssim = reduce_mean(ssim, dist.get_world_size())
            psnr = reduce_mean(psnr, dist.get_world_size())
            
            # reduce sum across all GPUs
            total_losses += mse_loss * N * dist.get_world_size()
            total_nmse_losses += nmse_loss * N * dist.get_world_size()
            total_ssim += ssim * N * dist.get_world_size()
            total_psnr += psnr * N * dist.get_world_size()
            total_num += N * dist.get_world_size()
    
            if dist.get_rank() == 0:
                print(f"BATCH RMSE: {math.sqrt(mse_loss)}")
                print(f"BATCH NMSE: {nmse_loss}")
                print(f"BATCH SSIM: {ssim}")
                print(f"BATCH PSNR: {psnr}")
    
                print(f"AVG RMSE: {math.sqrt(total_losses/ total_num)}")
                print(f"AVG NMSE: {total_nmse_losses/ total_num}")
                print(f"AVG SSIM: {total_ssim/ total_num}")
                print(f"AVG PSNR: {total_psnr/ total_num}")
    
    
            if args.save_img:
                for i in range(N):
                    save_path = batch['path']
                    save_name = save_path[i].replace("gain/","")
                    if args.val == 'val':
                        path = f"{args.save_dir}/image_log/val/pred_images_{args.means}_{args.vars}/{save_name}"
                    elif args.val == 'train':
                        path = f"{args.save_dir}/image_log/train/train_images_{args.means}_{args.vars}/{save_name}"
    
                    else:
                        path = f"{args.save_dir}/image_log/test/pred_images_{args.means}_{args.vars}/{save_name}"
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
    
                    image = pred[i]
    
                    if isinstance(image, torch.Tensor):
                        image = image.detach().cpu()
                        image = torch.clamp(image, -1., 1.)
    
                    image = (image + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    
                    image = image.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    image = image.numpy()
                    image = (image * 255).astype(np.uint8)
                    Image.fromarray(image).save(path)
    
    if dist.get_rank() == 0:
        print(f"FINAL RMSE: {math.sqrt(total_losses / total_num)}")
        print(f"FINAL NMSE: {total_nmse_losses / total_num}")
        print(f"FINAL SSIM: {total_ssim / total_num}")
        print(f"FINAL PSNR: {total_psnr / total_num}")
        print(f"TOTAl NUMBER: {total_num}")
    
        elapsed = time.time() - start_time
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        t_s = elapsed / total_num
        print(f"单个样本耗时：{t_s} 秒")
        print(f"总耗时: {int(h)} 小时 {int(m)} 分钟 {s:.2f} 秒")
        logs = f"{args.ddim_steps}\t{args.means}\t{args.vars}\t{math.sqrt(total_losses / total_num)}\t" \
               f"{total_nmse_losses / total_num}\t{total_ssim / total_num}\t{total_psnr / total_num}\t" \
               f"单个样本耗时：{t_s} 秒\t" \
               f"总耗时: {int(h)} 小时 {int(m)} 分钟 {s:.2f} 秒\t" \
               f"test_batch_size: {args.test_batch_size}\n"
    
    
        if args.building_mask:
            if args.val == 'val':
                with open(f"{args.save_dir}/{args.test_simulation}_val_rmse_building_mask.txt", "a") as f:
                    f.write(logs)
            elif args.val == 'train':
                with open(f"{args.save_dir}/{args.test_simulation}_train_rmse_building_mask.txt", "a") as f:
                    f.write(logs)
            else:
                with open(f"{args.save_dir}/{args.test_simulation}_test_rmse_building_mask.txt", "a") as f:
                    f.write(logs)
        else:
            if args.val == 'val':
                with open(f"{args.save_dir}/{args.test_simulation}_val_rmse.txt", "a") as f:
                    f.write(logs)
            elif args.val == 'train':
                with open(f"{args.save_dir}/{args.test_simulation}_train_rmse.txt", "a") as f:
                    f.write(logs)
            else:
                with open(f"{args.save_dir}/{args.test_simulation}_test_rmse.txt", "a") as f:
                    f.write(logs)
                    
def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

from torch import distributed as dist

def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

if __name__ == "__main__":

    print(torch.cuda.device_count())  # 打印gpu数量
    torch.distributed.init_process_group(backend="nccl")  # 并行训练初始化，建议'nccl'模式
    print('world_size', torch.distributed.get_world_size())  # 打印当前进程数

    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RadioMapPCL', choices=['RadioMapPCL', 'RadioMapSeer_RadioDiff', 'RadioMapSeer'])
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--prompt_type', type=str, default='v4')

    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'])
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--channel_in', type=int, default=3)
    parser.add_argument('--simulation', type=str, default='DPM',
                        help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')
    parser.add_argument('--train_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')
    parser.add_argument('--test_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')

    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--config_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--vae_resume_path', type=str, default=None)

    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)

    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

    parser.add_argument("--sd_locked", type=str2bool, nargs="?", const=True, default=True,
                        help="sd_locked = True is sd_lock & sd_locked = False is sd_tune")
    parser.add_argument("--only_mid_control", type=str2bool, nargs="?", const=True, default=False)

    parser.add_argument('--logger_freq', type=int, default=1000)

    # vae_locked
    parser.add_argument("--vae_locked", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--vae_weight", type=float, default=1.0)
    # physical loss
    parser.add_argument('--physical_loss', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--physical_loss_weight',
                        type=float,
                        nargs='+',  # 表示可以传入一个或多个值
                        default=[1.0, 1.0],
                        help='List of weights for physical loss components'
                        )

    # param unconditional_guidance_scale
    parser.add_argument('--ugs', type=float, default=1.0, help="unconditional_guidance_scale")
    # sample: true or false, default false
    parser.add_argument('--sample', type=str2bool, nargs="?", const=True, default=False)

    # using cars image or not
    parser.add_argument('--carsInput', type=str, default='no')

    # for resume
    parser.add_argument('--resume_epochs', type=int,
                        default=0)  # for load trained resume model, 0 means not trained yet
    parser.add_argument('--trained_sd_locked', type=str2bool, default=None)
    parser.add_argument('--trained_vae_locked', type=str2bool, default=None)
    parser.add_argument('--trained_batch_size', type=int, default=None)
    parser.add_argument('--trained_learning_rate', type=float, default=None)

    # noise control
    parser.add_argument('--noise_control', type=str2bool, nargs="?", const=True, default=False)

    # for test
    parser.add_argument('--test_batch_size', type=int, default=None)
    parser.add_argument('--trained_seed', type=int, default=None)

    parser.add_argument("--means", type=float, default=0.0)
    parser.add_argument("--vars", type=float, default=1.0)
    # parser.add_argument('--save_dataset_dir', type=str, default='./experiments/time_dataset')
    # parser.add_argument('--frames', type=int, default=150)

    # 下面这个参数需要加上，torch内部调用多进程时，会使用该参数，对每个gpu进程而言，其local_rank都是不同的；
    parser.add_argument('--local_rank', default=-1, type=int)
    # ture for val:
    # parser.add_argument('--val', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--save_img', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--building_mask', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--val', type=str, default='test')

    # ddim_steps
    parser.add_argument('--ddim_steps', type=int, default=50)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)  # 设置gpu编号为local_rank;此句也可能看出local_rank的值是什么

    if args.channel_in == 1:
        args.resume_path = './models/control_sd15_ini_radio.ckpt'
        args.config_path = './models/cldm_v15_radio.yaml'

    if args.train_simulation is None:
        args.train_simulation = args.simulation
    if args.test_simulation is not None:
        args.simulation = args.test_simulation
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    args.save_dir = args.save_dir + '/prompt_' + args.prompt_type

    if args.sd_locked:
        args.exp_name = f'control_sd15_{args.channel_in}ch_{args.learning_rate}_{args.batch_size}_{args.max_epochs}_sd_lock_seed{args.seed}'
    else:
        args.exp_name = f'control_sd15_{args.channel_in}ch_{args.learning_rate}_{args.batch_size}_{args.max_epochs}_sd_tune_seed{args.seed}'
    if not args.vae_locked:
        args.exp_name = args.exp_name + f'_ft_vae{args.vae_weight}'

    if args.physical_loss:
        args.exp_name = args.exp_name + f'_physical_loss_{args.physical_loss_weight[0]}_{args.physical_loss_weight[1]}'

    if args.noise_control:
        args.exp_name = args.exp_name + f'_noise_control'

    if args.sd_version == 'sd21':
        # replace sd15 with sd21
        args.resume_path = args.resume_path.replace('sd15', 'sd21')
        args.config_path = args.config_path.replace('v15', 'v21')
        args.exp_name = args.exp_name.replace('sd15', 'sd21')

    if args.sample is True:
        args.ugs = -1.0
    if args.trained_seed is not None:
        args.exp_name = args.exp_name.replace(f'seed{args.seed}', f'seed{args.trained_seed}')

    print(args)
    main(args)
