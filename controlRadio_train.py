from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from controlRadio_dataset import RadioMapDataset, RadioMapSeerDataset_RadioDiff, RadioMapSeerDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from glob import glob

def main(args):
    # Misc
    if args.dataset == 'RadioMapPCL':
        args.save_dir = f"{args.save_dir}/{args.dataset}/{args.exp_name}"
        dataset = RadioMapDataset
    elif args.dataset == 'RadioMapSeer_RadioDiff':
        if args.carsInput=="no":
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.simulation}-{args.carsInput}-carsInput/{args.exp_name}"
        else:
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.simulation}-carsInput/{args.exp_name}"
        dataset = RadioMapSeerDataset_RadioDiff
    elif args.dataset == 'RadioMapSeer':
        if args.carsInput=="no":
            args.save_dir = f"{args.save_dir}/ControlRadio_{args.dataset}-{args.simulation}-{args.carsInput}-carsInput/{args.exp_name}"
        else:
            args.save_dir = f"{args.save_dir}/ControlRadio_{args.dataset}-{args.simulation}-carsInput/{args.exp_name}"
        dataset = RadioMapSeerDataset
    else:
        raise NotImplementedError

    # set seed
    pl.seed_everything(args.seed)

    if args.resume_epochs > 0:
        if args.trained_batch_size is None:
            args.trained_batch_size = args.batch_size
        if args.trained_learning_rate is None:
            args.trained_learning_rate = args.learning_rate
        resume_save_dir = args.save_dir.replace(f'{args.learning_rate}_{args.batch_size}_{args.max_epochs}',
                                                f'{args.trained_learning_rate}_{args.trained_batch_size}_{args.resume_epochs}')

        if args.trained_seed is not None:
            resume_save_dir = resume_save_dir.replace(f'seed{args.seed}', f'seed{args.trained_seed}')

        if args.trained_sd_locked is not None:
            if args.trained_sd_locked is True and args.sd_locked is False:
                resume_save_dir = resume_save_dir.replace('sd_tune', 'sd_lock')
                args.save_dir = args.save_dir.replace('sd_tune', 'sd_lock->sd_tune')
            elif args.trained_sd_locked is False and args.sd_locked is True:
                resume_save_dir = resume_save_dir.replace('sd_lock', 'sd_tune')
                args.save_dir = args.save_dir.replace('sd_lock', 'sd_tune->sd_lock')

        if args.trained_vae_locked is not None:
            if args.trained_vae_locked is True and args.vae_locked is False:
                resume_save_dir = resume_save_dir.replace(f'_ft_vae{args.vae_weight}', '')
                args.save_dir = args.save_dir.replace(f'_ft_vae{args.vae_weight}', f'_no_ft_vae->ft_vae{args.vae_weight}')
            elif args.trained_vae_locked is False and args.vae_locked is True:
                resume_save_dir = resume_save_dir + f'_ft_vae{args.vae_weight}'
                args.save_dir = args.save_dir + f'_ft_vae{args.vae_weight}->no_ft_vae'

        # args.max_epochs -= args.resume_epochs
        # dynamic find ckpt files
        ckpt_path = f'{resume_save_dir}/lightning_logs/version_0/checkpoints/*.ckpt'
        ckpt_files = glob(ckpt_path)
        if ckpt_files:
            latest_ckpt = ckpt_files[0]
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_path}")
        # args.resume_path = latest_ckpt
        print(latest_ckpt)
        print(args.save_dir)
        resume_from_checkpoint = latest_ckpt
    else:
        resume_from_checkpoint = None

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config_path).cpu()
    model.learning_rate = args.learning_rate
    model.sd_locked = args.sd_locked
    model.only_mid_control = args.only_mid_control
    model.vae_locked = args.vae_locked
    model.vae_weight = args.vae_weight
    model.physical_loss = args.physical_loss
    model.physical_loss_weight = args.physical_loss_weight
    model.noise_control = args.noise_control
    model.max_epochs = args.max_epochs
    model.means = args.means
    model.vars = args.vars
    
    if resume_from_checkpoint is not None:
        if args.trained_sd_locked is not None or args.trained_vae_locked is not None:
            model.load_state_dict(load_state_dict(resume_from_checkpoint, location='cpu'), strict=False)
        else:
            pass
    else:

        model_state_dict = load_state_dict(args.resume_path, location='cpu')
        if args.vae_resume_path is not None:
            print(f"Loading VAE weight")
            vae_state_dict = load_state_dict(args.vae_resume_path, location='cpu')
            for k in vae_state_dict.keys():
                copy_k = 'first_stage_model.' + k
                if copy_k in model_state_dict.keys():
                    model_state_dict[copy_k] = vae_state_dict[k]
                else:
                    print(f"Skip {k} in vae weight")
        model.load_state_dict(model_state_dict, strict=False)

    train_dataset = dataset(args, partition='train')
    val_dataset = dataset(args, partition='test')
    print(len(train_dataset), len(val_dataset))

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    # log_images_kwargs
    # log_images_kwargs = {"unconditional_guidance_scale": args.ugs}
    # logger = ImageLogger(batch_frequency=args.logger_freq, log_images_kwargs=log_images_kwargs)

    trainer = pl.Trainer(default_root_dir=args.save_dir, gpus=args.gpus, strategy="ddp", precision=32, callbacks=None, max_epochs=args.max_epochs)

    # Train!
    if args.trained_sd_locked is not None or args.trained_vae_locked is not None:
        trainer.fit(model, train_dataloader)
    else:
        trainer.fit(model, train_dataloader, ckpt_path=resume_from_checkpoint)
    #     model.eval()
    #     torch.set_grad_enabled(False)
    #     trainer.test(model, dataloaders=val_dataloader)

def str2bool(value):
    if value.lower() in ('true', '1', 't', 'y', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'f', 'n', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    # Configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='RadioMapPCL', choices=['RadioMapPCL', 'RadioMapSeer_RadioDiff', 'RadioMapSeer'])
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--prompt_type', type=str, default='v4')
    parser.add_argument('--sd_version', type=str, default='sd15', choices=['sd15', 'sd21'])
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--channel_in', type=int, default=3)
    parser.add_argument('--simulation', type=str, default='DPM', help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')

    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--config_path', type=str, default='./models/cldm_v15.yaml')
    parser.add_argument('--vae_resume_path', type=str, default=None)

    parser.add_argument('--learning_rate', type=float, default=5e-5) # 5e-5 is better
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)

    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

    parser.add_argument("--sd_locked", type=str2bool, nargs="?", const=True, default=True, help="sd_locked = True is sd_lock & sd_locked = False is sd_tune")
    parser.add_argument("--only_mid_control", type=str2bool, nargs="?", const=True, default=False)

    # vae_locked
    parser.add_argument("--vae_locked", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--vae_weight", type=float, default=1.0)

    parser.add_argument('--logger_freq', type=int, default=1000000000)

    # physical loss
    parser.add_argument('--physical_loss', type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument('--physical_loss_weight',
        type=float,
        nargs='+',  # 表示可以传入一个或多个值
        default=[1.0, 1.0],
        help='List of weights for physical loss components'
    )

    # noise control
    parser.add_argument('--noise_control', type=str2bool, nargs="?", const=True, default=False)

    # using cars image or not
    parser.add_argument('--carsInput', type=str, default='no')

    # param unconditional_guidance_scale
    parser.add_argument('--ugs', type=float, default=1.0, help="unconditional_guidance_scale")


    # for resume
    parser.add_argument('--resume_epochs', type=int, default=0) # for load trained resume model, 0 means not trained yet
    parser.add_argument('--trained_sd_locked', type=str2bool, default=None)
    parser.add_argument('--trained_vae_locked', type=str2bool, default=None)
    parser.add_argument('--trained_batch_size', type=int, default=None)
    parser.add_argument('--trained_learning_rate', type=float, default=None)
    parser.add_argument('--trained_seed', type=int, default=None)

    parser.add_argument("--means", type=float, default=0.0)
    parser.add_argument("--vars", type=float, default=1.0)    
    args = parser.parse_args()

    if args.channel_in == 1:
        args.resume_path = './models/control_sd15_ini_radio.ckpt'
        args.config_path = './models/cldm_v15_radio.yaml'

    args.save_dir = args.save_dir+'/prompt_'+args.prompt_type

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

    print(args)
    main(args)
