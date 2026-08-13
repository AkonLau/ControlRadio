from share import *
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from controlRadio_dataset import RadioMapDataset, RadioMapSeerDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
from glob import glob

def main(args):
    # set dataset
    if args.dataset == 'RadioMapPCL':
        args.save_dir = f"{args.save_dir}/{args.dataset}/{args.exp_name}"
        dataset = RadioMapDataset
    elif args.dataset == 'RadioMapSeer':
        if args.carsInput=="no":
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.train_simulation}-{args.carsInput}-carsInput/{args.exp_name}"
        else:
            args.save_dir = f"{args.save_dir}/{args.dataset}-{args.train_simulation}-carsInput/{args.exp_name}"
        dataset = RadioMapSeerDataset
    else:
        raise NotImplementedError

    # set seed
    pl.seed_everything(args.seed)

    # set model
    if args.resume_epochs > 0:
        if args.trained_batch_size is None:
            args.trained_batch_size = args.batch_size
        if args.trained_learning_rate is None:
            args.trained_learning_rate = args.learning_rate
        resume_save_dir = args.save_dir.replace(f'{args.learning_rate}_{args.batch_size}_{args.max_epochs}',
                                                f'{args.trained_learning_rate}_{args.trained_batch_size}_{args.resume_epochs}')
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
    #
    # if resume_from_checkpoint is not None:
    #     model.load_state_dict(load_state_dict(resume_from_checkpoint, location='cpu'))
    # else:
    #     model.load_state_dict(load_state_dict(args.resume_path, location='cpu'))

    train_dataset = dataset(args, partition='train')
    val_dataset = dataset(args, partition='test')
    print(len(train_dataset), len(val_dataset))

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=args.batch_size, shuffle=False)

    # log_images_kwargs
    logger = ImageLogger(batch_frequency=args.logger_freq)
    trainer = pl.Trainer(default_root_dir=args.save_dir, gpus=args.gpus, strategy="ddp", precision=32, callbacks=[logger], max_epochs=args.max_epochs)

    # Train!
    if args.resume_epochs > 0:
        trainer.fit(model, train_dataloader, ckpt_path=resume_from_checkpoint)
    else:
        trainer.fit(model, train_dataloader)

    # Evaluate!
    model.eval()
    torch.set_grad_enabled(False)
    trainer.validate(model, dataloaders=val_dataloader)

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
    parser.add_argument('--dataset', type=str, default='RadioMapSeer', choices=['RadioMapPCL', 'RadioMapSeer_RadioDiff', 'RadioMapSeer'])
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--sd_version', type=str, default='sd21', choices=['sd15', 'sd21'])
    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--channel_in', type=int, default=3)
    parser.add_argument('--simulation', type=str, default='DPM', help='simulation type for RadioMapSear dataset: DPM, IRT2, IRT4')
    parser.add_argument('--prompt_type', type=str, default='v4')

    parser.add_argument('--resume_path', type=str, default='./models/control_sd15_ini.ckpt')
    parser.add_argument('--config_path', type=str, default='./models/cldm_v15_vae_mse.yaml')

    parser.add_argument('--learning_rate', type=float, default=5e-5) # 5e-5 is better
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)

    parser.add_argument('--gpus', type=lambda s: [int(item.strip()) for item in s.split(',')], default='0',
                        help='comma delimited of gpu ids to use. Use "-1" for cpu usage')

    parser.add_argument('--logger_freq', type=int, default=1000)

    # using cars image or not
    parser.add_argument('--carsInput', type=str, default='no')

    # for resume
    parser.add_argument('--resume_epochs', type=int, default=0) # for load trained resume model, 0 means not trained yet
    parser.add_argument('--trained_batch_size', type=int, default=None)
    parser.add_argument('--trained_learning_rate', type=float, default=None)

    # for test
    parser.add_argument('--test_batch_size', type=int, default=None)


    parser.add_argument('--train_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')
    parser.add_argument('--test_simulation', type=str, default=None,
                        help='simulation type for RadioMapSear dataset: None, DPM, IRT2, IRT4')



    args = parser.parse_args()

    if args.channel_in == 1:
        args.resume_path = './models/control_sd15_ini_radio.ckpt'
        args.config_path = './models/cldm_v15_radio.yaml'

    if args.train_simulation is None:
        args.train_simulation = args.simulation
    if args.test_simulation is not None:
        args.simulation = args.test_simulation
    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size
        
    args.save_dir = args.save_dir+'/vae_ft'

    args.exp_name = f'control_sd15_{args.channel_in}ch_{args.learning_rate}_{args.batch_size}_{args.max_epochs}_seed{args.seed}'

    if args.sd_version == 'sd21':
        # replace sd15 with sd21
        args.resume_path = args.resume_path.replace('sd15', 'sd21')
        args.config_path = args.config_path.replace('v15', 'v21')
        args.exp_name = args.exp_name.replace('sd15', 'sd21')

    print(args)
    main(args)
