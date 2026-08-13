# 2025-12-30
 python controlRadio_train_vae.py --sd_version sd21 --dataset RadioMapSeer_RadioDiff --simulation Seer --batch_size 6 \
 --carsInput 'yes' --learning_rate 1e-5 --max_epochs 100 --gpus 0,1,2,3,4,5,6,7 --seed 1230

 python controlRadio_train.py --sd_version sd21 --dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
 --batch_size 3 --learning_rate 5e-5 --sd_locked False --max_epochs 50 --vae_locked True --vae_weight 10 --gpus 0,1,2,3,4,5,6,7 \
 --seed 1230 --carsInput 'yes' --vae_resume_path  ./experiments/vae_ft/RadioMapSeer-Seer-no-carsInput/control_sd21_3ch_1e-05_6_100_seed1230/lightning_logs/version_0/checkpoints/epoch\=99-step\=168799.ckpt

python controlRadio_train.py --sd_version sd21 --dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 --gpus 0,1,2,3,4,5,6,7 \
--seed 1230 --carsInput 'yes'  --resume_epochs 50  --trained_learning_rate 5e-5

 # 2025-12-30 test
python controlRadio_test.py --sd_version sd21 --dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 --gpus 0,1,2,3,4,5,6,7 \
--seed 1230 --carsInput 'yes' --test_batch_size 32 --test_simulation DPM
python controlRadio_test.py --sd_version sd21 --dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 --gpus 0,1,2,3,4,5,6,7 \
--seed 1230 --carsInput 'yes'  --test_batch_size 32 --test_simulation DPM --means -0.1 --vars 0.01
