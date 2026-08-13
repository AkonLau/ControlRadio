
# 2026-01-13 time test
# test for DPM under SRM
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 controlRadio_inference.py --sd_version sd21 \
--dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 \
--seed 1230 --carsInput 'no'  --test_batch_size 24 --building_mask --val test \
--test_simulation DPM --means -0.1 --vars 0.01 --ddim_steps 15

# test for IRT4 under SRM
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 controlRadio_inference.py --sd_version sd21 \
--dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 \
--seed 1230 --carsInput 'no'  --test_batch_size 32 --building_mask --val test \
--test_simulation IRT4 --means -0.1 --vars 0.01 --ddim_steps 15

# test for DPM under DRM
CUDA_VISIBLE_DEVICES=7 python -m torch.distributed.launch --nproc_per_node=1 controlRadio_inference.py --sd_version sd21 \
--dataset RadioMapSeer_RadioDiff --simulation Seer --prompt_type v6 --channel_in 3 \
--batch_size 3 --learning_rate 1e-5 --sd_locked False --max_epochs 100 --vae_locked True --vae_weight 10 \
--seed 1230 --carsInput 'yes'  --test_batch_size 24 --building_mask --val test \
--test_simulation DPM --means -0.1 --vars 0.01 --ddim_steps 15