# 2025-03-24
conda env create -f environment.yaml
conda activate control

pip install open_clip_torch==2.0.2 pytorch_lightning==1.5.0 omegaconf==2.1.1
export HF_ENDPOINT="https://hf-mirror.com"
pip uninstall torchtext pillow
pip install pillow

# before training the model, you need to download the model from huggingface
cp -r models/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K/ ~/.cache/huggingface/hub/
