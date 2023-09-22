export TORCH_HOME=./$TORCH_HOME
python exps/dair-v2x/bev_prompt_lss_r101_864_1536_256x256.py --amp_backend native -b 2 --gpus 8
python exps/dair-v2x/bev_prompt_lss_r101_864_1536_256x256.py --ckpt outputs/bev_prompt_lss_r101_864_1536_256x256/checkpoints/ -e -b 1 --gpus 8
