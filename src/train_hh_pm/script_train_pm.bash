CUDA_VISIBLE_DEVICES="4,5,6,7"
accelerate launch --config_file accelerate_config.yaml src/train_hh_pm/train_pm.py  \
    --output_dir=models \