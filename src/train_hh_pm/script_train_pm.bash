accelerate launch --config_file accelerate_config.yaml src/train_hh_pm/train_pm.py  \
    --output_dir=models \
    --per_device_train_batch_size=64 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="none" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --eval_strategy="steps" \
    --eval_steps=500 \
    --max_length=512 \
 