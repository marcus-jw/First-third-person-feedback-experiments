export HF_HOME=/nas/ucb/constantinweisser/cache/

for num_run in {1..3}
do for num_perspective in {1..3..2} 
do  perspective="3_"$num_perspective;
accelerate launch --config_file accelerate_config.yaml src/train_hh_pm/train_pm_personalization.py  \
    --output_dir="models/llama_${perspective}_trainonperso-intended_grm_big_10epochs_lr5e-4_${num_run}" \
    --perspective=${perspective} \
    --model_name=Ray2333/GRM-llama3-8B-sftreg \
    --tokenizer_name=sfairXC/FsfairX-LLaMA3-RM-v0.1   \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=16 \
    --num_train_epochs=10 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing=True \
    --learning_rate=5e-4 \
    --report_to="none" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=1 \
    --eval_strategy="steps" \
    --eval_steps=0.25 \
    --max_length=2048 \
    --LoRA=True \
    --LoRA_r=8 \
    --LoRA_alpha=32 \
    --LoRA_dropout=0.1 \
    --fp16=True \
    --lr_scheduler_type="cosine";
done
done  

