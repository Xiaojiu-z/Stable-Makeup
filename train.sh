export MODEL_DIR="../../model_zoo/sd_model_v1-5"  # your SD path
export OUTPUT_DIR="./train_lr5e-5_1wface_with_mask"  # your save path

if [ -n "$QS_LOG_DIR" ] && [ -n "$TRIAL_NAME" ]; then
        LOG_PATH="$QS_LOG_DIR/$TRIAL_NAME"
else
         LOG_PATH="./logs"
fi
echo "LOG_PATH is "${LOG_PATH}

cd  /mnt/public02/usr/zhangyuxuan1/projects/stable_makeup   # cd to your path

accelerate launch train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --train_data_dir "data.jsonl" \
    --resolution=512 \
    --learning_rate=5e-5 \
    --num_validation=2 \
    --validation_ids './test_img/id/1.png' './test_img/id/2.png' './test_img/id/3.png' './test_img/id/4.png' \
    --validation_makeups  './test_img/makeup/1.png' './test_img/makeup/2.png' './test_img/makeup/3.png' './test_img/makeup/4.png' \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --num_train_epochs=1000 \
    --validation_steps=10000 \
    --checkpointing_steps=10000
