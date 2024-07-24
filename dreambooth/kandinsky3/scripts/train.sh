#!/bin/bash
#SBATCH --job-name=train                     # Название эксперимента
#SBATCH --error=/home/mdnikolaev/aschumachenko/diffusers/examples/dreambooth/kandinsky3/scripts/runs/train%j.err             # Файл для вывода ошибок 
#SBATCH --output=/home/mdnikolaev/aschumachenko/diffusers/examples/dreambooth/kandinsky3/scripts/runs/train%j.log            # Файл для вывода результатов
#SBATCH --gpus=1                             # Количество запрашиваемых гпу
#SBATCH --cpus-per-task=1                    # Выполнение расчёта на 8 ядрах CPU
#SBATCH --time=12:00:00                      # Максимальное время выполнения, после его окончания програмаа просто сбрасывается
#SBATCH --constraint="type_e"

source deactivate
source activate aschumachenko
wandb login 04971049ca813b25cd6db3d781313e4ea63ffd0f

concept=${1}            # dog6
superclass=${2}         # dog
placeholder_token=${3}  # sks

export MODEL_NAME="/home/mdnikolaev/aschumachenko/diffusers/kandinsky-3"
export INSTANCE_DIR="/home/mdnikolaev/aschumachenko/diffusers/datasets/dataset/${concept}"
export OUTPUT_DIR="/home/mdnikolaev/aschumachenko/diffusers/examples/dreambooth/kandinsky3/res_DB/0-res-${concept}_DB"
export CLASS_DIR="/home/mdnikolaev/aschumachenko/diffusers/examples/dreambooth/kandinsky3/reg_kandinsky/${superclass}_class_dir"

accelerate launch train_dreambooth_lora_kandinsky3.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --train_data_dir=$INSTANCE_DIR \
  --test_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of a ${placeholder_token} ${superclass}" \
  --class_data_dir=$CLASSinference_DIR \
  --placeholder_token=${placeholder_token} \
  --class_name=${superclass} \
  --class_prompt="a photo of a ${superclass}" \
  --exp_name="0-res-${concept}_DB" \
  --report_to="wandb" \
  --variant="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-4 \
  --adam_weight_decay=0 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --checkpointing_steps=100 \
  --validation_prompt="a ${placeholder_token} ${superclass} with a city in the background" \
  --validation_steps=100 \
  --num_class_images=100 