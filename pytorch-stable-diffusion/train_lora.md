python sd/train_lora.py \
  --image_dir data/img \
  --captions_file data/captions.txt \
  --ckpt_path path/to/your/base_model.ckpt \
  --output_path lora_weights.pt \
  --batch_size 4 \
  --epochs 10