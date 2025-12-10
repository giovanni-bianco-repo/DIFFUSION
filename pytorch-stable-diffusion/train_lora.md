python sd/train_lora.py \
  --image_dir /home/giovanni_bianco_pro/DIFFUSION/pytorch-stable-diffusion/data/styles/baroque/img \
  --captions_file /home/giovanni_bianco_pro/DIFFUSION/pytorch-stable-diffusion/data/styles/baroque/captions.txt \
  --ckpt_path /home/giovanni_bianco_pro/DIFFUSION/pytorch-stable-diffusion/data/v1-5-pruned.ckpt \
  --output_path lora_weights.pt \
  --batch_size 2 \
  --epochs 4