### Training Parameters
Using the caption.txt as captions and 50 images in artbench as images to train:
| Dataset Size (Images) | Training Steps |
|----------------------:|---------------:|
| 2                     | 2,500          |
| 4                     | 3,500          |
| 6                     | 5,000          |
| 10                    | 7,500          |
| 20                    | 9,500          |
| 50                    | 12,500         |

fixed parameters:
resolution = 512, lr=1e-4, batch_size = 1, lora_rank = 4 when size < 10 and 8 when size > 10, lora_alpha = lora_rank * 2, target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.0,
