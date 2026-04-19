TASK_INDEX=3
python3 scripts/train_reward_classifier.py \
  --dataset_repo_or_path /public/home/chenyuyao1/.cache/huggingface/lerobot/ybpy/libero_pistar_rc \
  --task_index ${TASK_INDEX} \
  --sample_mode all_steps \
  --epochs 40 \
  --batch_size 64 \
  --lr 1e-4 \
  --weight_decay 1e-6 \
  --image_size 256 \
  --threshold 0.75 \
  --val_ratio 0.1 \
  --augment true \
  --balanced_sampling true \
  --num_workers 4 \
  --save_dir /public/home/chenyuyao1/code/pistar/checkpoints/reward_classifier \
  --backbone_ckpt_path /public/home/chenyuyao1/model/torchvision/resnet18-f37072fd.pth \
  --wandb_enabled true \
  --wandb_project pistar_rc \
  --wandb_run_name rc_task_${TASK_INDEX}
