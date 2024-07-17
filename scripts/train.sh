
#!/bin/sh
VIDEO_DATASET=../video_data/VCDB/distractors/frames
EXPERIMENTS=../experiments/

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 train.py \
          --dataset_path $VIDEO_DATASET \
          --experiment_path $EXPERIMENTS \
          --augmentations GT,FT,TT,ViV \
          --iter_epochs 30000 \
          --warmup_iters 1000 \
          --batch_sz 64 \
          --f2f_sim_module TopKChamfer \
          --v2v_sim_module TopKChamfer \
          --loss_select TBInnerQuadLinearAP,QuadLinearAP,InfoNCE,SSHN \
          --qlap_sigma 0.05 \
          --qlap_rho 0.1 \
          --innerAP_qlap_sigma 0.05 \
          --innerAP_qlap_rho 5 \
          --pseudo_label_top_rate 0.35 \
          --pseudo_label_bottom_rate 0.35 \
          --f_topk_rate 0.1 \
          --v_topk_rate 0.03 \
          --inner_parameter 6 \
          --batch_sz_fe 512 \
          --workers 12 \
          --log_step 100 \
          --eval_step 1500 \
          --window_sz 28 \
          --learning_rate 4e-5 \
          --weight_decay 0.01 \
          --temperature 0.03 \
          --lambda_parameter 3. \
          --r_parameter 1. \
          --use_fp16 true \