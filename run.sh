CUDA_VISIBLE_DEVICES=5,6,7,0 python train.py \
--model_dir ./tmp \
--sample_dir ./samples \
--exp_name train_pd \
--batch_size 4 \
--learning_rate 0.001 \
--warmup_steps 10 \
--decay_steps 100000 \
--hid_dim 64 \
--num_slots 45 \
--weight_mask 0.5 \
--weight_temporal 0.1 \
--supervision moving

