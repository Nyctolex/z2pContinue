# Arguments
# 1: train data dir
# 2: export dir
# 3: test data dir
# 4: checkpoint dir

python train.py --data $1 \
                --export_dir $2 \
                --test_data $3 \
                --checkpoint_dir $4 \
                --checkpoint_every 2000 \
                --train_strategy GRAYSCALE \
                --session_name grayscale \
                --nof_layers 3 \
                --style_enc_layers 6 \
                --start_channels 32 \
                --batch_size 4 \
                --test_batch_size 500 \
                --num_workers 4 \
                --log_iter 1000 \
                --epochs 6 \
                --lr 3e-4 \
                --lr_decay 0.5 \
                --lr_decay_loss_thd 0.98 \
                --lr_decay_iteration_cnt 5000 \
                --losses mse \
                --l_weight 1 \
                --splat_size 3 \
                --controls colors light_sph_relative
                
# It is recommented to run afterwards an additional epoch with the same parameters but add "--gradient_descent" and change lr to the last lr that the train process ran with. Also add "--load_weights_path" with the path to your last epoch.