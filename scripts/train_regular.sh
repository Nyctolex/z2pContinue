# Arguments
# 1: train data dir
# 2: test data dir
# 3: export dir
#python train.py --data /home/jupyter/data/color_ds/train \
#                --load_weights_path /home/jupyter/logs/grayscale_train5/epoch:0.pt \
#                --gradient_descent \
#                --scan_for_used_data \
#                --weight_decay 0.001 \
#                --splat_size 1 \

python train.py --data /home/jupyter/data/colour_ds2/train \
                --export_dir /home/jupyter/logs/color_train12 \
                --test_data /home/jupyter/data/colour_ds2/test \
                --checkpoint_dir /home/jupyter/logs/checkpoints \
                --checkpoint_every 2000 \
                --train_strategy COLOR \
                --session_name color_train12 \
                --nof_layers 3 \
                --style_enc_layers 6 \
                --start_channels 64 \
                --batch_size 4 \
                --test_batch_size 500 \
                --num_workers 4 \
                --log_iter 1000 \
                --epochs 8 \
                --lr 1.5e-4 \
                --lr_decay 0.5 \
                --lr_decay_loss_thd 0.96 \
                --lr_decay_iteration_cnt 10000 \
                --losses mse intensity \
                --l_weight 1 0.7 \
                --controls colors light_sph_relative
                