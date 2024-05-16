sudo /opt/conda/bin/python /home/jupyter/copy/z2pContinue/generate_color_ds.py --data /home/jupyter/data/renders_shade_abs_test\
    --export_dir /home/jupyter/data/colour_ds2/test \
    --grayscale_controls colors light_sph_relative  \
    --outline_controls light_sph_relative \
    --grayscale_pickle_path /home/jupyter/logs/checkpoints/grayscale_train9_checkpoint_epoch_6.pkl \
    --outline_pickle_path  /home/jupyter/logs/checkpoints/outline_train14_checkpoint_epoch_2.pkl \
    --grayscale_splat_size 3 \
    --outline_splat_size 1