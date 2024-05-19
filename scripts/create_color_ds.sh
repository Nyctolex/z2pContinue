# Arguments
# 1: The path to the dataset
# 2: The path to the export directory
# 3: The path to grayscale model ( .pkl) file
# 4: The path to outlining model ( .pkl) file

python generate_color_ds.py --data $1\
    --export_dir $2 \
    --grayscale_controls colors light_sph_relative  \
    --outline_controls colors light_sph_relative \
    --grayscale_pickle_path $3 \
    --outline_pickle_path  $4 \
    --grayscale_splat_size 3 \
    --outline_splat_size 1