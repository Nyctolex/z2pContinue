from pathlib import Path
import torch
from data import GenericDataset, TrainingStrategy, get_settings_vector, ColorDataset, load_files
import cv2
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu'))
torch.set_default_dtype(torch.float64)

def to_numpy(im, clip=False, sigmiod=False):
    if sigmiod:
        im = torch.sigmoid(im)
    if clip:
        im = im.clamp(0, 1)
    return (im.permute(1, 2, 0)* 255).detach().cpu().numpy()
def generate_color_dataset(source_dataset_path: Path, target_dataset_path: Path, grayscale_model: torch.nn.Module,
                           outline_model: torch.nn.Module, outline_splat_size=3, grayscale_splat_size=3, grayscale_keys=['light_sph_relative'],
                           outline_keys=['light_sph_relative']):

    sample_set = GenericDataset(source_dataset_path, train_strategy = TrainingStrategy.ALL, splat_size=grayscale_splat_size,
                                cache=False, keys=['light_sph_relative'])
    if not source_dataset_path.exists():
        raise FileNotFoundError(f'{source_dataset_path} does not exist')
    if not target_dataset_path.exists():
        target_dataset_path.mkdir(parents=True, exist_ok=True)
    grayscale_model = grayscale_model.to(device)
    outline_model = outline_model.to(device)
    grayscale_model.eval()
    outline_model.eval()

    vector_parse = lambda x: torch.tensor(x, dtype=torch.float64).to(device).unsqueeze(0)
    bar = tqdm(total=len(sample_set))
    start_time = time()
    for i in range(len(sample_set)):
        datapoint = sample_set[i]
        img, _, _, zbuffer, settings_dict, img_path, z_buffer_path = datapoint
        model_name = img_path.parent.parent.name
        view = img_path.parent.name
        z_buffer_index = int(z_buffer_path.name.split('.')[0])
        zbuffer = vector_parse(zbuffer)
        folder = target_dataset_path / model_name / view
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)
        img_path = folder / img_path.name
        gray_img_path = folder / f'gray_img{z_buffer_index}.png'
        outline_img_path = folder / f'outline_img{z_buffer_index}.png'
        settings_path = folder / f'settings.npy'

        if not img_path.exists():
            img = to_numpy(img)
            cv2.imwrite(str(img_path), img)
        if not settings_path.exists():
            np.save(str(settings_path), settings_dict)

        if not gray_img_path.exists():
            settings_vector = get_settings_vector(settings_dict, grayscale_keys)
            settings_vector = vector_parse(settings_vector)
            gray_img = grayscale_model(zbuffer, settings_vector).squeeze(0)
            gray_img = to_numpy(gray_img)
            cv2.imwrite(str(gray_img_path), gray_img)

        if not outline_img_path.exists():
            settings_vector = get_settings_vector(settings_dict, outline_keys)
            settings_vector = vector_parse(settings_vector)
            _, zbuffer = load_files(img_path, z_buffer_path, splat_size=outline_splat_size, cache=False)
            zbuffer = vector_parse(torch.from_numpy(zbuffer).unsqueeze(0))
            outline_img = outline_model(zbuffer, settings_vector).squeeze(0)
            outline_img = torch.sigmoid(outline_img)
            outline_img = to_numpy(outline_img)
            cv2.imwrite(str(outline_img_path), outline_img)
        bar.update(1)
        avg_itter_time = (time() - start_time) / (i+1)
        time_left = (len(sample_set) - i -1)*avg_itter_time
        hours = time_left// 3600
        minutes = (time_left%3600)//60
        bar.set_description(f'Estimated time left{hours}h {minutes}m  Progress: {i}/{len(sample_set)}')





if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--grayscale_controls', nargs='+', default=['colors', 'light_sph_relative'])
    parser.add_argument('--outline_controls', nargs='+', default=['colors', 'light_sph_relative'])
    parser.add_argument('--grayscale_pickle_path', type=Path)
    parser.add_argument('--outline_pickle_path', type=Path)
    parser.add_argument('--grayscale_splat_size', type=int)
    parser.add_argument('--outline_splat_size', type=int)

    opts = parser.parse_args()
    grayscale_model = pickle.loads(open(opts.grayscale_pickle_path, 'rb').read()).get_model()
    outline_model = pickle.loads(open(opts.outline_pickle_path, 'rb').read()).get_model()

    generate_color_dataset(opts.data, target_dataset_path=opts.export_dir, grayscale_model=grayscale_model,
        outline_model=outline_model, grayscale_splat_size = opts.grayscale_splat_size,
                           outline_splat_size= opts.outline_splat_size, grayscale_keys = opts.grayscale_controls,
                            outline_keys = opts.outline_controls)
    ds = ColorDataset(opts.data_dir)
    #print(len(ds))
    #for x in ds:
    #    pass
