#TODO: change color to color_pretrain
#TODO add color regular

from pathlib import Path
import torch
from data import GenericDataset, TrainingStrategy, get_settings_vector, ColorDataset
import cv2
import numpy as np
import argparse
import pickle

device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu'))

def generate_color_dataset(source_dataset_path: Path, target_dataset_path: Path, grayscale_model: torch.nn.Module,
                           outline_model: torch.nn.Module, splat_size=3, grayscale_keys=['light_sph_relative'],
                           outline_keys=['light_sph_relative']):

    sample_set = GenericDataset(source_dataset_path, train_strategy = TrainingStrategy.ALL, splat_size=splat_size,
                                cache=False, keys=['light_sph_relative'])
    if not source_dataset_path.exists():
        raise FileNotFoundError(f'{source_dataset_path} does not exist')
    if not target_dataset_path.exists():
        target_dataset_path.mkdir(parents=True, exist_ok=True)
    grayscale_model = grayscale_model.to(device)
    outline_model = outline_model.to(device)
    grayscale_model.eval()
    outline_model.eval()
    to_numpy = lambda im: (im.permute(1, 2, 0).clip(0, 1) * 255).detach().cpu().numpy()
    vector_parse = lambda x: torch.tensor(x, dtype=torch.float32).to(device).unsqueeze(0)
    #TODO len(sample_set)
    for i in range(20):
        datapoint = sample_set[i]
        _, img, _, _, zbuffer, settings_dict = datapoint
        zbuffer = vector_parse(zbuffer)
        folder = target_dataset_path / str(i)
        folder.mkdir(parents=True, exist_ok=True)
        img_path = folder / 'img.png'
        gray_img_path = folder / 'gray_img.png'
        outline_img_path = folder / 'outline_img.png'
        settings_path = folder / 'settings.npy'

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
            outline_img = outline_model(zbuffer, settings_vector).squeeze(0)
            outline_img = to_numpy(outline_img)
            cv2.imwrite(str(outline_img_path), outline_img)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--grayscale_controls', nargs='+', default=['colors', 'light_sph_relative'])
    parser.add_argument('--outline_controls', nargs='+', default=['colors', 'light_sph_relative'])
    parser.add_argument('--grayscale_pickle_path', type=Path)
    parser.add_argument('--outline_pickle_path', type=Path)
    parser.add_argument('--splat_size', type=int)

    opts = parser.parse_args()
    grayscale_model = pickle.loads(open(opts.grayscale_pickle_path, 'rb').read()).get_model()
    outline_model = pickle.loads(open(opts.outline_pickle_path, 'rb').read()).get_model()

    generate_color_dataset(opts.data, target_dataset_path=opts.export_dir, grayscale_model=grayscale_model,
        outline_model=outline_model, splat_size = opts.splat_size, grayscale_keys = opts.grayscale_controls,
        outline_keys = opts.outline_controls)