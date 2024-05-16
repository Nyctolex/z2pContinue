import argparse
from inspect import getmembers, isfunction
from pathlib import Path
from time import time
import cv2 as cv
import loguru
import torch
from torch.utils.data import DataLoader
from checkpoint_utils import CheckpointHandler
import data as data
import losses
import util
from models import PosADANet
from util import TrainingStrategy
from loguru import logger
import numpy as np

losses_funcs = {}
for val in getmembers(losses):
    if isfunction(val[1]):
        losses_funcs[val[0]] = val[1]


def log_images(path, msg, img_tensor, style):
    img_tensor = util.embed_color(img_tensor, style[:, :3])
    white_img = util.embed_background(img_tensor)
    for i, img in enumerate(white_img):
        img = img.permute(1, 2, 0)
        img = img.clip(0, 1) * 255
        img = img.detach().cpu().numpy()
        cv.imwrite(f'{str(path)}/{msg}_{i}.png', img)


def get_loss_function(lname):
    return losses_funcs[lname]


class Trainer:
    def __init__(self, opts: argparse.Namespace):
        self.opts = opts
        logger.debug('Starting trainer')
        export_dir = opts.export_dir
        self.export_dir = export_dir
        export_dir.mkdir(exist_ok=True, parents=True)
        self.train_export_dir: Path = export_dir / 'train'
        self.train_export_dir.mkdir(exist_ok=True)
        self. test_export_dir: Path = export_dir / 'test'
        self.test_export_dir.mkdir(exist_ok=True)
        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else
            torch.device("mps") if torch.backends.mps.is_available() else torch.device('cpu'))
        logger.debug(f'using {self.device}')
        self.train_strategy = opts.train_strategy

        # loading data
        self.train_loader = None
        self.test_loader = None
        self.num_samples = None
        self.control_vector_length = None
        self.load_data()

        # loading model
        self.model = None
        self.optimizer = None
        self.start_epoch = None
        self.checkpoint_handler = None
        self.load_checkpoint()


    def load_data(self):
        opts = self.opts
        if self.train_strategy != TrainingStrategy.COLOR:
            train_set = data.GenericDataset(opts.data, splat_size=opts.splat_size, cache=opts.cache,
                                            keys=opts.controls, train_strategy=self.train_strategy)
            test_set = data.GenericDataset(opts.test_data, splat_size=opts.splat_size, cache=opts.cache,
                                           keys=opts.controls, train_strategy=self.train_strategy)
        else:
            train_set = data.ColorDataset(opts.data, keys=opts.controls)
            test_set = data.ColorDataset(opts.test_data, keys=opts.controls)
        control_vector_length = train_set.control_length()
        logger.debug('loading test set')

        test_elements = opts.batch_size * opts.test_batch_size
        test_set = torch.utils.data.random_split(test_set, [test_elements, len(test_set) - test_elements])[0]

        self.train_loader = DataLoader(train_set, batch_size=opts.batch_size,
                                  shuffle=True, num_workers=opts.num_workers, pin_memory=True)
        self.test_loader = DataLoader(test_set, batch_size=opts.batch_size,
                                 shuffle=True, num_workers=opts.num_workers, pin_memory=True)

        self.num_samples = len(self.train_loader)
        self.control_vector_length = control_vector_length

    def get_new_model(self):
        opts = self.opts
        if self.train_strategy == TrainingStrategy.OUTLINE or self.train_strategy == TrainingStrategy.GRAYSCALE:
            input_channels = 1
            output_channels = 1
        elif self.train_strategy == TrainingStrategy.COLOR:
            input_channels = 2
            output_channels = 4
        else:
            raise NotImplementedError('Training strategy not implemented')
        model = PosADANet(input_channels=input_channels, output_channels=output_channels, n_style=self.control_vector_length,
                          padding=opts.padding, bilinear=not opts.trans_conv,
                          nfreq=opts.nfreq, magnitude=opts.freq_magnitude, nof_layers = opts.nof_layers,
                          style_enc_layers = opts.style_enc_layers, start_channels = opts.start_channels
                          ).to(self.device)
        return model

    def load_checkpoint(self):
        opts = self.opts
        if opts.checkpoint_dir is not None and \
                (checkpoint_handler := CheckpointHandler.load_checkpoint(opts.checkpoint_dir, opts.session_name)) is not None:
            logger.debug('loaded checkpoint')
            model = checkpoint_handler.get_model().to(self.device)
            optimizer = checkpoint_handler.get_optimizer()
            start_epoch = checkpoint_handler.get_current_epoch()
            # free allocated memory
            checkpoint_handler.free_space()
            checkpoint_handler._look_for_used_data = True
            if not opts.scan_for_used_data:
                checkpoint_handler._look_for_used_data = False

        else:
            model = self.get_new_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
            checkpoint_handler = CheckpointHandler(opts.checkpoint_dir, session_name=opts.session_name)
            start_epoch = 0
        if opts.load_weights_path is not None:
            model.load_state_dict(torch.load(opts.load_weights_path, map_location="cpu"))
        if opts.gradient_descent:
            optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
        self.optimizer = optimizer
        self.model = model
        self.start_epoch = start_epoch
        self.checkpoint_handler = checkpoint_handler



    def start_train(self):
        self.model.train()
        self.checkpoint_handler.train()
        if self.checkpoint_handler.iteration == 0:
            self.checkpoint_handler.save_lr(self.opts.lr)


    def parse_data(self, data: tuple[str|torch.Tensor]) -> ((
            tuple)[str, torch.Tensor, torch.Tensor, torch.Tensor] |
            tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
            tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None):
        '''
        :param data: a tuple of (img_paths, img, outline, gray_scale, z_buffer, settings_vector)
        :return: None if all points are seen. Otherwise, it would return
        image_paths, input tensors, label tensor
        '''
        device = self.device
        img_paths = data[0]
        unseen = self.checkpoint_handler.get_unseen_points(img_paths)
        all_seen = not any(unseen)
        parse_tensor = lambda x: x[unseen].float().to(device)
        if all_seen:
            return None
        if self.train_strategy == TrainingStrategy.OUTLINE:
            img_paths, _, outline, z_buffer, settings_vector = data
            outline = parse_tensor(outline)
            z_buffer = parse_tensor(z_buffer)
            settings_vector = parse_tensor(settings_vector)
            return img_paths, z_buffer, settings_vector, outline
        elif self.train_strategy == TrainingStrategy.GRAYSCALE:
            img_paths, _, gray_scale, z_buffer, settings_vector = data
            gray_scale = parse_tensor(gray_scale)
            z_buffer = parse_tensor(z_buffer)
            settings_vector = parse_tensor(settings_vector)
            return img_paths, z_buffer, settings_vector, gray_scale
        elif self.train_strategy == TrainingStrategy.COLOR:
            img_path, img, gray_img, outline_img , settings_vector = data
            img = parse_tensor(img)
            gray_img = parse_tensor(gray_img)
            outline_img = parse_tensor(outline_img)
            settings_vector = parse_tensor(settings_vector)
            return img_path, img, gray_img, outline_img , settings_vector

        else:
            raise NotImplementedError('Training strategy not implemented')


    def predict(self, data: tuple[str|torch.Tensor]):
        """
        :param data: The relevent data to the training strategy
        :return: prediction tensor, label tensor
        """
        if self.train_strategy == TrainingStrategy.OUTLINE:
            img_paths, zbuffer, settings_vector, outline = data
            return self.model(zbuffer, settings_vector), outline
        elif self.train_strategy == TrainingStrategy.GRAYSCALE:
            img_paths, zbuffer, settings_vector, gray_scale = data
            return self.model(zbuffer, settings_vector), gray_scale
        elif self.train_strategy == TrainingStrategy.COLOR:
            _, img, gray_img, outline_img , settings_vector = data
            input_data = torch.cat([gray_img, outline_img], dim=1)
            return self.model(input_data, settings_vector), img
        else:
            raise NotImplementedError('Training strategy not implemented')

    def calc_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = torch.tensor(0.0, device=self.device)
        for weight, lname in zip(self.opts.l_weight, self.opts.losses):
            loss += weight * get_loss_function(lname)(prediction, target)
        return loss

    @staticmethod
    def expand_dimensions(one_channel_image: torch.Tensor) -> torch.Tensor:
        one_channel_image_buffer = one_channel_image.repeat((1, 4, 1, 1))
        one_channel_image_buffer[:, -1, :, :] = 1.0
        return one_channel_image_buffer.clamp(0, 1)
    def log_images(self, data: tuple, prediction: torch.Tensor, iteration: int, export_dir: Path):

        if self.train_strategy == TrainingStrategy.OUTLINE:
            prediction = torch.sigmoid(prediction)
            prediction = self.expand_dimensions(prediction)
            _, zbuffer, settings_vector, outline = data
            outline = self.expand_dimensions(outline)
            zbuffer = self.expand_dimensions(zbuffer)
            cat_img = torch.cat([outline, prediction, zbuffer], dim=2)


        elif self.train_strategy == TrainingStrategy.GRAYSCALE:
            prediction = self.expand_dimensions(prediction)
            _, zbuffer, settings_vector, gray_scale = data
            gray_scale = self.expand_dimensions(gray_scale)
            zbuffer = self.expand_dimensions(zbuffer)
            cat_img = torch.cat([gray_scale, prediction, zbuffer], dim=2)
        elif self.train_strategy == TrainingStrategy.COLOR:
            _, img, gray_img, outline_img , settings_vector = data
            outline_img = self.expand_dimensions(outline_img)
            gray_img = self.expand_dimensions(gray_img)
            cat_img = torch.cat([outline_img, gray_img, prediction, img], dim=2)
        else:
            raise NotImplementedError('Training strategy not implemented')
        log_images(export_dir, f'train_imgs{iteration}', cat_img.detach(), settings_vector)


    def train_epoch(self, epoch: int):
        sum_train_time = 0
        train_times_count = 0
        modulo_batch_size = 0
        for i, data  in enumerate(self.train_loader):
            self.model.train()
            logger.debug(f'Epoch {epoch}   Iteration {i}/len{len(self.train_loader)}')
            start_time = time()
            data = self.parse_data(data)
            if data is None:
                continue

            self.optimizer.zero_grad()
            prediction, target = self.predict(data)
            loss: torch.Tensor = self.calc_loss(prediction, target)

            if loss != 0:
                loss.backward()
                self.optimizer.step()

            if i % self.opts.log_iter == 0:
                self.log_images(data, prediction, i, self.train_export_dir)

            self.checkpoint_handler.add_train_loss(loss.item())
            img_paths = data[0]
            self.checkpoint_handler.add_seen_paths(img_paths)
            logger.debug(f'epoch: {epoch}; iter: {i}/{self.num_samples} loss: {loss}')
            sum_train_time += time() - start_time
            train_times_count += 1
            if i % self.opts.checkpoint_every == self.opts.checkpoint_every-1:
                logger.debug('saving checkpoint')
                logger.debug(f'average train time: {sum_train_time / train_times_count} seconds per iteration')
                self.checkpoint_handler.save_checkpoint(self.model, self.optimizer)

            if self.checkpoint_handler.iteration % self.opts.lr_decay_iteration_cnt == self.opts.lr_decay_iteration_cnt-1:
                self.test()
                if (self.checkpoint_handler.iteration > self.opts.lr_decay_iteration_cnt):
                    relevant_history_map = np.array(self.checkpoint_handler.test_history_marker) <= (self.checkpoint_handler.iteration - self.opts.lr_decay_iteration_cnt)
                    if self.checkpoint_handler.test_loss_history[-1] > np.array(self.checkpoint_handler.test_loss_history)[relevant_history_map][-1]*self.opts.lr_decay_loss_thd:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] *= self.opts.lr_decay
                            self.opts.lr = param_group['lr']
                        logger.debug(f'decreasing learning rate to {self.opts.lr}')
                        self.checkpoint_handler.save_lr(self.opts.lr)
                        self.opts.lr_decay_loss_thd = 1 - (1 - self.opts.lr_decay_loss_thd)*self.opts.lr_decay
                self.checkpoint_handler.plot_loss(save_path=str(self.export_dir)+'/loss.png')
            curr_batch_size = len(img_paths)
            if modulo_batch_size + curr_batch_size >= self.opts.batch_size:
                self.checkpoint_handler.iteration += 1
            modulo_batch_size = (modulo_batch_size + curr_batch_size) % self.opts.batch_size
            
            
                    

    def test(self):
        if self.checkpoint_handler.last_test_iteration > self.checkpoint_handler.iteration-5:
            return
        self.model.eval()
        self.checkpoint_handler.test()
        self.checkpoint_handler.save_checkpoint(self.model, self.optimizer)
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                data = self.parse_data(data)
                if data is None:
                    continue
                prediction, target = self.predict(data)
                test_loss = self.calc_loss(prediction, target)
                self.checkpoint_handler.add_test_loss(test_loss.item())
                self.log_images(data, prediction, i, self.test_export_dir)
            self.checkpoint_handler.save_average_loss()

        logger.opt(colors=True).debug(f'<magenta>average train loss: {self.checkpoint_handler.train_loss_history[-1]} </magenta>')
        logger.opt(colors=True).debug(f'<magenta>average test loss: {self.checkpoint_handler.test_loss_history[-1]}</magenta>')

        self.checkpoint_handler.last_test_iteration = self.checkpoint_handler.iteration


    def train(self):
        for epoch in range(self.start_epoch, self.opts.epochs):
            self.start_train()
            self.train_epoch(epoch)
            self.test()
            self.checkpoint_handler.end_epoch()
            self.checkpoint_handler.save_checkpoint(self.model, self.optimizer)
            torch.save(self.model.state_dict(), self.opts.export_dir / f'epoch:{epoch}.pt')


def train(opts: argparse.Namespace):
    trainer = Trainer(opts)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--test_data', type=Path)
    parser.add_argument('--checkpoint_dir', type=Path, default=None)
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='how often to save checkpoints (in iterations, not epochs)')
    parser.add_argument('--train_strategy', type=TrainingStrategy, choices=list(TrainingStrategy))
    parser.add_argument('--session_name', type=str, default='default')

    parser.add_argument('--load_weights_path', type=str, default=None)
    parser.add_argument('--scan_for_used_data', default=False, action="store_true")
    parser.add_argument('--nof_layers', type=int, default=4)
    parser.add_argument('--style_enc_layers', type=int, default=6)
    parser.add_argument('--start_channels', type=int, default=64)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--test_batch_size', type=int, default=75)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--nfreq', type=int, default=20)
    parser.add_argument('--freq_magnitude', type=int, default=10)
    parser.add_argument('--log_iter', type=int, default=1000)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--lr_decay_loss_thd', type=float, default=0.98)
    parser.add_argument('--lr_decay_iteration_cnt', type=float, default=4000)
    parser.add_argument('--losses', nargs='+', default=['mse', 'intensity', 'color_SSIM'])
    parser.add_argument('--l_weight', nargs='+', default=[1, 1, 0.5], type=float)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gradient_descent', default=False, action="store_true")
    parser.add_argument('--tb', action='store_true')
    parser.add_argument('--padding', default='zeros', type=str)
    parser.add_argument('--trans_conv', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--splat_size', type=int)
    parser.add_argument('--controls', nargs='+', default=['colors', 'light_sph_relative'])
    opts = parser.parse_args()
    logger_path =  opts.export_dir/ (opts.session_name +".log")
    logger.remove()
    print(f'logging to {logger_path}')
    logger.add(logger_path, level="DEBUG", diagnose=True)
    for arg in vars(opts):
        logger.debug(f'{arg}, {getattr(opts, arg)}')

    train(opts)