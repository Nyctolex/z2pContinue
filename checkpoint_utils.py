from util import RunningAverage
from pathlib import Path
import pickle


class CheckpointHandler:
    def __init__(self, checkpoint_dir: str | None | Path, session_name: str):
        if checkpoint_dir is None:
            self.checkpoint_dir = None
        else:
            assert isinstance(session_name, str)
            assert isinstance(checkpoint_dir, str) or isinstance(checkpoint_dir, Path)

            assert checkpoint_dir != '', 'Checkpoint directory cannot be empty'
            assert session_name != '', 'Session name cannot be empty'
            self.checkpoint_dir = Path(checkpoint_dir)
            if not self.checkpoint_dir.exists():
                self.checkpoint_dir.mkdir()
        # set of all the datapoints seen so far
        self.datapoint_paths = set()
        self._running_train_loss = RunningAverage()
        self._running_test_loss = RunningAverage()
        self._epoch = 0
        self.train_loss_history = []
        self.test_loss_history = []
        self._training_phase = True
        self._skip_training = False
        self._hyperparameters = {}
        self._session_name = session_name
        self._model = None
        self._optimizer = None

    def should_skip_training(self):
        if self._skip_training:
            self._skip_training = False
            return True
        return False
    def set_hyperparameters(self, **hyperparameters):
        assert isinstance(hyperparameters, dict)
        for k, v in hyperparameters.items():
            self._hyperparameters[k] = v

    def get_hyperparameters(self):
        return self._hyperparameters

    def is_training(self):
        return self._training_phase

    def is_testing(self):
        return not self._training_phase

    def train(self):
        self._training_phase = True

    def test(self):
        self._training_phase = False

    def add_seen_path(self, path: str):
        self.datapoint_paths.add(path)

    def add_seen_paths(self, paths: list[str]):
        for p in paths:
            self.add_seen_path(p)

    def already_seen(self, path: str):
        return path in self.datapoint_paths

    def get_unseen_points(self, paths: str) -> list[bool]:
        return [not self.already_seen(p) for p in paths]

    def add_train_loss(self, loss: float):
        self._running_train_loss.add(loss)

    def add_test_loss(self, loss: float):
        self._running_test_loss.add(loss)

    def get_avg_test_loss(self):
        return self._running_test_loss.get_average()

    def get_avg_train_loss(self):
        return self._running_train_loss.get_average()

    def save_checkpoint(self, model, optimizer, free_space=True):
        """
        Save the model, optimizer and the checkpoint tracker
        :param model: The NN model
        :param optimizer: The optimizer
        :param free_space: Should it free the memory after saving
        :return: None
        """
        if self.checkpoint_dir is None:
            return
        self._model = model
        self._optimizer = optimizer
        # pickle the object
        file_name = f'{self._session_name}_checkpoint_epoch_{self._epoch}.pkl'
        file_path = self.checkpoint_dir / file_name
        open(file_path, 'wb').write(pickle.dumps(self))
        if free_space:
            self.free_space()

    @staticmethod
    def load_checkpoint(checkpoint_dir: str, session_name: str, epoch: int | None = None):
        if checkpoint_dir is None:
            raise ValueError('Checkpoint directory is not set')
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None

        if epoch is not None:
            file_name = f'{session_name}_checkpoint_epoch_{epoch}.pkl'
        else:
            files = list(checkpoint_dir.iterdir())
            files = [f for f in files if f.is_file() and f'{session_name}_checkpoint_epoch_' in f.name]
            if len(files) == 0:
                return None
            #get last epoch
            files.sort(key=lambda x: int(x.name.split('_')[-1].split('.')[0]))
            file_name = files[-1].name

        checkpoint_path = checkpoint_dir / file_name

        if not checkpoint_path.exists():
            return None

        # load the object
        handler: CheckpointHandler = pickle.loads(open(checkpoint_path, 'rb').read())
        if handler.is_testing():
            handler._skip_training = True
        return handler

    def get_model(self):
        return self._model

    def delete_model(self):
        self._model = None

    def get_optimizer(self):
        return self._optimizer

    def delete_optimizer(self):
        self._optimizer = None

    def free_space(self):
        self.delete_model()
        self.delete_optimizer()


    def get_current_epoch(self):
        return self._epoch

    def end_epoch(self):
        self.datapoint_paths = set()
        self._running_train_loss.reset()
        self._running_test_loss.reset()
        self.train_loss_history.append(self.get_avg_train_loss())
        self.test_loss_history.append(self.get_avg_test_loss())
        self._epoch += 1