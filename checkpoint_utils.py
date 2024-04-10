from util import RunningAverage
from pathlib import Path
import pickle


class CheckpointTracker:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir()
        self.datapoint_paths = set()
        self.running_train_loss = RunningAverage()
        self.running_test_loss = RunningAverage()
        self.epoch = 0
        self.train_loss_history = []
        self.test_loss_history = []
        self.training_phase = True
        self.skip_training = False

    def is_training(self):
        return self.training_phase

    def is_testing(self):
        return not self.training_phase

    def train(self):
        self.training_phase = True

    def test(self):
        self.training_phase = False

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
        self.running_train_loss.add(loss)

    def add_test_loss(self, loss: float):
        self.running_test_loss.add(loss)

    def get_avg_test_loss(self):
        return self.running_test_loss.get_average()

    def get_avg_train_loss(self):
        return self.running_train_loss.get_average()

    def save_checkpoint_tracker(self):
        # pickle the object
        open(self.checkpoint_dir / 'checkpoint.pkl', 'wb').write(pickle.dumps(self))

    def save_all(self, model, optimizer):
        # pickle the object
        open(self.checkpoint_dir / 'checkpoint.pkl', 'wb').write(pickle.dumps(self))
        # save the model and optimizer as pickle
        open(self.checkpoint_dir / 'model.pkl', 'wb').write(pickle.dumps(model))
        open(self.checkpoint_dir / 'optimizer.pkl', 'wb').write(pickle.dumps(optimizer))

    @staticmethod
    def load_checkpoint_tracker(checkpoint_dir: str):
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return None
        if not (checkpoint_dir / 'checkpoint.pkl').exists():
            return None
        if not (checkpoint_dir / 'model.pkl').exists():
            return None
        if not (checkpoint_dir / 'optimizer.pkl').exists():
            return None
        # load the object
        handler: CheckpointTracker = pickle.loads(open(checkpoint_dir / 'checkpoint.pkl', 'rb').read())
        if handler.is_testing():
            handler.skip_training = True
        return handler

    def get_model(self):
        return pickle.loads(open(self.checkpoint_dir / 'model.pkl', 'rb').read())

    def get_optimizer(self):
        return pickle.loads(open(self.checkpoint_dir / 'optimizer.pkl', 'rb').read())


    def get_current_epoch(self):
        return self.epoch

    def end_epoch(self):
        self.datapoint_paths = set()
        self.running_train_loss.reset()
        self.running_test_loss.reset()
        self.train_loss_history.append(self.get_avg_train_loss())
        self.test_loss_history.append(self.get_avg_test_loss())
        self.epoch += 1