import time
from lightning.pytorch.callbacks import Callback


class EpochTimerCallback(Callback):
    def __init__(self):
        self._epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._epoch_start is None:
            return
        elapsed = time.perf_counter() - self._epoch_start
        pl_module.log("train/epoch_seconds", elapsed, on_epoch=True, sync_dist=False)
