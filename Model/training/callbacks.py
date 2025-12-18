import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def _is_improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > self.best + self.min_delta
        return metric < self.best - self.min_delta

    def step(self, metric: float) -> bool:
        if self._is_improved(metric):
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter > self.patience


class CheckpointManager:
    def __init__(self, path: str, monitor: str = "val_acc", mode: str = "max") -> None:
        self.path = path
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_improved(self, metric: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return metric > self.best
        return metric < self.best

    def save(self, model: torch.nn.Module, metric: float, epoch: int, metadata: dict | None = None) -> bool:
        if not self._is_improved(metric):
            return False
        self.best = metric
        state = {
            "epoch": epoch,
            "metric": metric,
            "monitor": self.monitor,
            "metadata": metadata or {},
            "model_state": model.state_dict(),
        }
        torch.save(state, self.path)
        return True
