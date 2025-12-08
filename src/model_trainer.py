import torch
from torch.utils.data import DataLoader, Dataset
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from share.gesture_model import AbstractGestureModel
from dataclasses import dataclass
from collections import Counter
import logging


@dataclass
class TrainingConfig:
    name: str
    weight: list[float] | None = None
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 10


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GestureModelTrainer:
    def __init__(
        self,
        output_dir: str,
        model: AbstractGestureModel,
        dataset: TensorDataset,
        test_size: float,
        configs: list[TrainingConfig],
    ):
        self.output_dir = output_dir
        self.logger = logging.getLogger("gesture_model_trainer")

        self.model = model
        self.configs = configs
        self.is_test = test_size > 0

        if self.is_test:
            train_ds, val_ds, test_ds = self._stratified_split(
                dataset, test_size=test_size
            )
            self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
            self.test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        else:
            train_ds, val_ds = self._stratified_split_no_test(dataset)
            self.train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            self.val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.logger.info(f"Using device: {self.device}")

    def run_all(self):
        for config in self.configs:
            self.run_config(config)

    def run_config(self, config: TrainingConfig):
        self.logger.info(f"Start training with config: {config.name}")
        self.logger.debug(f"Config details: {config}")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss(
            weight=(
                torch.tensor(config.weight).to(self.device) if config.weight else None
            )
        )

        best_val_loss = float("inf")
        count = 0
        start_time = time.time()

        for epoch in range(config.max_epochs):
            train_loss, train_acc = self._train(optimizer, criterion)
            val_loss, val_acc = self._validate(criterion)

            self.logger.info(
                f"[Epoch {epoch+1}/{config.max_epochs}]"
                f" Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    f"{self.output_dir}/best_model_{config.name}.pth",
                )
                count = 0
            else:
                count += 1
                if count >= config.early_stopping_patience:
                    self.logger.info(f"Early stopped.")
                    break

        self.logger.info(
            f"Training completed in {time.time() - start_time:.2f} seconds."
        )

        # test the best model
        if self.is_test:
            self.model.load_state_dict(
                torch.load(f"{self.output_dir}/best_model_{config.name}.pth")
            )
            test_result = self._test(criterion)

            self.logger.debug(f"\n---- Test result of model {config.name} ----")
            self.logger.debug(
                f"Loss: {test_result['loss']:.4f} Acc: {test_result['acc']:.4f}"
            )
            self.logger.debug("Confusion Matrix:")
            self.logger.debug(f"\n{test_result['cmat']}")
            self.logger.debug("Classification Report:")
            self.logger.debug(f"\n{test_result['class_report']}")

    def _train(self, optimizer, criterion):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            out = self.model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            pred = out.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total += X.size(0)

        return total_loss / total, total_correct / total

    def _validate(self, criterion):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total = 0

        with torch.no_grad():
            true_y = []
            pred_y = []
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)

                out = self.model(X)
                loss = criterion(out, y)

                total_loss += loss.item() * X.size(0)
                pred = out.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total += X.size(0)

                true_y.extend(y.cpu().numpy())
                pred_y.extend(pred.cpu().numpy())

        return total_loss / total, total_correct / total

    def _test(self, criterion):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total = 0

        with torch.no_grad():
            true_y = []
            pred_y = []
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                out = self.model(X)
                loss = criterion(out, y)

                total_loss += loss.item() * X.size(0)
                pred = out.argmax(dim=1)
                total_correct += (pred == y).sum().item()
                total += X.size(0)

                true_y.extend(y.cpu().numpy())
                pred_y.extend(pred.cpu().numpy())

            return {
                "loss": total_loss / total,
                "acc": total_correct / total,
                "cmat": confusion_matrix(true_y, pred_y),
                "class_report": classification_report(true_y, pred_y),
            }

    def _stratified_split(
        self, dataset: TensorDataset, val_size=0.2, test_size=0.2
    ) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
        X = dataset.X
        y = dataset.y

        N = len(y)
        indices = list(range(N))

        # train + val | test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=y.numpy(), random_state=42
        )

        # train | val
        train_size = 1 - val_size / (1 - test_size)
        y_train_val = y[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=1 - train_size,
            stratify=y_train_val.numpy(),
            random_state=42,
        )

        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        val_dataset = TensorDataset(X[val_idx], y[val_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])

        self.logger.debug("Dataset distribution:")
        self.logger.debug(f"> Train: {Counter(y[train_idx].numpy())}")
        self.logger.debug(f"> Val:   {Counter(y[val_idx].numpy())}")
        self.logger.debug(f"> Test:  {Counter(y[test_idx].numpy())}")

        return train_dataset, val_dataset, test_dataset

    def _stratified_split_no_test(
        self, dataset: TensorDataset, val_size=0.2
    ) -> tuple[TensorDataset, TensorDataset]:
        X = dataset.X
        y = dataset.y

        N = len(y)
        indices = list(range(N))

        # train | val
        train_size = 1 - val_size
        train_idx, val_idx = train_test_split(
            indices,
            test_size=1 - train_size,
            stratify=y.numpy(),
            random_state=42,
        )

        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        val_dataset = TensorDataset(X[val_idx], y[val_idx])

        self.logger.debug("Dataset distribution:")
        self.logger.debug(f"> Train: {Counter(y[train_idx].numpy())}")
        self.logger.debug(f"> Val:   {Counter(y[val_idx].numpy())}")

        return train_dataset, val_dataset


def setup_logging(filepath):
    logger = logging.getLogger("gesture_model_trainer")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(filepath, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger
