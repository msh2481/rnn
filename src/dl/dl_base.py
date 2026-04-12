from abc import abstractmethod
from collections.abc import Callable
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from beartype import beartype
from einops import rearrange, reduce, repeat
from jaxtyping import Float, jaxtyped
from numpy import ndarray as ND
from torch import Tensor as TT
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.utils import DataPoint, TEST_FILE, TRAIN_FILE


def typed(fn):
    return jaxtyped(fn, typechecker=beartype)


class SequenceDataset(Dataset):
    @typed
    def __init__(self, source: str | pd.DataFrame | None = None):
        if source is None:
            source = TRAIN_FILE
        if isinstance(source, str):
            source = pd.read_parquet(source)

        meta_cols = 3  # seq_ix, step_in_seq, need_prediction
        values = source.values
        seq_ids = values[:, 0].astype(int)
        states = values[:, meta_cols:].astype(np.float32)

        unique_seqs = np.unique(seq_ids)
        n_seq = len(unique_seqs)
        seq_len = len(seq_ids) // n_seq
        D = states.shape[1]

        data = states.reshape(n_seq, seq_len, D)
        self.data: Float[TT, "N T D"] = torch.from_numpy(data)

        need_pred = values[:, 2].astype(bool).reshape(n_seq, seq_len)
        self.scored_mask: Float[TT, "N T"] = torch.from_numpy(
            need_pred[:, :-1].astype(np.float32)
        )

    def __len__(self) -> int:
        return self.data.shape[0]

    @typed
    def __getitem__(self, idx: int) -> tuple[Float[TT, "T D"], Float[TT, "T_1"]]:
        return self.data[idx], self.scored_mask[idx]


class DLBase(nn.Module):
    """Base class for deep-learning sequence predictors.

    Subclasses must implement:
        forward(x)       — (B, T, D) -> (B, T, D)  full-sequence, training
        predict_raw(x_t) — (D,) -> (D,)             single-step, inference
        reset_state()    — clear hidden state for a new sequence

    D = mimo * n_features during both training and inference.
    """

    @beartype
    def __init__(
        self,
        *,
        name: str = "unknown",
        n_epochs: int,
        batch_size: int,
        mimo: int,
        optimizer_fn: Callable[..., torch.optim.Optimizer],
        scheduler_fn: Callable[..., torch.optim.lr_scheduler.LRScheduler] | None,
    ):
        super().__init__()
        self.name = name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mimo = mimo
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self._current_seq_ix: int | None = None

    def _get_config(self) -> dict:
        skip = {"_current_seq_ix", "_h"}
        config = {}
        for k, v in self.__dict__.items():
            if k.startswith("_") and k in skip:
                continue
            if isinstance(v, nn.Module):
                continue
            config[k] = repr(v) if callable(v) else v
        return config

    @abstractmethod
    @typed
    def forward(self, x: Float[TT, "B T D"]) -> Float[TT, "B T D"]: ...

    @abstractmethod
    @typed
    def predict_raw(self, x_t: Float[TT, "D"]) -> Float[TT, "D"]: ...

    @abstractmethod
    def reset_state(self) -> None: ...

    def fit(
        self,
        dataset: str | pd.DataFrame | None = None,
        val_dataset: str | pd.DataFrame | None = TEST_FILE,
        show_progress: bool = True,
        wandb_project: str = "rnn_challenge",
    ):
        ds = SequenceDataset(dataset)
        val_ds = SequenceDataset(val_dataset)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        device = next(self.parameters()).device
        optimizer = self.optimizer_fn(self.parameters())
        scheduler = self.scheduler_fn(optimizer) if self.scheduler_fn else None

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        run = wandb.init(
            project=wandb_project,
            name=f"{self.name}_{timestamp}",
            config=self._get_config(),
        )
        step = 0

        self.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            it = (
                tqdm(loader, desc=f"Epoch {epoch+1}/{self.n_epochs}")
                if show_progress
                else loader
            )

            for seqs, scored in it:
                seqs, scored = seqs.to(device), scored.to(device)
                seqs, scored = self._apply_mimo(seqs, scored)

                inputs = seqs[:, :-1]
                targets = seqs[:, 1:]
                preds = self.forward(inputs)

                diff = (preds - targets) ** 2
                mask = scored.unsqueeze(-1)
                loss = (diff * mask).sum() / mask.sum().clamp(min=1) / preds.shape[-1]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1
                run.log({"train/batch_loss": loss.item()}, step=step)

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            val_metrics = self._eval_metrics(val_ds, device)
            run.log(
                {
                    "train/epoch_loss": avg_loss,
                    "val/loss": val_metrics["val/loss"],
                    "val/r2": val_metrics["val/r2"],
                    "epoch": epoch,
                },
                step=step,
            )

            if show_progress:
                print(
                    f"  train_loss={avg_loss:.6f}"
                    f"  val_loss={val_metrics['val/loss']:.6f}"
                    f"  val_r2={val_metrics['val/r2']:.6f}"
                )

        self.eval()
        run.finish()

    @torch.no_grad()
    def _eval_metrics(self, val_ds: SequenceDataset, device: torch.device) -> dict:
        was_training = self.training
        self.eval()

        loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_targets = []
        all_masks = []

        for seqs, scored in loader:
            seqs, scored = seqs.to(device), scored.to(device)
            seqs, scored = self._apply_mimo(seqs, scored)

            inputs = seqs[:, :-1]
            targets = seqs[:, 1:]
            preds = self.forward(inputs)

            mask = scored.unsqueeze(-1)
            all_preds.append(preds)
            all_targets.append(targets)
            all_masks.append(mask)

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        mask = torch.cat(all_masks, dim=0)

        # MSE on scored steps
        diff = (preds - targets) ** 2
        mse = (diff * mask).sum() / mask.sum().clamp(min=1) / preds.shape[-1]

        # R² per feature, then mean
        # flatten batch and time: (B*T, D), mask: (B*T, 1)
        preds_flat = preds.reshape(-1, preds.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
        mask_flat = mask.reshape(-1, 1)

        n = mask_flat.sum()
        target_mean = (targets_flat * mask_flat).sum(dim=0) / n.clamp(min=1)
        ss_res = ((targets_flat - preds_flat) ** 2 * mask_flat).sum(dim=0)
        ss_tot = ((targets_flat - target_mean) ** 2 * mask_flat).sum(dim=0)
        r2_per_feature = 1 - ss_res / ss_tot.clamp(min=1e-8)
        mean_r2 = r2_per_feature.mean().item()

        if was_training:
            self.train()

        return {"val/loss": mse.item(), "val/r2": mean_r2}

    @typed
    def _apply_mimo(
        self,
        seqs: Float[TT, "B T D"],
        scored: Float[TT, "B T_1"],
    ) -> tuple[Float[TT, "Bm T Dm"], Float[TT, "Bm T_1"]]:
        M = self.mimo
        if M <= 1:
            return seqs, scored
        seqs = rearrange(seqs, "(Bm M) T D -> Bm T (M D)", M=M)
        scored = rearrange(scored, "(Bm M) T -> Bm M T", M=M)
        scored = scored.min(dim=1).values
        return seqs, scored

    @beartype
    def predict(self, data_point: DataPoint) -> ND:
        if self._current_seq_ix != data_point.seq_ix:
            self._current_seq_ix = data_point.seq_ix
            self.reset_state()

        x_t = torch.from_numpy(data_point.state.astype(np.float32))
        device = next(self.parameters()).device
        x_t = x_t.to(device)

        with torch.no_grad():
            x_mimo = repeat(x_t, "D -> (M D)", M=self.mimo)
            pred_mimo = self.predict_raw(x_mimo)
            pred = reduce(pred_mimo, "(M D) -> D", "mean", M=self.mimo)

        return pred.cpu().numpy()
