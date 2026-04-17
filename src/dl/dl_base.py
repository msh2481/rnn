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

from src.dl.asgd import NTASGD
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
        grad_clip: float,
        asgd_patience: int | None = None,
        input_noise: float = 0.0,
        aux_horizons: tuple[int, ...] = (),
        aux_weight: float = 0.1,
        mixup_alpha: float = 0.0,
    ):
        super().__init__()
        self.name = name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.mimo = mimo
        self.asgd_patience = asgd_patience
        self.input_noise = input_noise
        self.aux_horizons = aux_horizons
        self.aux_weight = aux_weight
        self.mixup_alpha = mixup_alpha
        self._aux_heads = nn.ModuleDict()
        self.optimizer_fn = optimizer_fn
        self.scheduler_fn = scheduler_fn
        self.grad_clip = grad_clip
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

    def forward_hidden(self, x: Float[TT, "B T D"]) -> Float[TT, "B T H"]:
        """Return pre-readout hidden states. Override in subclasses that support aux predictions."""
        raise NotImplementedError("Model must implement forward_hidden() for aux predictions")

    def _init_aux_heads(self, hidden_dim: int, output_dim: int):
        """Call from subclass __init__ after readout is created."""
        for k in self.aux_horizons:
            self._aux_heads[str(k)] = nn.Linear(hidden_dim, output_dim)

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
        loader = DataLoader(ds, batch_size=self.batch_size * self.mimo, shuffle=True, drop_last=self.mimo > 1)
        device = next(self.parameters()).device
        optimizer = self.optimizer_fn(self.parameters())
        scheduler = self.scheduler_fn(optimizer) if self.scheduler_fn else None
        # averaging: use optimizer's built-in if it has avg_step, else NTASGD
        if hasattr(optimizer, "avg_step"):
            averager = optimizer
        elif self.asgd_patience:
            averager = NTASGD(self, patience=self.asgd_patience)
        else:
            averager = None


        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        run = wandb.init(
            project=wandb_project,
            name=f"{self.name}_{timestamp}",
            config=self._get_config(),
        )
        step = 0

        self.train()
        best_r2 = -float("inf")
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

                if self.mixup_alpha > 0:
                    lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample()
                    lam = max(lam.item(), 1 - lam.item())
                    perm = torch.randperm(seqs.shape[0], device=device)
                    seqs = lam * seqs + (1 - lam) * seqs[perm]
                    scored = torch.min(scored, scored[perm])

                max_k = max(self.aux_horizons) if self.aux_horizons else 1
                inputs = seqs[:, :-max_k]
                targets = seqs[:, 1:seqs.shape[1] - max_k + 1]
                scored_slice = scored[:, :targets.shape[1]]
                if self.input_noise > 0:
                    inputs = inputs + self.input_noise * torch.randn_like(inputs)

                if self.aux_horizons:
                    hidden = self.forward_hidden(inputs)
                    preds = self.readout(hidden)
                else:
                    preds = self.forward(inputs)

                diff = (preds - targets) ** 2
                mask = scored_slice.unsqueeze(-1)
                loss = (diff * mask).sum() / mask.sum().clamp(min=1) / preds.shape[-1]

                # auxiliary multi-horizon losses
                for k in self.aux_horizons:
                    aux_targets = seqs[:, k:seqs.shape[1] - max_k + k]
                    aux_preds = self._aux_heads[str(k)](hidden)
                    aux_diff = (aux_preds - aux_targets) ** 2
                    aux_loss = (aux_diff * mask).sum() / mask.sum().clamp(min=1) / preds.shape[-1]
                    loss = loss + self.aux_weight * aux_loss

                optimizer.zero_grad()
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                step += 1
                run.log({"train/batch_loss": loss.item()}, step=step)

            if scheduler is not None:
                scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            val_metrics = self._eval_metrics(val_ds, device)
            log_dict = {
                "train/epoch_loss": avg_loss,
                "val/loss": val_metrics["val/loss"],
                "val/r2": val_metrics["val/r2"],
                "epoch": epoch,
            }

            if averager is not None:
                if epoch >= self.n_epochs * 3 // 4:
                    averager.force_trigger()
                averager.avg_step(val_metrics["val/loss"])
                if averager.triggered:
                    with averager.averaged():
                        avg_metrics = self._eval_metrics(val_ds, device)
                    log_dict["val/r2_avg"] = avg_metrics["val/r2"]
                    log_dict["val/loss_avg"] = avg_metrics["val/loss"]

            run.log(log_dict, step=step)

            # track best r2 across epochs (prefer averaged when available)
            epoch_r2 = log_dict.get("val/r2_avg", log_dict["val/r2"])
            if epoch_r2 > best_r2:
                best_r2 = epoch_r2

            if show_progress:
                asgd_status = ""
                if averager is not None and averager.triggered:
                    asgd_status = f"  r2_avg={log_dict['val/r2_avg']:.6f}  n_avg={averager.n_averaged}"
                print(
                    f"  train_loss={avg_loss:.6f}"
                    f"  val_loss={val_metrics['val/loss']:.6f}"
                    f"  val_r2={val_metrics['val/r2']:.6f}"
                    f"{asgd_status}"
                )

        if averager is not None:
            averager.swap_in_averaged()
        self.eval()
        run.finish()
        return best_r2

    @torch.no_grad()
    def _eval_metrics(self, val_ds: SequenceDataset, device: torch.device) -> dict:
        was_training = self.training
        self.eval()

        loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_targets = []
        all_masks = []

        M = self.mimo
        for seqs, scored in loader:
            seqs, scored = seqs.to(device), scored.to(device)

            if M > 1:
                # repeat each sequence M times to match inference behavior
                seqs = repeat(seqs, "B T D -> B T (M D)", M=M)

            inputs = seqs[:, :-1]
            targets_raw = seqs[:, 1:]
            preds_raw = self.forward(inputs)

            if M > 1:
                # average M copies, compare against original targets
                preds = reduce(preds_raw, "B T (M D) -> B T D", "mean", M=M)
                targets = reduce(targets_raw, "B T (M D) -> B T D", "mean", M=M)
            else:
                preds = preds_raw
                targets = targets_raw

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
