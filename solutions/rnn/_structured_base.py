import os
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/../..")

from utils import DataPoint, ScorerStepByStep


class StructuredESNBase(ABC):
    def __init__(
        self,
        reservoir_size: int,
        spectral_radius=None,
        leak_rate=None,
        input_scale=None,
        bias_scale=None,
        optimizer=None,
        lr=None,
        weight_decay=None,
        batch_size=None,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)
        self.reservoir_size = reservoir_size

        self.spectral_radius = spectral_radius
        if self.spectral_radius is None:
            self.spectral_radius = self.rng.uniform(0.8, 1.0)

        self.leak_rate = leak_rate
        if self.leak_rate is None:
            self.leak_rate = np.exp(self.rng.uniform(np.log(0.08), np.log(0.16)))

        self.input_scale = input_scale
        if self.input_scale is None:
            self.input_scale = 0.4

        self.bias_scale = bias_scale
        if self.bias_scale is None:
            self.bias_scale = 0.05

        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = "adamw"
        self.optimizer = self.optimizer.lower()
        if self.optimizer not in {"sgd", "adamw"}:
            raise ValueError("optimizer must be one of: 'sgd', 'adamw'")

        self.lr = lr
        if self.lr is None:
            if self.optimizer == "sgd":
                self.lr = np.exp(self.rng.uniform(np.log(1e-4), np.log(5e-2)))
            else:
                self.lr = np.exp(self.rng.uniform(np.log(3e-4), np.log(5e-4)))

        self.weight_decay = weight_decay
        if self.weight_decay is None:
            self.weight_decay = np.exp(self.rng.uniform(np.log(1e-6), np.log(1e-4)))

        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = 128

        self.current_seq_ix = None
        self.state = None
        self.input_dim = None
        self.w_in = None
        self.bias = None
        self.w_out = None
        self.recurrent_scale = 1.0
        self.readout_layer = None
        self.readout_optimizer = None
        self._torch_dtype = torch.float32

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def _variant_repr_fields(self):
        return []

    def _common_repr_fields(self):
        return [
            ("spectral_radius", self.spectral_radius),
            ("leak_rate", self.leak_rate),
            ("input_scale", self.input_scale),
            ("bias_scale", self.bias_scale),
            ("optimizer", repr(self.optimizer)),
            ("lr", self.lr),
            ("weight_decay", self.weight_decay),
            ("batch_size", self.batch_size),
        ]

    def __repr__(self):
        fields = self._variant_repr_fields() + self._common_repr_fields()
        joined = ", ".join(f"{key}={value}" for key, value in fields)
        return f"{self.model_name}({joined})"

    def _ensure_initialized(self, input_dim: int):
        if self.input_dim == input_dim and self.w_in is not None:
            return

        self.input_dim = input_dim
        self.state = np.zeros(self.reservoir_size)
        self.w_out = None
        self.recurrent_scale = 1.0
        self.readout_layer = None
        self.readout_optimizer = None

        self.w_in = self.rng.normal(
            loc=0.0,
            scale=self.input_scale / np.sqrt(max(1, input_dim)),
            size=(self.reservoir_size, input_dim),
        )
        self.bias = self.rng.normal(
            loc=0.0, scale=self.bias_scale, size=self.reservoir_size
        )

        self._build_recurrent_operator()
        self._normalize_recurrent_operator()

    @abstractmethod
    def _build_recurrent_operator(self):
        pass

    @abstractmethod
    def _apply_recurrent_unscaled(self, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _apply_recurrent_transpose_unscaled(self, state: np.ndarray) -> np.ndarray:
        pass

    def _apply_recurrent(self, state: np.ndarray) -> np.ndarray:
        return self.recurrent_scale * self._apply_recurrent_unscaled(state)

    def _apply_recurrent_transpose(self, state: np.ndarray) -> np.ndarray:
        return self.recurrent_scale * self._apply_recurrent_transpose_unscaled(state)

    def _estimate_recurrent_norm(self, num_iters: int = 20) -> float:
        vec = self.rng.normal(size=self.reservoir_size)
        vec_norm = np.linalg.norm(vec)
        if vec_norm == 0:
            return 0.0
        vec = vec / vec_norm

        sigma = 0.0
        for _ in range(num_iters):
            forward = self._apply_recurrent_unscaled(vec)
            sigma = np.linalg.norm(forward)
            if sigma == 0:
                return 0.0

            backward = self._apply_recurrent_transpose_unscaled(forward)
            backward_norm = np.linalg.norm(backward)
            if backward_norm == 0:
                return 0.0
            vec = backward / backward_norm

        return float(sigma)

    def _normalize_recurrent_operator(self):
        current_norm = self._estimate_recurrent_norm()
        if current_norm > 0:
            self.recurrent_scale = self.spectral_radius / current_norm

    def _reset_sequence(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.state = np.zeros(self.reservoir_size)

    def _advance_state(self, data_point: DataPoint):
        self._ensure_initialized(data_point.state.shape[0])

        if self.current_seq_ix != data_point.seq_ix:
            self._reset_sequence(data_point.seq_ix)

        pre_activation = (
            self.w_in @ data_point.state + self._apply_recurrent(self.state) + self.bias
        )
        candidate_state = np.tanh(pre_activation)
        self.state = (
            1 - self.leak_rate
        ) * self.state + self.leak_rate * candidate_state

    def _readout_features(self, state_vector: np.ndarray) -> np.ndarray:
        return np.concatenate([state_vector, self.state, np.array([1.0])])

    def _init_readout(self, feature_dim: int, output_dim: int):
        self.readout_layer = nn.Linear(feature_dim, output_dim, bias=False)
        nn.init.zeros_(self.readout_layer.weight)
        if self.optimizer == "sgd":
            self.readout_optimizer = torch.optim.SGD(
                self.readout_layer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            self.readout_optimizer = torch.optim.AdamW(
                self.readout_layer.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        self.w_out = np.zeros((feature_dim, output_dim), dtype=np.float32)

    def _update_readout_batch(self, features_batch, targets_batch):
        if not features_batch:
            return

        x = torch.tensor(np.asarray(features_batch), dtype=self._torch_dtype)
        y = torch.tensor(np.asarray(targets_batch), dtype=self._torch_dtype)

        self.readout_optimizer.zero_grad()
        predictions = self.readout_layer(x)
        loss = nn.functional.mse_loss(predictions, y)
        loss.backward()
        self.readout_optimizer.step()

    def train(self, dataset_path: str, show_progress: bool = True):
        dataset = pd.read_parquet(dataset_path)
        rows = tqdm(dataset.values) if show_progress else dataset.values

        next_features = None
        batch_features = []
        batch_targets = []

        self.current_seq_ix = None
        self.state = None
        self.w_out = None
        self.readout_layer = None
        self.readout_optimizer = None

        for row in rows:
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            new_state = row[3:]

            if next_features is not None:
                if self.readout_layer is None:
                    self._init_readout(
                        feature_dim=next_features.shape[0],
                        output_dim=new_state.shape[0],
                    )
                batch_features.append(next_features.astype(np.float32, copy=False))
                batch_targets.append(new_state.astype(np.float32, copy=False))
                if len(batch_features) >= self.batch_size:
                    self._update_readout_batch(batch_features, batch_targets)
                    batch_features = []
                    batch_targets = []

            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, new_state)
            self._advance_state(data_point)

            if need_prediction:
                next_features = self._readout_features(new_state)
            else:
                next_features = None

        self._update_readout_batch(batch_features, batch_targets)

        if self.readout_layer is None:
            raise ValueError("No training examples were collected for the ESN readout.")
        self.w_out = self.readout_layer.weight.detach().cpu().numpy().T
        self.current_seq_ix = None
        self.state = np.zeros(self.reservoir_size)
        return self

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.w_out is None:
            raise ValueError(
                f"{self.model_name} readout is not trained. Call .train(dataset_path) first."
            )

        self._advance_state(data_point)

        if not data_point.need_prediction:
            return None

        features = self._readout_features(data_point.state)
        return features @ self.w_out


def evaluate_model(model, label: str):
    train_file = f"{CURRENT_DIR}/../../datasets/train.parquet"
    test_file = f"{CURRENT_DIR}/../../datasets/test.parquet"

    print(model)
    print(f"Training {label} readout...")
    model.train(train_file)
    scorer = ScorerStepByStep(test_file)
    print(f"Testing {label}...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")
    results = scorer.score(model)
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")
    print(f"MSE score: {results['mse_score']:.6f}")
