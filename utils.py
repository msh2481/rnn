from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tqdm.auto import tqdm


@dataclass
class DataPoint:
    seq_ix: int
    step_in_seq: int
    need_prediction: bool
    #
    state: np.ndarray


class PredictionModel:
    def predict(self, data_point: DataPoint) -> np.ndarray:
        # return current state as dummy prediction
        return data_point.state


class Stack:
    def __init__(self, f, g):
        self.f = f
        self.g = g
        self.f_last_pred = None
        self.current_seq_ix = None

    def predict(self, data_point: DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.f_last_pred = np.zeros_like(data_point.state)
        f_pred = self.f.predict(data_point)
        g_point = DataPoint(
            data_point.seq_ix,
            data_point.step_in_seq,
            data_point.need_prediction,
            data_point.state - self.f_last_pred,
        )
        g_pred = self.g.predict(g_point)
        self.f_last_pred = f_pred
        return f_pred + g_pred
    
    def train(self, dataset, show_progress: bool = True):
        if isinstance(dataset, str):
            dataset = pd.read_parquet(dataset)

        # Train f on original dataset
        self.f.train(dataset, show_progress=show_progress)

        # Build residual dataset for g by replaying f through the data
        rows = tqdm(dataset.values) if show_progress else dataset.values
        residual_rows = []
        f_last_pred = None
        current_seq_ix = None

        for row in rows:
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            new_state = row[3:]

            if current_seq_ix != seq_ix:
                current_seq_ix = seq_ix
                f_last_pred = np.zeros_like(new_state)

            residual_state = new_state - f_last_pred
            residual_rows.append(
                np.concatenate([[seq_ix, step_in_seq, need_prediction], residual_state])
            )

            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, new_state)
            f_last_pred = self.f.predict(data_point)

        residual_df = pd.DataFrame(residual_rows, columns=dataset.columns)
        self.g.train(residual_df, show_progress=show_progress)


class ScorerStepByStep:
    def __init__(self, dataset_path: str):
        self.dataset = pd.read_parquet(dataset_path)

        # Calc feature dimension: first 3 columns are seq_ix, step_in_seq & need_prediction
        self.dim = self.dataset.shape[1] - 3
        self.features = self.dataset.columns[3:]

    def score(self, model: PredictionModel, show_progress: bool = True) -> dict:
        predictions = []
        targets = []

        prev_needed = False
        prev_prediction = None

        rows = tqdm(self.dataset.values) if show_progress else self.dataset.values
        for row in rows:
            seq_ix = row[0]
            step_in_seq = row[1]
            need_prediction = row[2]
            new_state = row[3:]
            #
            if prev_needed:
                predictions.append(prev_prediction)
                targets.append(new_state)
            #
            data_point = DataPoint(seq_ix, step_in_seq, need_prediction, new_state)
            prev_prediction = model.predict(data_point)
            prev_needed = need_prediction

            if prev_prediction.shape[0] != self.dim:
                raise ValueError(
                    f"Prediction has wrong shape: {prev_prediction.shape[0]} != {self.dim}"
                )

        # report metrics
        return self.calc_metrics(np.array(predictions), np.array(targets))

    def calc_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        scores = {}
        for ix_feature, feature in enumerate(self.features):
            scores[feature] = r2_score(
                targets[:, ix_feature], predictions[:, ix_feature]
            )
        scores["mean_r2"] = np.mean(list(scores.values()))
        scores["mse_score"] = 0.98 - np.mean((predictions - targets) ** 2)
        return scores
