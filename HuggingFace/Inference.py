"""
Inference for the released SMT-Learner checkpoint.

Loads the fine-tuned (adaptive transfer) SMT-Learner model, embeds spatiotemporal
movement trajectories, and computes the exploration-exploitation (E-E) ratio with
the hyperparameters reported in the paper (W=120, alpha=0.05, beta1=0.10, beta2=0.90).

    from HuggingFace.Inference import SMTLearnerInference

    smt = SMTLearnerInference.from_pretrained("raihan116/SMT-Learner")
    embeddings = smt.embed_trials(paths, time_diff_ms)      # (n_trials, 64)
    ee = smt.exploration_exploitation(embeddings)           # (n_trials,)

The model is small enough for on-device inference: 260,704 parameters,
1.19 MB checkpoint, 14.4 ms per trial and 0.042 ms per incremental E-E update
on a single CPU thread.
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DataProcessing.Normalization import normalize_trajectory_sequence_3d
from SMTLearner.TransformerEncoder import SMTLearnerTransformer

# E-E hyperparameters selected by the sensitivity analysis reported in the paper
EE_WINDOW = 120
EE_ALPHA = 0.05
EE_BETA1 = 0.10
EE_BETA2 = 0.90


class SMTLearnerInference:
    """Trajectory embedding and E-E estimation with a trained SMT-Learner."""

    def __init__(self, model, config, device="cpu"):
        self.model = model.to(device).eval()
        self.config = config
        self.device = device

    # ------------------------------------------------------------------ load
    @classmethod
    def from_local(cls, model_dir, device="cpu"):
        """Load from a directory holding config.json and smt_learner.pt."""
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        model = SMTLearnerTransformer(
            seq_len=config["seq_len"],
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            nhead=config["nhead"],
            num_layers=config["num_layers"],
            dropout=config.get("dropout", 0.1),
            metadata_dim=config.get("metadata_dim", 1),
        )
        state = torch.load(os.path.join(model_dir, "smt_learner.pt"), map_location=device)
        state = state["model_state_dict"] if "model_state_dict" in state else state
        # the released checkpoint stores the positional encoding trimmed to seq_len;
        # the module regenerates the same deterministic buffer at construction
        state = {k: v for k, v in state.items() if not k.endswith("pos_encoder.pe")}
        missing = model.load_state_dict(state, strict=False)
        assert not missing.unexpected_keys, f"unexpected keys in checkpoint: {missing.unexpected_keys}"
        return cls(model, config, device)

    @classmethod
    def from_pretrained(cls, repo_id="raihan116/SMT-Learner", device="cpu", **kwargs):
        """Download the released checkpoint from the Hugging Face Hub."""
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(repo_id=repo_id, **kwargs)
        return cls.from_local(model_dir, device)

    # ------------------------------------------------------------- embedding
    def embed_trials(self, paths, time_diff_ms, task_type=None, batch_size=32):
        """
        Embed a sequence of trials.

        Args:
            paths: list of trajectories, each a sequence of (x, y) points or the
                   string form stored in the dataset CSVs
            time_diff_ms: matching list of per-point timestamps in milliseconds
            task_type: optional binary task-type indicator per trial (metadata
                   channel); defaults to 0
            batch_size: trials per forward pass

        Returns:
            (n_trials, hidden_dim) array of Z_global embeddings. Trials that
            cannot be normalized are returned as rows of NaN.
        """
        normalized, valid = [], []
        for i, (path, t) in enumerate(zip(paths, time_diff_ms)):
            traj = normalize_trajectory_sequence_3d(path, t)
            if traj is not None:
                normalized.append(traj)
                valid.append(i)

        out = np.full((len(paths), self.config["hidden_dim"]), np.nan, dtype=np.float32)
        if not normalized:
            return out

        if task_type is None:
            meta = np.zeros((len(normalized), 1), dtype=np.float32)
        else:
            meta = np.asarray([task_type[i] for i in valid], dtype=np.float32).reshape(-1, 1)

        with torch.no_grad():
            for start in range(0, len(normalized), batch_size):
                x = torch.tensor(np.asarray(normalized[start:start + batch_size]),
                                 dtype=torch.float32, device=self.device)
                m = torch.tensor(meta[start:start + batch_size], device=self.device)
                encoded, _, _ = self.model(x, m)
                out[valid[start:start + batch_size]] = encoded.cpu().numpy()
        return out

    # ---------------------------------------------------------------- E-E
    @staticmethod
    def exploration_exploitation(embeddings, window=EE_WINDOW, alpha=EE_ALPHA,
                                 beta1=EE_BETA1, beta2=EE_BETA2):
        """
        E-E ratio for a chronologically ordered sequence of trial embeddings.

        Exploration(i) = min_{j<i} dist(e_i, e_j) * (beta1 + beta2 * exp(-i*alpha))
        Exploitation(i) = mean_{j in W_i} cos_sim(e_i, e_j)
        E-E ratio      = Exploration(i) / Exploitation(i)

        The first trial has no history and is returned as NaN.
        """
        e = np.asarray(embeddings, dtype=np.float64)
        n = len(e)
        ratio = np.full(n, np.nan)
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        unit = np.divide(e, norms, out=np.zeros_like(e), where=norms > 0)

        for i in range(1, n):
            prior = e[:i]
            if not np.isfinite(prior).all(axis=1).any():
                continue
            dist = np.linalg.norm(prior - e[i], axis=1)
            exploration = np.nanmin(dist) * (beta1 + beta2 * np.exp(-i * alpha))

            lo = max(0, i - window)
            sims = unit[lo:i] @ unit[i]
            exploitation = np.nanmean(sims) if len(sims) else np.nan
            if exploitation and np.isfinite(exploitation) and exploitation != 0:
                ratio[i] = exploration / exploitation
        return ratio


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed trials and compute the E-E ratio.")
    parser.add_argument("--model_dir", required=True, help="Directory with config.json + smt_learner.pt")
    parser.add_argument("--csv", required=True, help="CSV with path and time_diff_ms columns")
    parser.add_argument("--out", default="ee_results.csv")
    args = parser.parse_args()

    import pandas as pd

    df = pd.read_csv(args.csv)
    smt = SMTLearnerInference.from_local(args.model_dir)
    emb = smt.embed_trials(df["path"].tolist(), df["time_diff_ms"].tolist(),
                           task_type=df["task_type"].tolist() if "task_type" in df else None)
    df["ee_ratio"] = smt.exploration_exploitation(emb)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(df)} trials, {np.isfinite(df['ee_ratio']).sum()} with an E-E ratio)")
