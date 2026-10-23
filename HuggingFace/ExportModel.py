"""
Package a trained SMT-Learner checkpoint for https://huggingface.co/raihan116/SMT-Learner

Builds the upload directory (config.json, smt_learner.pt, README.md model card),
verifies that the checkpoint matches the architecture reported in the paper, and
optionally pushes it to the Hub.

    # package only, then inspect before uploading
    python HuggingFace/ExportModel.py \
        --checkpoint saved_models/transfer_learning_results/exp4/exp4_results/adaptive_model.pt \
        --out HuggingFace/upload

    # package and push (needs `huggingface-cli login` or HF_TOKEN)
    python HuggingFace/ExportModel.py --checkpoint <path> --out <dir> --push

The released model is the EXP4 adaptive-transfer checkpoint: the MSL-pretrained
encoder after combined cross-task and cross-subject fine-tuning on MP.
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SMTLearner.TransformerEncoder import SMTLearnerTransformer

REPO_ID = "raihan116/SMT-Learner"

# Architecture reported in Suppl. TABLE I of the revised manuscript
PAPER_CONFIG = {
    "architecture": "SMTLearnerTransformer",
    "seq_len": 512,
    "input_dim": 3,
    "hidden_dim": 64,
    "nhead": 8,
    "num_layers": 2,
    "dropout": 0.1,
    "metadata_dim": 1,
}
PAPER_PARAM_COUNT = 260_704


def load_state_dict(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def count_parameters(state_dict):
    """Trainable parameters, excluding the fixed positional-encoding buffer."""
    return sum(v.numel() for k, v in state_dict.items()
               if hasattr(v, "numel") and not k.endswith("pos_encoder.pe"))


def export(checkpoint, out_dir, strict=True):
    os.makedirs(out_dir, exist_ok=True)
    state = load_state_dict(checkpoint)
    n_params = count_parameters(state)

    model = SMTLearnerTransformer(
        seq_len=PAPER_CONFIG["seq_len"], input_dim=PAPER_CONFIG["input_dim"],
        hidden_dim=PAPER_CONFIG["hidden_dim"], nhead=PAPER_CONFIG["nhead"],
        num_layers=PAPER_CONFIG["num_layers"], dropout=PAPER_CONFIG["dropout"],
        metadata_dim=PAPER_CONFIG["metadata_dim"],
    )
    # The positional encoding is a deterministic buffer whose stored length depends
    # on how the checkpoint was written; it is regenerated at load time, so it is
    # excluded from the compatibility check and rewritten below.
    weights = {k: v for k, v in state.items() if not k.endswith("pos_encoder.pe")}
    try:
        result = model.load_state_dict(weights, strict=False)
        unexpected = list(result.unexpected_keys)
        absent = [k for k in result.missing_keys if not k.endswith("pos_encoder.pe")]
        shape_ok = not unexpected and not absent
        missing = f"unexpected={unexpected} missing={absent}" if not shape_ok else ""
    except RuntimeError as exc:
        shape_ok, missing = False, exc

    if n_params != PAPER_PARAM_COUNT or not shape_ok:
        message = (
            f"Checkpoint does not match the architecture reported in the paper.\n"
            f"  checkpoint : {checkpoint}\n"
            f"  parameters : {n_params:,} (paper reports {PAPER_PARAM_COUNT:,})\n"
            f"  config     : hidden_dim/num_layers must be "
            f"{PAPER_CONFIG['hidden_dim']}/{PAPER_CONFIG['num_layers']}\n"
            f"  detail     : {missing}\n"
            f"Publishing a checkpoint that does not match the reported model would "
            f"make the Hugging Face release inconsistent with the manuscript."
        )
        if strict:
            raise SystemExit("ERROR: " + message)
        print("WARNING: " + message)

    # The positional encoding is a deterministic buffer allocated for max_len=5000
    # at construction but only ever read up to seq_len. Trim it so the released
    # artifact is the 1.19 MB checkpoint reported in the paper; Inference.py
    # regenerates it at load time.
    state = dict(weights)
    state["pos_encoder.pe"] = model.pos_encoder.pe[:, :PAPER_CONFIG["seq_len"]].clone()

    torch.save({"model_state_dict": state}, os.path.join(out_dir, "smt_learner.pt"))
    config = dict(PAPER_CONFIG)
    config["num_parameters"] = n_params
    config["source_checkpoint"] = os.path.basename(checkpoint)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    card = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MODEL_CARD.md")
    shutil.copyfile(card, os.path.join(out_dir, "README.md"))

    size_mb = os.path.getsize(os.path.join(out_dir, "smt_learner.pt")) / 1e6
    print(f"packaged -> {out_dir}  ({n_params:,} parameters, {size_mb:.2f} MB)")
    return out_dir


def push(out_dir, repo_id=REPO_ID):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=out_dir, repo_id=repo_id, repo_type="model")
    print(f"pushed -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package SMT-Learner for the Hugging Face Hub.")
    parser.add_argument("--checkpoint", required=True, help="Trained .pt checkpoint to release")
    parser.add_argument("--out", default="HuggingFace/upload", help="Directory to build")
    parser.add_argument("--push", action="store_true", help="Upload to the Hub after packaging")
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--allow_mismatch", action="store_true",
                        help="Package even if the checkpoint does not match the reported architecture")
    args = parser.parse_args()

    out = export(args.checkpoint, args.out, strict=not args.allow_mismatch)
    if args.push:
        push(out, args.repo_id)
