import os
import json
import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional

from STCRL.TrainSTCRL import train_stcrl_model
from STCRL.TransferLearning.STCRLTransferLearning import STCRLTransferLearningRunner
from STCRL.STCRLDataset import STCRLModelFittingDataset
from STCRL.EvaluationFramework import EvaluationMetrics
from STCRL.TransformerEncoder import STCRLTransformer

############################################################
# Utility Functions
############################################################

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

from DataProcessing.Normalization import normalize_trajectory_sequence_3d

def enforce_canonical_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every row has a canonical normalized trajectory (x,y,t) via normalize_trajectory_sequence_3d.
    Existing normalized_trajectory will be overwritten to guarantee consistency.
    """
    # Factorize participant_id to integers if present
    if 'participant_id' in df.columns:
        df = df.copy()
        df['participant_id'], _ = pd.factorize(df['participant_id'])
    # Preserve and factorize task_type if non-numeric (e.g., monkey dataset with Scott_2001_e1...e5)
    if 'task_type' in df.columns:
        if not np.issubdtype(df['task_type'].dtype, np.number):
            df = df.copy()
            df['task_type_str'] = df['task_type']
            df['task_type'], _ = pd.factorize(df['task_type'])

    # If this is a D2 (monkey) style dataframe with HandPos column, convert it to 'path'
    if 'HandPos' in df.columns:
        df = df.copy()
        def handpos_to_path(handpos_val):
            # Accept already-parsed sequences
            if isinstance(handpos_val, (list, tuple, np.ndarray)):
                arr = np.array(handpos_val)
                # Common MATLAB shape (2, N): x row then y row
                if arr.ndim == 2:
                    if arr.shape[0] == 2 and arr.shape[1] >= 2:
                        return list(zip(arr[0].astype(float), arr[1].astype(float)))
                    if arr.shape[1] == 2:  # (N,2)
                        return [tuple(map(float, pt)) for pt in arr]
                # Fallback: flatten into pairs
                flat = arr.flatten()
                path_pairs = []
                for i in range(0, len(flat) - 1, 2):
                    try:
                        path_pairs.append((float(flat[i]), float(flat[i+1])))
                    except Exception:
                        continue
                return path_pairs
            # String parsing expected like: "[x1 y1; x2 y2; ...]"
            s = str(handpos_val).strip().strip('[]')
            point_strings = [p for p in s.split(';') if p.strip()]
            path = [(0,0)] # Initialize with dummy to avoid empty list
            for pstr in point_strings:
                # Replace commas with spaces, split on whitespace
                coords = [c for c in pstr.replace(',', ' ').split() if c]
                if len(coords) >= 2:
                    try:
                        x = float(coords[0]); y = float(coords[1])
                        path.append((x, y))
                    except Exception:
                        continue
            return path

        if 'path' not in df.columns:
            df['path'] = df['HandPos'].apply(handpos_to_path)
        else:
            # Repair any non-iterable or malformed path entries by re-parsing HandPos
            def ensure_path(row):
                val = row['path']
                if isinstance(val, (list, tuple)) and len(val) > 0:
                    first = val[0]
                    if isinstance(first, (list, tuple)) and len(first) == 2:
                        return val
                return handpos_to_path(row['HandPos'])
            df['path'] = df.apply(ensure_path, axis=1)

    # Derive time_diff_ms from MATLAB-style Time column for D2-like inputs
    if 'Time' in df.columns:
        def matlab_time_to_array(matlab_str):
            try:
                if isinstance(matlab_str, (list, tuple, np.ndarray)):
                    arr = np.array(matlab_str, dtype=float)
                else:
                    s = str(matlab_str).strip().strip('[]')
                    # Split by semicolons primarily; fallback commas/newlines
                    if ';' in s:
                        parts = [p for p in s.split(';') if p != '']
                    else:
                        parts = [p for p in s.replace('\n', ',').split(',') if p != '']
                    arr = np.array([float(val) for val in parts], dtype=float)
                return arr * 1000.0
            except Exception:
                return np.array([])

        df = df.copy()
        df['time_diff_ms'] = df['Time'].apply(matlab_time_to_array)

    if 'path' not in df.columns or 'time_diff_ms' not in df.columns:
        raise ValueError("DataFrame must contain 'path' and 'time_diff_ms' columns for normalization.")
    norm_trajs = []
    for idx, row in df.iterrows():
        try:
            norm = normalize_trajectory_sequence_3d(row['path'], row['time_diff_ms'])
        except Exception:
            norm = np.array([])
        norm_trajs.append(norm)
    df = df.copy()
    df['normalized_trajectory'] = norm_trajs
    return df

def debug_print_norm_stats(df: pd.DataFrame, label: str, max_samples: int = 5):
    total = len(df)
    valid = 0
    empty = 0
    shapes = {}
    samples_empty = []
    for i, v in enumerate(df.get('normalized_trajectory', [])):
        arr = v
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            empty += 1
            if len(samples_empty) < max_samples:
                samples_empty.append(i)
            continue
        valid += 1
        shp = tuple(arr.shape)
        shapes[shp] = shapes.get(shp, 0) + 1
    print(f"[DEBUG] {label}: total={total}, valid={valid}, empty={empty}, shapes={list(shapes.items())[:5]}")
    if samples_empty:
        print(f"[DEBUG] {label}: sample empty idx={samples_empty}")

def filter_unimanual(df: pd.DataFrame) -> pd.DataFrame:
    # task_type encodes unimanual vs bimanual (e.g., 0=unimanual,1=bimanual). Adjust if schema differs.
    return df[df['task_type'] == 0]

def filter_bimanual(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['task_type'] == 1]

def filter_cohort(df: pd.DataFrame, cohort_value: str) -> pd.DataFrame:
    return df[df['Cohort'] == cohort_value]

def holdout_last_session(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Hold out the max session_no for zero-shot evaluation
    if 'session_no' not in df.columns:
        return df, pd.DataFrame([])
    max_session = df['session_no'].max()
    test_df = df[df['session_no'] == max_session]
    train_df = df[df['session_no'] != max_session]
    return train_df, test_df

def compute_zero_shot_pct_diff(zero_shot_loss: float, finetuned_loss: float) -> float:
    if finetuned_loss <= 0: return 0.0
    return ((zero_shot_loss - finetuned_loss) / finetuned_loss) * 100.0

############################################################
# Pretraining
############################################################

def pretrain_model(source_df: pd.DataFrame, save_prefix: str, hidden_dim=128, nhead=8, num_layers=3, epochs=25) -> Tuple[str, Dict]:
    from STCRL.ContrastiveLossFunctions import MultiTemporalLoss
    loss_fn = MultiTemporalLoss()  # full multi-temporal loss
    model, optimizer, history = train_stcrl_model(source_df, loss_fn, hidden_dim=hidden_dim, nhead=nhead, num_layers=num_layers, epochs=epochs, batch_size=32, val_split=0.1, early_stop_patience=5)
    # Save model + architecture for TransferLearningRunner compatibility
    arch = {
        'seq_len': 512,
        'input_dim': 3,
        'hidden_dim': hidden_dim,
        'nhead': nhead,
        'num_layers': num_layers
    }
    torch.save({'model_state_dict': model.state_dict()}, save_prefix + '.pt')
    with open(save_prefix + '_architecture.json', 'w') as f:
        json.dump(arch, f)
    with open(save_prefix + '_history.json', 'w') as f:
        json.dump(history, f)
    return save_prefix, history

############################################################
# Reporting Helpers
############################################################

def write_pretrain_history(save_dir: str,
                           experiment_id: str,
                           description: str,
                           pretrain_cfg: Dict,
                           finetune_cfg: Optional[Dict],
                           evaluation_protocol: Dict,
                           history: Dict):
    ensure_dir(save_dir)
    final_summary = {
        'final_train_loss': history['train_loss'][-1] if history.get('train_loss') else None,
        'final_val_loss': history['val_loss'][-1] if history.get('val_loss') else None,
        'final_recon_loss': history['recon_loss'][-1] if history.get('recon_loss') else None,
        'final_contrastive_loss': history['contrastive_loss'][-1] if history.get('contrastive_loss') else None
    }
    payload = {
        'meta': {
            'experiment_id': experiment_id,
            'description': description,
            'pretrain': pretrain_cfg,
            'finetune': finetune_cfg,
            'evaluation_protocol': evaluation_protocol,
            'final_epoch_summary': final_summary
        },
        'train_loss': history.get('train_loss', []),
        'val_loss': history.get('val_loss', []),
        'recon_loss': history.get('recon_loss', []),
        'contrastive_loss': history.get('contrastive_loss', [])
    }
    out_path = os.path.join(save_dir, experiment_id + '_history.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[REPORT] Wrote pretrain history: {out_path}')

def write_evaluation_report(save_dir: str,
                            experiment_id: str,
                            source_cfg: Dict,
                            target_cfg: Dict,
                            metrics: Dict,
                            zero_shot: bool,
                            use_epochs_normalized: bool = False):
    ensure_dir(save_dir)
    # Align source epochs key naming pattern ("epochs" vs "epochs_normalized") per examples
    source_meta = source_cfg.copy()
    if use_epochs_normalized and 'epochs' in source_meta:
        source_meta['epochs_normalized'] = source_meta.pop('epochs')
    payload = {
        'meta': {
            'experiment_id': experiment_id,
            'source': source_meta,
            'target': {
                'dataset': target_cfg.get('dataset'),
                'task': target_cfg.get('task'),
                'split': target_cfg.get('split', 'test'),
                'zero_shot': zero_shot
            }
        },
        'metrics': metrics
    }
    out_path = os.path.join(save_dir, experiment_id + '.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[REPORT] Wrote evaluation report: {out_path}')

############################################################
# Experiment Scenarios
############################################################

def experiment_1_pretrain_D1_test_D2(D1_path: str, D2_path: str, out_dir: str, epochs: int) -> Dict:
    ensure_dir(out_dir)
    D1 = enforce_canonical_normalization(load_dataset(D1_path))
    D2 = enforce_canonical_normalization(load_dataset(D2_path))
    debug_print_norm_stats(D1, 'D1 after normalization')
    debug_print_norm_stats(D2, 'D2 after normalization')

    save_prefix = os.path.join(out_dir, 'exp1_pretrain_D1')
    source_model, history = pretrain_model(D1, save_prefix, epochs=epochs)

    # Write structured pretrain history report
    pretrain_cfg = {'datasets': ['D1'], 'tasks': ['unimanual','bimanual'], 'cohorts': ['term','preterm'], 'epochs': epochs}
    eval_protocol = {
        'zero_shot': ['D2'],
        'tested_on': ['D1','D2'],
        'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']
    }
    write_pretrain_history(out_dir, 'exp1_pretrain_D1', 'Pretrain on D1 (human) full dataset; evaluate zero-shot on D2 (non-human primate). No fine-tuning on target.', pretrain_cfg, None, eval_protocol, history)

    runner = STCRLTransferLearningRunner(source_model_path=save_prefix, save_dir=ensure_dir(os.path.join(out_dir, 'exp1_results')))
    # Zero-shot on D2
    zero_shot_D2 = runner.run_zero_shot_transfer(D2)
    write_evaluation_report(os.path.join(out_dir, 'exp1_results'), 'exp1_D2_zero_shot', {'pretrain_dataset': 'D1', 'pretrain_task': 'all', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all'}, zero_shot_D2, zero_shot=True)
    # Zero-shot on D1 (held-out session)
    D1_train, D1_holdout = holdout_last_session(D1)
    zero_shot_D1 = runner.run_zero_shot_transfer(D1_holdout if len(D1_holdout) else D1)
    write_evaluation_report(os.path.join(out_dir, 'exp1_results'), 'exp1_D1_zero_shot_holdout', {'pretrain_dataset': 'D1', 'pretrain_task': 'all', 'epochs': epochs}, {'dataset': 'D1', 'task': 'holdout'}, zero_shot_D1, zero_shot=True)

    return {
        'zero_shot_D1': zero_shot_D1,
        'zero_shot_D2': zero_shot_D2
    }

def experiment_2_cross_task(D1_path: str, D2_path: str, out_dir: str, epochs: int) -> Dict:
    ensure_dir(out_dir)
    D1 = enforce_canonical_normalization(load_dataset(D1_path))
    D2 = enforce_canonical_normalization(load_dataset(D2_path))
    debug_print_norm_stats(D1, 'D1 after normalization (exp2)')
    debug_print_norm_stats(D2, 'D2 after normalization (exp2)')

    D1_unimanual = filter_unimanual(D1)
    save_prefix = os.path.join(out_dir, 'exp2_pretrain_D1_unimanual')
    source_model, history = pretrain_model(D1_unimanual, save_prefix, epochs=epochs)
    write_pretrain_history(out_dir, 'exp2_pretrain_D1_unimanual', 'Pretrain on D1 unimanual tasks; evaluate on bimanual and cross-species tasks.', {'datasets': ['D1'], 'tasks': ['unimanual'], 'cohorts': ['term','preterm'], 'epochs': epochs}, None, {'zero_shot': ['D1_bimanual','D2_tasks'], 'tested_on': ['D1','D2'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, history)

    runner = STCRLTransferLearningRunner(source_model_path=save_prefix, save_dir=ensure_dir(os.path.join(out_dir,'exp2_results')))

    # Zero-shot on D1 bimanual
    D1_bimanual = filter_bimanual(D1)
    zero_shot_D1_bimanual = runner.run_zero_shot_transfer(D1_bimanual)
    write_evaluation_report(os.path.join(out_dir,'exp2_results'), 'exp2_D1_bimanual_test', {'pretrain_dataset': 'D1', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D1', 'task': 'bimanual'}, zero_shot_D1_bimanual, zero_shot=False, use_epochs_normalized=True)

    # Zero-shot on specific D2 task (Scott_2001_e1) and all tasks
    if 'task_type_str' in D2.columns:
        D2_e1 = D2[D2['task_type_str'] == 'Scott_2001_e1']
        zero_shot_D2_e1 = runner.run_zero_shot_transfer(D2_e1) if len(D2_e1) else {'total_loss': None}
        if len(D2_e1):
            write_evaluation_report(os.path.join(out_dir,'exp2_results'), 'exp2_D2_zero_shot_Scott_2001_e1', {'pretrain_dataset': 'D1', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D2', 'task': 'Scott_2001_e1'}, zero_shot_D2_e1, zero_shot=True)
    else:
        zero_shot_D2_e1 = {'total_loss': None}
    zero_shot_D2_all = runner.run_zero_shot_transfer(D2)
    write_evaluation_report(os.path.join(out_dir,'exp2_results'), 'exp2_D2_zero_shot_all', {'pretrain_dataset': 'D1', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all'}, zero_shot_D2_all, zero_shot=True)

    return {
        'zero_shot_D1_bimanual': zero_shot_D1_bimanual,
        'zero_shot_D2_e1': zero_shot_D2_e1,
        'zero_shot_D2_all': zero_shot_D2_all
    }

def experiment_3_cross_subject(D1_path: str, D2_path: str, out_dir: str, epochs: int) -> Dict:
    ensure_dir(out_dir)
    D1 = enforce_canonical_normalization(load_dataset(D1_path))
    D2 = enforce_canonical_normalization(load_dataset(D2_path))
    debug_print_norm_stats(D1, 'D1 after normalization (exp3)')
    debug_print_norm_stats(D2, 'D2 after normalization (exp3)')

    D1_term = filter_cohort(D1, 'Term')
    save_prefix = os.path.join(out_dir, 'exp3_pretrain_D1_term')
    source_model, history = pretrain_model(D1_term, save_prefix, epochs=epochs)
    write_pretrain_history(out_dir, 'exp3_pretrain_D1_term', 'Pretrain on D1 Term cohort; fine-tune on Preterm for cross-subject transfer.', {'datasets': ['D1'], 'tasks': ['all'], 'cohorts': ['Term'], 'epochs': epochs}, {'phase': 'cross_subject', 'target_cohorts': ['Preterm'], 'epochs': 10}, {'zero_shot': ['D2'], 'tested_on': ['D1_preterm','D2'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, history)

    runner = STCRLTransferLearningRunner(source_model_path=save_prefix, save_dir=ensure_dir(os.path.join(out_dir,'exp3_results')))

    D1_preterm = filter_cohort(D1, 'Preterm')
    # Fine-tune: reuse cross-subject transfer API on preterm subset
    cs_model, cs_history, cs_results = runner.run_cross_subject_transfer(D1_preterm, epochs=10)
    # Fine-tune history JSON
    if isinstance(cs_history, dict):
        write_pretrain_history(out_dir, 'exp3_finetune_D1_preterm', 'Fine-tune on D1 Preterm cohort (cross-subject).', {'datasets': ['D1'], 'tasks': ['all'], 'cohorts': ['Preterm'], 'epochs': 10}, None, {'zero_shot': ['D2'], 'tested_on': ['D1_preterm','D2'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, cs_history)

    # Zero-shot on D2
    zero_shot_D2 = runner.run_zero_shot_transfer(D2)
    write_evaluation_report(os.path.join(out_dir,'exp3_results'), 'exp3_D2_zero_shot', {'pretrain_dataset': 'D1_term', 'pretrain_task': 'all', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all'}, zero_shot_D2, zero_shot=True)
    # Evaluate fine-tuned model on D1_preterm and D2
    from STCRL.TransferLearning.TransferLearningFramework import TransferLearningFramework
    framework = runner.transfer_framework
    from STCRL.STCRLDataset import STCRLModelFittingDataset
    D1_preterm_dataset = STCRLDataset = STCRLModelFittingDataset(D1_preterm)
    D2_dataset = STCRLModelFittingDataset(D2)
    ft_D1_eval = framework.evaluate_transfer_performance(cs_model, D1_preterm_dataset)
    ft_D2_eval = framework.evaluate_transfer_performance(cs_model, D2_dataset)
    write_evaluation_report(os.path.join(out_dir,'exp3_results'), 'exp3_D1_preterm_finetuned_test', {'pretrain_dataset': 'D1_term', 'pretrain_task': 'all', 'epochs': epochs}, {'dataset': 'D1', 'task': 'Preterm'}, ft_D1_eval, zero_shot=False, use_epochs_normalized=True)
    write_evaluation_report(os.path.join(out_dir,'exp3_results'), 'exp3_D2_finetuned_test', {'pretrain_dataset': 'D1_term', 'pretrain_task': 'all', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all'}, ft_D2_eval, zero_shot=False, use_epochs_normalized=True)

    return {
        'fine_tuned_D1_preterm_eval': ft_D1_eval,
        'fine_tuned_D2_eval': ft_D2_eval,
        'zero_shot_D2': zero_shot_D2,
        'cross_subject_history': cs_history
    }

def experiment_4_adaptive(D1_path: str, D2_path: str, out_dir: str, epochs: int) -> Dict:
    ensure_dir(out_dir)
    D1 = enforce_canonical_normalization(load_dataset(D1_path))
    D2 = enforce_canonical_normalization(load_dataset(D2_path))
    debug_print_norm_stats(D1, 'D1 after normalization (exp4)')
    debug_print_norm_stats(D2, 'D2 after normalization (exp4)')

    # Combined filter: Term cohort & unimanual tasks
    D1_term_unimanual = filter_cohort(filter_unimanual(D1), 'Term')
    save_prefix = os.path.join(out_dir, 'exp4_pretrain_D1_term_unimanual')
    source_model, history = pretrain_model(D1_term_unimanual, save_prefix, epochs=epochs)
    write_pretrain_history(out_dir, 'exp4_pretrain_D1_term_unimanual', 'Pretrain on D1 Term cohort unimanual; adaptive sequential fine-tuning.', {'datasets': ['D1'], 'tasks': ['unimanual'], 'cohorts': ['Term'], 'epochs': epochs}, {'phase': 'adaptive', 'sequence': ['Preterm','bimanual'], 'epochs_each': 8}, {'zero_shot': ['D2_initial'], 'tested_on': ['D1_final','D2_final'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, history)

    runner = STCRLTransferLearningRunner(source_model_path=save_prefix, save_dir=ensure_dir(os.path.join(out_dir,'exp4_results')))

    # Sequential adaptation: first Preterm cohort, then bimanual tasks
    D1_preterm = filter_cohort(D1, 'Preterm')
    cs_model, cs_history, _ = runner.run_cross_subject_transfer(D1_preterm, epochs=8)
    if isinstance(cs_history, dict):
        write_pretrain_history(out_dir, 'exp4_finetune_D1_preterm', 'Adaptive phase 1: cross-subject fine-tune on Preterm cohort.', {'datasets': ['D1'], 'tasks': ['unimanual'], 'cohorts': ['Preterm'], 'epochs': 8}, None, {'zero_shot': [], 'tested_on': ['D1_preterm'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, cs_history)

    # Replace source model in framework with adapted one for next phase
    runner.source_model = cs_model
    runner.transfer_framework.source_model = cs_model

    D1_bimanual = filter_bimanual(D1)
    ct_model, ct_history, _ = runner.run_cross_task_transfer(D1_bimanual, epochs=8)
    if isinstance(ct_history, dict):
        write_pretrain_history(out_dir, 'exp4_finetune_D1_bimanual', 'Adaptive phase 2: cross-task fine-tune on bimanual tasks.', {'datasets': ['D1'], 'tasks': ['bimanual'], 'cohorts': ['Term','Preterm'], 'epochs': 8}, None, {'zero_shot': [], 'tested_on': ['D1_bimanual'], 'metrics': ['total_loss','reconstruction_loss','completion_time_loss','task_type_loss','rmsd_loss','success_loss']}, ct_history)

    # Evaluate final adaptive model
    from STCRL.TransferLearning.TransferLearningFramework import TransferLearningFramework
    framework = runner.transfer_framework
    from STCRL.STCRLDataset import STCRLModelFittingDataset
    D1_dataset_final = STCRLModelFittingDataset(D1)
    D2_dataset_final = STCRLModelFittingDataset(D2)
    adaptive_D1_eval = framework.evaluate_transfer_performance(ct_model, D1_dataset_final)
    adaptive_D2_eval = framework.evaluate_transfer_performance(ct_model, D2_dataset_final)
    write_evaluation_report(os.path.join(out_dir,'exp4_results'), 'exp4_D1_adaptive_final_test', {'pretrain_dataset': 'D1_term_unimanual', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D1', 'task': 'all_final'}, adaptive_D1_eval, zero_shot=False, use_epochs_normalized=True)
    write_evaluation_report(os.path.join(out_dir,'exp4_results'), 'exp4_D2_adaptive_final_test', {'pretrain_dataset': 'D1_term_unimanual', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all_final'}, adaptive_D2_eval, zero_shot=False, use_epochs_normalized=True)

    zero_shot_D2_from_initial = runner.run_zero_shot_transfer(D2)
    write_evaluation_report(os.path.join(out_dir,'exp4_results'), 'exp4_D2_zero_shot_initial', {'pretrain_dataset': 'D1_term_unimanual', 'pretrain_task': 'unimanual', 'epochs': epochs}, {'dataset': 'D2', 'task': 'all'}, zero_shot_D2_from_initial, zero_shot=True)

    return {
        'adaptive_D1_eval': adaptive_D1_eval,
        'adaptive_D2_eval': adaptive_D2_eval,
        'zero_shot_initial_D2': zero_shot_D2_from_initial,
        'cross_subject_history': cs_history,
        'cross_task_history': ct_history
    }

############################################################
# Orchestrator
############################################################

def run_all_transfer_experiments(D1_path: str, D2_path: str, output_root: str, pretrain_epochs: int) -> Dict:
    ensure_dir(output_root)
    results = {}
    results['exp1'] = experiment_1_pretrain_D1_test_D2(D1_path, D2_path, os.path.join(output_root, 'exp1'), pretrain_epochs)
    results['exp2'] = experiment_2_cross_task(D1_path, D2_path, os.path.join(output_root, 'exp2'), pretrain_epochs)
    results['exp3'] = experiment_3_cross_subject(D1_path, D2_path, os.path.join(output_root, 'exp3'), pretrain_epochs)
    results['exp4'] = experiment_4_adaptive(D1_path, D2_path, os.path.join(output_root, 'exp4'), pretrain_epochs)

    # Compute representative zero-shot pct diff where possible
    # Example: exp3 fine-tuned vs zero-shot D2
    try:
        zs_loss = results['exp3']['zero_shot_D2']['total_loss']
        ft_loss = results['exp3']['fine_tuned_D2_eval']['total_loss']
        results['exp3']['zero_shot_pct_diff'] = compute_zero_shot_pct_diff(zs_loss, ft_loss)
    except Exception:
        results['exp3']['zero_shot_pct_diff'] = None

    # Save summary
    summary_path = os.path.join(output_root, 'transfer_experiments_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run comprehensive transfer experiments for rebuttal.')
    parser.add_argument('--D1', type=str, required=True, help='Path to human dataset CSV')
    parser.add_argument('--D2', type=str, required=True, help='Path to monkey dataset CSV')
    parser.add_argument('--out', type=str, default='transfer_learning_results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=25, help='Base pretraining epochs')
    args = parser.parse_args()

    results = run_all_transfer_experiments(args.D1, args.D2, args.out, args.epochs)
    print('Transfer experiments completed. Summary saved.')
