import pandas as pd
from SMTLearner.TransferLearning.SMTLearnerTransferLearning import SMTLearnerTransferLearningRunner

def create_transfer_learning_pipeline(source_model_path, target_data_path, save_dir):
    """
    Complete pipeline for SMTLearner transfer learning experiments.

    Args:
        source_model_path: Path to pre-trained source model
        target_data_path: Path to target dataset CSV
        save_dir: Directory to save results
    """
    # Load target data
    target_df = pd.read_csv(target_data_path)

    # Preprocess if needed (add normalization, factorize participant IDs, etc.)
    if 'participant_id' in target_df.columns:
        target_df['participant_id'], _ = pd.factorize(target_df['participant_id'])

    # Initialize and run transfer learning
    runner = SMTLearnerTransferLearningRunner(source_model_path, save_dir)
    results, histories = runner.run_comprehensive_transfer_study(target_df)

    return results, histories

if __name__ == '__main__':
    source_model_path = "saved_models/SMTLearner/models/multi_model"
    target_data_path = "Dataset/SMT_Dataset/mp_preprocessed_dataset.csv"
    save_dir = "saved_models/SMTLearner_transfer_learning/"
    # Initialize runner
    runner = SMTLearnerTransferLearningRunner(source_model_path)