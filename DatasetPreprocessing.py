
from DataProcessing.DatasetCombining import combine_csv_files
from DataProcessing.Preprocessing import Preprocessing
from DataProcessing.Normalization import normalize_trajectory_sequence_3d
import pandas as pd

if __name__ == '__main__':
    selected_columns = {
        'file1': ['participant_id', 'session_no', 'task_type', 'trial_no', 'day', 'block',
                  'start_point_x', 'start_point_y', 'target_point_x', 'target_point_y',
                  'start_time', 'end_time', 'quadrant', 'is_success', 'actual_dist',
                  'movement_dist', 'completion_time', 'path', 'time_string',
                  'time_diff_ms'],
        'file2': ['participant_id', 'Age', 'Cohort', 'Gestational_Age']
    }

    merged_data = combine_csv_files('Dataset/SMT_Dataset/human_trajectory_dataset.csv', 'Dataset/BehavioralData/R03_Behavioral_Data.csv',
                                    selected_columns)

    mabc_df = pd.read_csv("Dataset/BehavioralData/MABC.csv")
    mabc = mabc_df[['participant_id', 'Total Test Score ', 'Standard Score ', 'Percentile Rank ']]
    mabc.columns = ['participant_id', 'mabc_total_test_score', 'mabc_standard_score', 'mabc_percentile']
    merged_data = pd.merge(merged_data, mabc,
                         on='participant_id',
                         how='inner')
    print(merged_data.shape)
    # Save the result
    merged_data.to_csv('Dataset/SMT_Dataset/dataset_combined.csv', index=False)
    data_path = 'Dataset/SMT_Dataset/dataset_combined.csv'
    save_path = 'Dataset/SMT_Dataset/'
    preprocessing = Preprocessing(data_path=data_path)
    dataset = preprocessing.getPreprocessedData()
    dataset["normalized_trajectory"] = dataset.apply(
        lambda x: normalize_trajectory_sequence_3d(x['path'], x['time_diff_ms']), axis=1)
    print(dataset.head(10))
    dataset.to_csv(save_path + 'preprocessed_human_smt_dataset.csv', index=False)

