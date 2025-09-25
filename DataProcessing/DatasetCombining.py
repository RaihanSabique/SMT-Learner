import pandas as pd


def combine_csv_files(file1_path, file2_path, selected_columns=None):
    """
    Read two CSV files and combine selected columns based on participant_id

    Parameters:
    file1_path (str): Path to first CSV file
    file2_path (str): Path to second CSV file
    selected_columns (dict): Dictionary specifying which columns to select from each file
                           Format: {'file1': ['col1', 'col2'], 'file2': ['col3', 'col4']}
                           If None, all columns will be used

    Returns:
    pandas.DataFrame: Combined dataframe with matched participant_id
    """
    # Read CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Verify participant_id exists in both files
    if 'participant_id' not in df1.columns or 'participant_id' not in df2.columns:
        raise ValueError("Both CSV files must contain 'participant_id' column")

    # Select columns if specified
    if selected_columns:
        if 'file1' in selected_columns:
            # Always include participant_id
            cols1 = ['participant_id'] + [col for col in selected_columns['file1']
                                          if col != 'participant_id']
            df1 = df1[cols1]

        if 'file2' in selected_columns:
            cols2 = ['participant_id'] + [col for col in selected_columns['file2']
                                          if col != 'participant_id']
            df2 = df2[cols2]

    # Merge dataframes on participant_id
    merged_df = pd.merge(df1, df2,
                         on='participant_id',
                         how='inner',
                         suffixes=('_file1', '_file2'))

    # Print some information about the merge
    print(f"Number of rows in file1: {len(df1)}")
    print(f"Number of rows in file2: {len(df2)}")
    print(f"Number of rows in merged file: {len(merged_df)}")
    print(f"Number of unique participant_ids in merged file: {merged_df['participant_id'].nunique()}")

    return merged_df
#
# if __name__ == '__main__':
#     # Specify which columns you want from each file
#     selected_columns = {
#         'file1': ['participant_id', 'session_no', 'task_type', 'trial_no', 'day', 'block',
#        'start_point_x', 'start_point_y', 'target_point_x', 'target_point_y',
#        'start_time', 'end_time', 'quadrant', 'is_success', 'actual_dist',
#        'movement_dist', 'completion_time', 'path', 'time_string',
#        'time_diff_ms'],
#         'file2': ['participant_id', 'Age', 'Cohort', 'Gestational_Age']
#     }
#
#     merged_data = combine_csv_files('../Dataset/dataset_new.csv', '../Dataset/R03_Behavioral_Data.csv', selected_columns)
#
#     # Save the result
#     merged_data.to_csv('../Dataset/dataset_combined.csv', index=False)
