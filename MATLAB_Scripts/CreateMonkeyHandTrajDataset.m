%% Clear the Workspace variables. 
clear all; close all; clc;

%% Load all participants data
% Specify the directory containing the files
folderPath = '../Dataset/Scott_2001/';
% Get a list of all .mat files in the directory
filePattern = fullfile(folderPath, '*.mat'); % Change the pattern if needed
files = dir(filePattern);
% varNamesDataTable = {'participant_id', 'session_no', 'task_type', 'trial_no', 'day', 'block',...
%        'start_point_x', 'start_point_y', 'target_point_x', 'target_point_y', 'start_time', 'end_time',...
%        'quadrant', 'is_success', 'actual_dist', 'movement_dist', 'completion_time',...
%        'path', 'time_string', 'time_diff_ms'};
% varTypes = {'datetime', 'double', 'string'};  % specify data types
resultsTable = table();

% Counter for rows
rowIdx = 1;

% Loop through each file and load it
for k = 1:length(files)
    baseFileName = files(k).name;
    fullFileName = fullfile(folderPath, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    name = strsplit(baseFileName, '.');
    pid = name{1};
    
    % Load the .mat file
    loadedData = load(fullFileName);
    fields = fieldnames(loadedData);
    d = loadedData.(fields{1});
    Subject = d;
    for session_idx = 1:size(Subject, 1)
    % Get the current subject
        current_session = Subject(session_idx);
    %     disp(current_session);
        % Check if the Trial field exists and is non-empty
        if isfield(current_session, 'Trial') && ~isempty(current_session.Trial)
            % Loop through each Trial for the current subject
            for trial_idx = 1:size(current_session.Trial, 1)
                current_trial = current_session.Trial(trial_idx);
                % Add this trial's data directly to the table
                resultsTable.Subject_Id(rowIdx) = string(pid);
                resultsTable.Session_No(rowIdx) = session_idx;
                resultsTable.Trial_No(rowIdx) = trial_idx;
                
                % Store the arrays as strings using mat2str
                resultsTable.Time(rowIdx) = string(mat2str(current_trial.Time));
                resultsTable.HandPos(rowIdx) = string(mat2str(current_trial.HandPos(:,1:2)));
                resultsTable.TargetPos(rowIdx) = string(mat2str(current_trial.TargetPos(:,1:2)));
                
                rowIdx = rowIdx + 1;
    
            end
        else
            fprintf('Trial field missing or empty for Subject %d\n', session_idx);
        end
    end
end

writetable(resultsTable, 'experiment_data.csv');

% resultsTable = table('Size', [0, 6], ...
%                     'VariableTypes', {'string', 'double', 'double', 'cell', 'cell', 'cell'}, ...
%                     'VariableNames', {'Subject_Id', 'Session_No', 'Trial_No', 'Time', 'HandPos', 'TargetPos'});
resultsTable = table();

% Counter for rows
rowIdx = 1;

subject_id = 1;
% Loop through each Subject
for session_idx = 1:size(Subject, 1)
    % Get the current subject
    current_session = Subject(session_idx);
%     disp(current_session);
    % Check if the Trial field exists and is non-empty
    if isfield(current_session, 'Trial') && ~isempty(current_session.Trial)
        % Loop through each Trial for the current subject
        for trial_idx = 1:size(current_session.Trial, 1)
            current_trial = current_session.Trial(trial_idx);
            % Add this trial's data directly to the table
            resultsTable.Subject_Id(rowIdx) = string(sprintf('Subject_%d', subject_id));
            resultsTable.Session_No(rowIdx) = session_idx;
            resultsTable.Trial_No(rowIdx) = trial_idx;
            
            % Store the arrays as strings using mat2str
            resultsTable.Time(rowIdx) = string(mat2str(current_trial.Time));
            resultsTable.HandPos(rowIdx) = string(mat2str(current_trial.HandPos(:,1:2)));
            resultsTable.TargetPos(rowIdx) = string(mat2str(current_trial.TargetPos(:,1:2)));
            
            rowIdx = rowIdx + 1;

        end
    else
        fprintf('Trial field missing or empty for Subject %d\n', session_idx);
    end
end

writetable(resultsTable, 'experiment_data.csv');

disp(resultsTable(1:10, :)); % Show first 10 rows

flatTable = table();
for i = 1:height(resultsTable)
    subj = resultsTable.Subject_Id(i);
    sess = resultsTable.Session_No(i);
    trial = resultsTable.Trial_No(i);
    
    % Get arrays
    timeArray = resultsTable.Time{i}{1};
    handPosArray = resultsTable.HandPos{i}{1};
    targetPosArray = resultsTable.TargetPos{i}{1};
    
    % Create column names for each time point and dimension
    for t = 1:length(timeArray)
        rowName = sprintf('%s_S%d_T%d_t%d', subj, sess, trial, t);
        flatTable.(rowName).Time = timeArray(t);
        flatTable.(rowName).HandPos_X = handPosArray(t,1);
        flatTable.(rowName).HandPos_Y = handPosArray(t,2);
        flatTable.(rowName).HandPos_Z = handPosArray(t,3);
        flatTable.(rowName).TargetPos_X = targetPosArray(t,1);
        flatTable.(rowName).TargetPos_Y = targetPosArray(t,2);
        flatTable.(rowName).TargetPos_Z = targetPosArray(t,3);
    end
end

% Write flattened table to CSV
writetable(flatTable, 'experiment_data_flat.csv');

[n, m] = size(all_hand_positions);

% Iterate through each element of the cell array
for i = 1:n
    for j = 1:m
        % Access the element at position (i,j)
        currentHandPos = all_hand_positions{i,j};
        currentTime = all_times{i,j};
        currentTargetPos = all_target_positions{i,j};
        
        % Do something with the current element
        disp(['Element at position (', num2str(i), ',', num2str(j), '):']);
        disp(currentElement);
    end
end

% Initialize an empty table to hold all trajectories
trajectories = table();

% Initialize counters for rows in the final table
row_count = 1;

% Loop through each Subject
for subject_idx = 1:size(all_hand_positions, 1)
    % Loop through each Trial for the current subject
    for trial_idx = 1:size(all_hand_positions, 2)
        % Get current hand position and time data
        hand_pos = all_hand_positions{subject_idx, trial_idx};
        time_data = all_times{subject_idx, trial_idx};
%         disp()
        % Check if both hand position and time data exist
        if ~isempty(hand_pos) && ~isempty(time_data)
            % Get number of time points
            num_points = length(time_data);
            
            % Create temporary table for this trial
            temp_table = table();
            
            % Extract x and y coordinates
            x_coords = hand_pos(:, 1)'; % Transpose to make column vector
            y_coords = hand_pos(:, 2)'; % Transpose to make column vector
            t = time_data(:)';
            % Create the table with trajectory data
            temp_table.x = x_coords';
            temp_table.y = y_coords';
            temp_table.time = t'; % Ensure it's a column vector
            
%             % Add subject and trial identifiers
%             temp_table.subject = repmat(subject_idx, num_points, 1);
%             temp_table.trial = repmat(trial_idx, num_points, 1);
            
            % Append to main table
            trajectories = [trajectories; temp_table];
        end
    end
    
%     % Display progress every 10 subjects
%     if mod(subject_idx, 10) == 0
%         fprintf('Processed %d/%d subjects\n', subject_idx, size(all_hand_positions, 1));
%     end
end

% Save the trajectory dataset
writetable(trajectories, '../Dataset/SMT_Dataset/monkey_trajectory_dataset.csv');

% Display summary
fprintf('Created trajectory dataset with %d data points from %d subjects\n', ...
    height(trajectories), size(all_hand_positions, 1));

% Optional: Create a visualization of trajectories from a few subjects
try
    figure;
    hold on;
    
    % Get unique subjects
    subjects_to_plot = unique(trajectories.subject);
    
    % Limit to first 5 subjects for clarity
    if length(subjects_to_plot) > 5
        subjects_to_plot = subjects_to_plot(1:5);
    end
    
    % Plot each subject with a different color
    colors = lines(length(subjects_to_plot));
    
    for i = 1:length(subjects_to_plot)
        subject = subjects_to_plot(i);
        
        % Get data for this subject
        subject_data = trajectories(trajectories.subject == subject, :);
        
        % Get unique trials
        trials = unique(subject_data.trial);
        
        % Plot first 3 trials
        max_trials = min(3, length(trials));
        
        for j = 1:max_trials
            trial = trials(j);
            
            % Get data for this trial
            trial_data = subject_data(subject_data.trial == trial, :);
            
            % Plot trajectory
            plot(trial_data.x, trial_data.y, 'Color', colors(i,:), ...
                 'DisplayName', sprintf('Subject %d, Trial %d', subject, trial));
        end
    end
    
    title('Sample Hand Position Trajectories');
    xlabel('X Position');
    ylabel('Y Position');
    legend('show');
    grid on;
    hold off;
catch
    fprintf('Error creating visualization\n');
end