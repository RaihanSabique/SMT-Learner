%% Clear the Workspace variables. 
clear all; close all; clc;

%% Load all participants data
% Specify the directory containing the files
folderPath = '../Dataset/MSL_ParticipantWiseData_MAT/';
% Get a list of all .mat files in the directory
filePattern = fullfile(folderPath, '*.mat'); % Change the pattern if needed
files = dir(filePattern);
varNamesDataTable = {'participant_id', 'session_no', 'task_type', 'trial_no', 'day', 'block',...
       'start_point_x', 'start_point_y', 'target_point_x', 'target_point_y', 'start_time', 'end_time',...
       'quadrant', 'is_success', 'actual_dist', 'movement_dist', 'completion_time',...
       'path', 'time_string', 'time_diff_ms'};
% varTypes = {'datetime', 'double', 'string'};  % specify data types
dataTable = [];

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
%     dob = datetime(d.dob, 'InputFormat', 'yyyy-MM-dd');
%     currentDate = datetime('today');
%     d.age = floor(years(currentDate - dob));
%     group = 'Term';
%     isValueInList = ismember(d.name, preterm_participant);
%     if isValueInList
%         group = 'Preterm';
%     end
%     d.group = group;
    uni_block = fieldnames(d.unimanual);
    d.uniday1 = sum(contains(uni_block, 'day1'));
    d.uniday2 = sum(contains(uni_block, 'day2'));
    d.uniday3 = sum(contains(uni_block, 'day3'));
    bi_block = fieldnames(d.bimanual);
    d.biday1 = sum(contains(bi_block, 'day1'));
    d.biday2 = sum(contains(bi_block, 'day2'));
    d.biday3 = sum(contains(bi_block, 'day3'));
    if(length(uni_block)==5 && length(bi_block)==5)
        for i=1:length(uni_block)
            for j=1:length(d.unimanual.(uni_block{i}))
                data = {};
                % Trial wise data
                data = [data, pid]; % Participant id
                data = [data, i]; % Session No
                data = [data, 0]; % 0 for Unimanual
                data = [data, j]; % Trial No
                cellData = d.unimanual.(uni_block{i})(j, :);
                data = [data, cellData{2}]; % Day
                data = [data, cellData{3}]; % Block
                data = [data, cellData{4}(1)]; % StartPosition X
                data = [data, cellData{4}(2)]; % StartPosition Y
                data = [data, cellData{5}(1)]; % TargetPosition X
                data = [data, cellData{5}(2)]; % TargetPosition Y
                data = [data, cellData{6}]; % start time 
                data = [data, cellData{7}]; % end time
                data = [data, cellData{8}]; % quadrant
                data = [data, cellData{9}]; % is success? 0 or 1
                data = [data, cellData{11}]; % actual dist
                data = [data, cellData{12}]; % mov dist
                data = [data, cellData{13}]; % completion time
                % spatiotemporal data
%               % Movement path
                path_uni = d.unimanual.(uni_block{i})(j, 14);
                points = cell2mat(path_uni);
                point_strings = arrayfun(@(n) sprintf('(%g,%g)', points(n,1), points(n,2)), ...
                    1:size(points,1), 'UniformOutput', false);
                points_str = ['[' strjoin(point_strings, ',') ']']; % Final points array data
                data = [data, points_str]; % Append Movement path array
                % Movement time
                time_uni = d.unimanual.(uni_block{i})(j, 15);
                times = cell2mat(time_uni);
                time_strings = cellstr(times);
                time_str = strjoin(cellfun(@(x) sprintf('''%s''', x), time_strings, 'UniformOutput', false), ',');
                time_str = ['[' time_str ']']; % final times array data
                data = [data, time_str]; % Append time array
                % Calculate time different in miliseconds
                % Convert to datetime array
                datetime_array = datetime(time_strings, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS');
                % Set the output format
                datetime_array.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';
                start_time = cellData{6}; % start time
                dt = datetime(start_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS');
                dt.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';
                time_diffs = datetime_array - dt;  % Returns duration array
                ms_diff = milliseconds(time_diffs);
                ms_diff_array = sprintf('%d,', ms_diff);
                ms_diff_array = ['[' ms_diff_array(1:end-1) ']'];  % Final time different array in ms
                data = [data, ms_diff_array]; % Append time diff in ms array
                dataTable = [dataTable; data];
            end
        end
        for i=1:length(bi_block)
            for j=1:length(d.bimanual.(bi_block{i}))
                data = {};
                % Trial wise data
                data = [data, pid]; % Participant id
                data = [data, i]; % Session No
                data = [data, 1]; % 1 for Bimanual
                data = [data, j]; % Trial No
                cellData = d.bimanual.(bi_block{i})(j, :);
                data = [data, cellData{2}]; % Day
                data = [data, cellData{3}]; % Block
                data = [data, cellData{4}(1)]; % StartPosition X
                data = [data, cellData{4}(2)]; % StartPosition Y
                data = [data, cellData{5}(1)]; % TargetPosition X
                data = [data, cellData{5}(2)]; % TargetPosition Y
                data = [data, cellData{6}]; % start time 
                data = [data, cellData{7}]; % end time
                data = [data, cellData{8}]; % quadrant
                data = [data, cellData{9}]; % is success? 0 or 1
                data = [data, cellData{11}]; % actual dist
                data = [data, cellData{12}]; % mov dist
                data = [data, cellData{13}]; % completion time
                % spatiotemporal data
%               % Movement path
                path_bi = d.bimanual.(bi_block{i})(j, 14);
                points = cell2mat(path_bi);
                point_strings = arrayfun(@(n) sprintf('(%g,%g)', points(n,1), points(n,2)), ...
                    1:size(points,1), 'UniformOutput', false);
                points_str = ['[' strjoin(point_strings, ',') ']']; % Final points array data
                data = [data, points_str]; % Append Movement path array
                % Movement time
                time_bi = d.bimanual.(bi_block{i})(j, 15);
                times = cell2mat(time_bi);
                time_strings = cellstr(times);
                time_str = strjoin(cellfun(@(x) sprintf('''%s''', x), time_strings, 'UniformOutput', false), ',');
                time_str = ['[' time_str ']']; % final times array data
                data = [data, time_str]; % Append time array
                % Calculate time different in miliseconds
                % Convert to datetime array
                datetime_array = datetime(time_strings, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS');
                % Set the output format
                datetime_array.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';
                start_time = cellData{6}; % start time
                dt = datetime(start_time, 'InputFormat', 'yyyy-MM-dd HH:mm:ss.SSSS');
                dt.Format = 'yyyy-MM-dd HH:mm:ss.SSSS';
                time_diffs = datetime_array - dt;  % Returns duration array
                ms_diff = milliseconds(time_diffs);
                ms_diff_array = sprintf('%d,', ms_diff);
                ms_diff_array = ['[' ms_diff_array(1:end-1) ']'];  % Final time different array in ms
                data = [data, ms_diff_array]; % Append time diff in ms array
                dataTable = [dataTable; data];
            end
        end
    end
end
T = cell2table(dataTable, 'VariableNames', varNamesDataTable);

writetable(T, '../Dataset/SMT_Dataset/human_trajectory_dataset.csv');
    