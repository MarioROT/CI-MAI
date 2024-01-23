clear; clc; close;
% Load the dataset from an Excel file
data = readtable('energy_efficiency_data.xlsx');

% Assuming the first 8 columns are input features and the 9th column is the target (heating load)
inputs = table2array(data(:, 1:8));
target = table2array(data(:, 9));

% Assuming inputs is your dataset
meanVals = mean(inputs);
stdVals = std(inputs);

% Standardize data
inputs_standardized = (inputs - meanVals) ./ stdVals;

%% Split data into training and testing sets
cv = cvpartition(size(data,1),'HoldOut',0.2); % for example, 80% training, 20% testing
idx = cv.test;

% Separate the training and testing data
inputs_train = inputs_standardized(~idx,:);
target_train = target(~idx,:);
inputs_test = inputs_standardized(idx,:);
target_test = target(idx,:);

%% Generate an initial FIS structure
optGF = genfisOptions("GridPartition");
%optGF = genfisOptions('FCMClustering','FISType','sugeno');
optGF.NumMembershipFunctions = 2;
optGF.InputMembershipFunctionType = "gbellmf";
fis = genfis(inputs_train, target_train, optGF);

%% Train the ANFIS model
numEpochs = 3; % Number of epochs for training
[trainedFis,trainError] = anfis([inputs_train target_train], fis, numEpochs);
%% Predict the heating load for the test set
predicted = evalfis(trainedFis, inputs_test);

% Calculate Mean Absolute Error (MAE)
mae = mean(abs(predicted - target_test));
disp(['Mean Absolute Error: ', num2str(mae)]);
