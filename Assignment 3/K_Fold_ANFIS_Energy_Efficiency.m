clear; clc; close;
tic
% Load the dataset from an Excel file
data = readtable('energy_efficiency_data.xlsx');

% Assuming the first 8 columns are input features and the 9th column is the target (heating load)
inputs = table2array(data(:, 1:8));
target = table2array(data(:, 9));

% Normalize the inputs
%meanVals = mean(inputs);
%stdVals = std(inputs);
%inputs_normalized = (inputs - meanVals) ./ stdVals;

% Normalize the targets
%meanTargs = mean(target);
%stdTargs = std(target);
%target_normalized = (targets - meanVals) ./ stdVals;

% Normalize the inputs using min-max normalization
minVals = min(inputs);
maxVals = max(inputs);
inputs_normalized = (inputs - minVals) ./ (maxVals - minVals);

% Normalize the targets using min-max normalization
minVals = min(target);
maxVals = max(target);
target_normalized = (inputs - minVals) ./ (maxVals - minVals);

% Initialize cross-validation
cv = cvpartition(size(data,1),'KFold',3);

% Initialize vectors to store metrics for each fold
mae_values = zeros(cv.NumTestSets,1);
mse_values = zeros(cv.NumTestSets,1);
mre_values = zeros(cv.NumTestSets,1);

for i = 1:cv.NumTestSets
    % Indices for training and test set
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    % Splitting the data
    inputs_train = inputs_normalized(trainIdx,:);
    target_train = target_normalized(trainIdx);
    inputs_test = inputs_normalized(testIdx,:);
    target_test = target_normalized(testIdx);
    
    % Generate an initial FIS structure
    %%optGF = genfisOptions('FCMClustering','FISType','sugeno');
    optGF = genfisOptions('GridPartition');
    optGF.NumMembershipFunctions = 2;
    optGF.InputMembershipFunctionType = "gbellmf";
    fis = genfis(inputs_train, target_train, optGF);
    
    % Train the ANFIS model
    numEpochs = 200; % Adjust as needed
    [trainedFis, trainError] = anfis([inputs_train target_train], fis, numEpochs);
    
    % Predict the heating load for the test set
    predicted = evalfis(trainedFis, inputs_test);
    
    % Calculate Mean Absolute Error (MAE)
    mae_values(i) = mean(abs(predicted - target_test));
    
    % Calculate Mean Squared Error (MSE)
    mse_values(i) = mean((predicted - target_test).^2);
    
    % Calculate Mean Relative Error (MRE)
    % Note: Add a small number to avoid division by zero
    mre_values(i) = mean(abs((predicted - target_test) ./ (target_test + eps)));
end

% Display average of the metrics
disp(['Average MAE: ', num2str(mean(mae_values))]);
disp(['Average MSE: ', num2str(mean(mse_values))]);
disp(['Average MRE: ', num2str(mean(mre_values))]);
toc