clear; clc; close;
tic
% Load the dataset 
data = readtable('energy_efficiency_data.xlsx');

% Getting inputs and targets
inputs = table2array(data(:, 1:8));
target = table2array(data(:, 9));

% Normalize the inputs - Z-score
%meanVals = mean(inputs);
%stdVals = std(inputs);
%inputs_normalized = (inputs - meanVals) ./ stdVals;

% Normalize the targets
%meanTargs = mean(target);
%stdTargs = std(target);
%target_normalized = (targets - meanVals) ./ stdVals;

% Normalize the inputs - Min-Max normalization
minVals = min(inputs);
maxVals = max(inputs);
inputs_normalized = (inputs - minVals) ./ (maxVals - minVals);

% Normalize the targets 
minVals = min(target);
maxVals = max(target);
target_normalized = (inputs - minVals) ./ (maxVals - minVals);

% Cross-validation
cv = cvpartition(size(data,1),'KFold',3);


mae_values = zeros(cv.NumTestSets,1);
mse_values = zeros(cv.NumTestSets,1);
mre_values = zeros(cv.NumTestSets,1);

for i = 1:cv.NumTestSets
    trainIdx = training(cv, i);
    testIdx = test(cv, i);
    
    % Splitting the data
    inputs_train = inputs_normalized(trainIdx,:);
    target_train = target_normalized(trainIdx);
    inputs_test = inputs_normalized(testIdx,:);
    target_test = target_normalized(testIdx);
    
    % Generate an initial FIS structure
    %optGF = genfisOptions('FCMClustering','FISType','sugeno');
    %optGF = genfisOptions('SubtractiveClustering');
    optGF = genfisOptions('GridPartition');
    optGF.NumMembershipFunctions = [4 4 3 2 2 2 2 3];
    optGF.InputMembershipFunctionType = "gauss2mf";
    fis = genfis(inputs_train, target_train, optGF);
    
    % Train the ANFIS model
    numEpochs = 10; 
    [trainedFis, trainError] = anfis([inputs_train target_train], fis, numEpochs);
    
    % Predictions
    predicted = evalfis(trainedFis, inputs_test);
    
    % Mean Absolute Error (MAE)
    mae_values(i) = mean(abs(predicted - target_test));
    
    % Mean Squared Error (MSE)
    mse_values(i) = mean((predicted - target_test).^2);
    
    %  Mean Relative Error (MRE)
    mre_values(i) = mean(abs((predicted - target_test) ./ (target_test + eps)));
end

disp(['Average MAE: ', num2str(mean(mae_values))]);
disp(['Average MSE: ', num2str(mean(mse_values))]);
disp(['Average MRE: ', num2str(mean(mre_values))]);
toc