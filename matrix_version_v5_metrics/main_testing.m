%% Testing of trained nodeParameters for ASMG1 model with BSM1 with exogenous excitment on BSM1 configurations

% Xiangjun Huang, Brunel University London, Jun 2024

%% 0. options
clc;
clear;
close all

nodeParametersFile = "data/20240601_rain_0_12_nodeParameters.mat";
normFile = "data/20240601_rain_0_12_globalMeanStd.mat";
normFromSaved  = true;
plotTestData = false;
plotResult = true;

%% 1. get testing data

%             file,       day~ day, if train? 
dataInfo = {"dry_r1.mat",   7,  14,  true; 
            "dry_r2.mat",   7,  14,  true; 
            "dry_r3.mat",   7,  14,  true; 
            "dry_r4.mat",   7,  14,  true; 
            "dry_r5.mat",   7,  14,  true; 
            "rain_r1.mat",  0,  12,  false;
            "rain_r2.mat",  0,  12,  false;
            "rain_r3.mat",  0,  12,  false;
            "rain_r4.mat",  0,  12,  false;
            "rain_r5.mat",  0,  12,  false;
            "storm_r1.mat", 0,  14,  false;
            "storm_r2.mat", 0,  14,  false;
            "storm_r3.mat", 0,  14,  false;
            "storm_r4.mat", 0,  14,  false;
            "storm_r5.mat", 0,  14,  false;};

[tTest, uTest, yTest, y0Test, tuyRowTest, rNameTest, vTest] = get_mat_data(dataInfo); % default folder: "../data/"

set(groot, 'defaultFigureWindowState', 'maximized'); % maximum figure windows

if plotTestData
    plot_data(tuyRowTest, rNameTest, tTest, uTest(:,1:19), yTest, yLabels={'uTrain','yTrain'}, figureTitle="Testing | "); 
    % plot_partial_data(tuyRowTest, rNameTest, tTest, uTest(:,1:19), yTest, yLabels={'uTrain','yTrain'}, figureTitle="Testing | "); 
end

%% 2. load NODE parameters

nodeParametersStruct = load(nodeParametersFile,"nodeParameters");
nodeParameters = nodeParametersStruct.nodeParameters;

%% 3. Normalisation parameters

global yMean yStd dyMean dyStd
if normFromSaved
    load(normFile);
else
    [yMean, yStd, dyMean, dyStd, ~] = mean_std_from_data(tTest, yTest, tuyRowTest); 
end

%% 4. Testing
yPred = prediction(y0Test, tTest, uTest, tuyRowTest, vTest, nodeParameters);
if plotResult
    plot_data(tuyRowTest, rNameTest, tTest, yTest, yPred, yLabels={'Test','Pred'}, figureTitle="Testing | ");
    % plot_partial_data(tuyRowTest, rNameTest, tTest, yTest, yPred, yLabels={'Test','Pred'}, figureTitle="Testing | ");
end

