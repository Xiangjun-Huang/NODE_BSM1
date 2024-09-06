%% NODE with exogenous excitment for ASMG1 model on BSM1 plant by collocation training and further NODE training

% Xiangjun Huang, Brunel University London, Jun 2024

%% 0. Start from scratch
clear;
clc;
close all;

%% 1. Scan training data 

%             file,       day~ day, if train? 
dataInfo = {"dry_r1.mat",   0,  7,   true; 
            "dry_r2.mat",   0,  7,   true; 
            "dry_r3.mat",   0,  7,   true; 
            "dry_r4.mat",   0,  7,   true; 
            "dry_r5.mat",   0,  7,   true; 
            "rain_r1.mat",  0,  14,  false;
            "rain_r2.mat",  0,  14,  false;
            "rain_r3.mat",  0,  14,  false;
            "rain_r4.mat",  0,  14,  false;
            "rain_r5.mat",  0,  14,  false;
            "storm_r1.mat", 0,  14,  false;
            "storm_r2.mat", 0,  14,  false;            
            "storm_r3.mat", 0,  14,  false;
            "storm_r4.mat", 0,  14,  false;
            "storm_r5.mat", 0,  14,  false;};

[tTrain, uTrain, yTrain, y0Train, tuyRowTrain, rNameTrain, vTrain] = get_mat_data(dataInfo); % default folder: "../data_ref/"

set(groot, 'defaultFigureWindowState', 'maximized'); % maximum figure windows
% plot_data(tuyRowTrain, rNameTrain, tTrain, uTrain(:,1:19), yTrain, yLabels={'uTrain','yTrain'}, figureTitle="Training | "); 
% plot_partial_data(tuyRowTrain, rNameTrain, tTrain, uTrain(:,1:19), yTrain, yLabels={'uTrain','yTrain'}, figureTitle="Training | "); 

%% 3. Initialize NODE network parameters

nodeParameters = initialize_NODE_parameters(); % options: stateSize=19, hiddenSize = 50

%% 4. Normalisation parameters

global yMean yStd dyMean dyStd

% load ../data_ref/20240620_fixed_globalMeanStd.mat yMean yStd dyMean dyStd;

[yMean, yStd, dyMean, dyStd, dyGrad] = mean_std_from_data(tTrain, yTrain, tuyRowTrain); 

% save data/20240620_dry_0_7_globalMeanStd.mat yMean yStd dyMean dyStd;
% save data/dyGrad.mat dyGrad;

%% 5. Collocation training

% Calculate collocated pair (yColl, dyColl).
[yColl, dyColl] = collocate_all_reactors(tTrain,yTrain,tuyRowTrain); % option: kernelStr = "EpanechnikovKernel"

% plot_data(tuyRowTrain, rNameTrain, tTrain, yTrain, yColl, yLabels={'yTrain','yColl'}, figureTitle = ""); 
% plot_data(tuyRowTrain, rNameTrain, tTrain, dyGrad, dyColl, yLabels={'dyGrad','dyColl'}, showUnit=true, showRMSE=true, showMAE=true, ...
%           yUnit={"mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg COD/l/d", "mg N/l/d", ...
%           "mg N/l/d", "mg N/l/d", "mg N/l/d", "mole ALK/m_3/d", "mg N/l/d", "mg N/l/d", "mg N/l/d", "mg N/l/d", "mg COD/l/d", "°C/d"});

[yMean, yStd, dyMean, dyStd] = mean_std_from_collocated(yColl, dyColl); 

% Collocation training 
nodeParameters = collocation_training(tTrain,uTrain,yColl,dyColl,tuyRowTrain,rNameTrain,vTrain,nodeParameters);

%% 6. Further direct NODE training

nodeParameters = NODE_training(tTrain, uTrain, yTrain, tuyRowTrain, rNameTrain, vTrain, nodeParameters);
% save data/20240620_dry_0_7_nodeParameters.mat nodeParameters;
