%% Train the network using a custom training loop with collocated dataset.
% For each iteration:
% * Construct a mini-batch data from the training data with the createMiniBatch_coll function.
% * Evaluate the loss/gradients using matlab dlfeval and the customised modelLoss_coll function.
% * Update the model parameters using the adamupdate function.
% * Update the training progress plot.

function neuralOdeParameters = collocation_training(tTrain, uTrain, yColl, dyColl, tuyRow, yRName, vol, neuralOdeParameters, opt)    
    arguments
        tTrain (:,1) double {mustBeNumeric};
        uTrain (:,:) double {mustBeNumeric};
        yColl (:,:) double {mustBeNumeric};
        dyColl (:,:) double {mustBeNumeric};
        tuyRow (:,2) {mustBeInteger};
        yRName (:,1) string;
        vol (1,:) double;
        neuralOdeParameters;

        % options for training process
        opt.numIter (1,1) {mustBePositive, mustBeInteger } = 100; 
        opt.perBatchSize (1,1) double {mustBePositive} = 0.2; % percentage of batch size in whol time steps
        opt.plotFrequency (1,1) {mustBePositive, mustBeInteger } = 50; 
        
        % options for Adam optimization.
        opt.gradDecay (1,1)  = 0.9;
        opt.sqGradDecay (1,1)  = 0.999;
        opt.learnRate (1,1)  = 0.01;
        
        % options for number of column of specified variables in U (u, y
        % are same at first 19 components)
        opt.colLComp (1,1) = 19; % col of last component in u and y
        opt.colO2 (1,1) = 8;     % col of O2 in u and y
        opt.colTemp (1,1) = 19;  % col of temp in u and y
        opt.colUQ (1,1) = 21;    % col of Q (flow) in u
        opt.colUKla (1,1) = 22;  % col of Kla in u
    end

    %% determine number of reactors
    nReactors = size(tuyRow,1);

    %% Initialize loss figure
    [fLoss, lineLossTrain] = plot_loss();

    %% Initialize the |averageGrad| and |averageSqGrad| parameters for the Adam solver.
    averageGrad = [];
    averageSqGrad = [];

    %% training loop    
    start = tic;
    fData = gobjects(1,nReactors);
    for i=1:nReactors
        fData(i) = figure;
    end
    
    for iter = 1:opt.numIter
        
        % Create batch 
        [uBatch, yBatch, dyBatch, uyRowBatch] = createMiniBatch_coll(opt.perBatchSize, uTrain, yColl, dyColl, tuyRow, shuffle=true);
    
        % Evaluate network and compute loss and gradients
        [loss,gradients] = dlfeval(@modelLoss_coll, uBatch, yBatch, dyBatch, uyRowBatch, vol, neuralOdeParameters, opt.colLComp, opt.colO2, opt.colTemp, opt.colUQ, opt.colUKla);
        
        % Update network 
        [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iter, ...
            opt.learnRate,opt.gradDecay,opt.sqGradDecay);
        
        % Plot loss
        titleTxt = "Collocation training | " + num2str(100*opt.perBatchSize,'%g%%') + " | Elapsed: " + string(duration(0,0,toc(start),Format="hh:mm:ss"));
        plot_loss_update(fLoss,lineLossTrain,loss,iter,titleTxt);
        
        % Plot data
        if mod(iter,opt.plotFrequency) == 0  || iter == 1

            % evaluate the NN 
            dyPred = model_coll(uTrain, yColl, tuyRow, vol, neuralOdeParameters, opt.colLComp, opt.colO2, opt.colTemp, opt.colUQ, opt.colUKla);
            plot_data(tuyRow, yRName, tTrain, dyColl, dyPred, f=fData, yLabels={'dyColl','dyPred'}, figureTitle="Collocation training | ",showRMSE=true,showR2=true,showUnit=true);
            drawnow
        end
    end
end

%% Collocation Create Mini-Batches Function

% The createMiniBatch function creates a batch of collocate y and dy to be 
% trained. shuffle input as a text "shuffle" will generate a BatchSize sequence 
% with each randomly selected pair, otherwise it will generate a randomly selected 
% first pair, followed by BatchSize seqence in its original order.

function [uBatch, yBatch, dyBatch, uyRowBatch] = createMiniBatch_coll(perBatchSize, u, y, dy, uyRow, opt)
    arguments
        perBatchSize (1,1) double {mustBeNumeric};
        u (:,:) double {mustBeNumeric};
        y (:,:) double {mustBeNumeric};
        dy (:,:) double {mustBeNumeric};
        uyRow (:,2) {mustBeInteger};

        opt.shuffle (1,1) logical {mustBeNumericOrLogical} = true;
    end

    nReactors = size(uyRow,1);
    nAllSteps = zeros(nReactors,1);
    nBatchSteps = zeros(nReactors,1);
    for i = 1:nReactors
        nAllSteps(i) = uyRow(i,2)-uyRow(i,1)+1;
        nBatchSteps(i) = floor(nAllSteps(i)*perBatchSize);
        if nBatchSteps(i) > nAllSteps(i)
            nBatchSteps(i) = nAllSteps(i);
        elseif nBatchSteps(i)<1
            nBatchSteps(i) = 1;
        end

    end
   
    uyRowBatch = zeros(size(uyRow));
    k=0;
    for i = 1:nReactors
        uyRowBatch(i,1) = k+1;
        uyRowBatch(i,2) = k+nBatchSteps(i);
        k=uyRowBatch(i,2);
    end    

    uBatch = dlarray(zeros(uyRowBatch(nReactors,2),size(u,2)));
    yBatch = dlarray(zeros(uyRowBatch(nReactors,2),size(y,2)));
    dyBatch = dlarray(zeros(uyRowBatch(nReactors,2),size(dy,2)));

    for i = 1:nReactors 
        if opt.shuffle
            s = randperm(nAllSteps(i), nBatchSteps(i));
        else
            s1 = randperm(nAllSteps(i) - nBatchSize(i), 1);
            s = s1:1:(s1+nBatchSteps(i)-1);
        end
        
        uBatch(uyRowBatch(i,1):uyRowBatch(i,2),:) = u(s+uyRow(i,1)-1,:);
        yBatch(uyRowBatch(i,1):uyRowBatch(i,2),:) = y(s+uyRow(i,1)-1,:);
        dyBatch(uyRowBatch(i,1):uyRowBatch(i,2),:) = dy(s+uyRow(i,1)-1,:);
    end
end

%% Collocation model function

% y is a Matrix! The model_coll function, which defines the neural network used 
% to make predictions. simply output matrix from the input matrix of the network 
% for the purpose to reuse the network structure.
function dyPred = model_coll(u, y, uyRow, vol, nodeParameters, colLComp, colO2, colTemp, colUQ, colUKla)

    nReactors = size(uyRow,1);

    dyPred = dlarray(zeros(size(y))); 

    for i=1:nReactors

        for j=uyRow(i,1):uyRow(i,2)
            y_u = u(j, 1:colLComp);
            temp_u = u(j, colTemp);
            Q_u = u(j, colUQ);
            kla_u = u(j, colUKla);
        
            dyNN =ode_model(y(j,:), nodeParameters);

            % ========== u input ==============
            
            % rate = in - out + reaction rate
            dy = Q_u*(y_u-y(j,:))/vol(i) + dyNN;
            
            % for oxygen, assume saturation concentration is 8 g/m3
            dy(colO2) = dy(colO2) + kla_u*(8-y(j,colO2));
        
            % for temperature, assume direct through
            dy(colTemp) = temp_u-y(j,colTemp);
       
            dyPred(j,:) = dy; 
        end
    end
end

%% Collocation Model Loss Function

% This function takes as inputs a batch of y and dy and net parameters.  it 
% computes the loss and the gradient of the loss with respect to the learnable 
% parameters of the network.

function [loss,gradients] = modelLoss_coll(u, y, dyTarget, uyRow, vol, nodeParameters, colLComp, colO2, colTemp, colUQ, colUKla)

    global yMean
    % Compute predictions.
    dyPred = model_coll(u, y, uyRow, vol, nodeParameters, colLComp, colO2, colTemp, colUQ, colUKla);
    dyTarget = dlarray(dyTarget);

    % Compute L1 loss.
    % loss = l1loss(dyPred, dyTarget, NormalizationFactor="all-elements",DataFormat="CB");

    % Scaled L1 loss (need yMean)
    yWeight = [1 yMean(2)/10 1 1 1 1 1 yMean(8)/10 1 1 yMean(11)/10 1 1 yMean(14)/10 yMean(15)/10 yMean(16)/10 1 1 1];
    % % yWeight = [1 1 1 1 1 1 1 yMean(8)/100 1 1 1 1 1 1 1 1 1 1 1];
    loss = l1loss(dyPred./yWeight,dyTarget./yWeight,NormalizationFactor="all-elements",DataFormat="CB");

    % Compute L2 loss.
    % loss = l2loss(dyPred, dy, NormalizationFactor="batch-size", DataFormat="CB");
    
    % Compute Huber loss.
    % loss = huber(dyPred, dy, NormalizationFactor="batch-size", DataFormat="CB");
    
    % Compute gradients.
    gradients = dlgradient(loss,nodeParameters);

end

