%% Train the network using a custom training loop with training dataset by NODE method.
% For each iteration:
% * Construct a mini-batch data from the training data with the createMiniBatch_NODE function.
% * Evaluate the loss/gradients using matlab dlfeval and the customised modelLoss_NODE function.
% * Update the model parameters using the adamupdate function.
% * Update the training progress plot.
% u , y has only 19 components

function neuralOdeParameters = NODE_training(t, u, y, tuyRow, yRName, vol, neuralOdeParameters, opt)    
    arguments 
        t (:,1) double {mustBeNumeric};
        u (:,:) double {mustBeNumeric};
        y (:,:) double {mustBeNumeric};
        tuyRow (:,2) {mustBeInteger};
        yRName (:,1) string;
        vol (1,:) double;
        neuralOdeParameters;

        % options for training process
        opt.numIter (1,1) {mustBePositive, mustBeInteger } = 1000; 
        opt.nNodeSteps (1,1) {mustBePositive, mustBeInteger } = 1;
        opt.perBatchSize (1,1) double {mustBePositive} = 0.2; % percentage of batch size in total size
        opt.plotFrequency (1,1) {mustBePositive, mustBeInteger } = 100;   
        
        % options for Adam optimization.
        opt.gradDecay (1,1)  = 0.9;
        opt.sqGradDecay (1,1)  = 0.999;
        opt.learnRate (1,1)  = 0.01;

        % options for number of column of specified variables in U (u, y
        % are same at first 19 components)
        opt.colTTime (1,1) = 1;  % col of time in t
        opt.colLComp (1,1) = 19; % col of last component in u and y
        opt.colO2 (1,1) = 8;     % col of O2 in u and y
        opt.colTemp (1,1) = 19;  % col of temp in u and y
        opt.colUQ (1,1) = 21;    % col of Q (flow) in u
        opt.colUKla (1,1) = 22;  % col of Kla in u
    end

    %% determine number of reactors
    nReactors = size(tuyRow,1);
    
    y0 = zeros(nReactors,size(y,2));
    for i=1:nReactors
        y0(i,:) = y(tuyRow(i,1),:);
    end

    %% Initialize loss figure    
    [fLoss, lineLossTrain] = plot_loss();

    %% Initialize the |averageGrad| and |averageSqGrad| parameters for the Adam solver    
    averageGrad = [];
    averageSqGrad = [];

    %% custom training by direct NODE approach
    
    start = tic;
    fData = gobjects(1,nReactors);
    for i=1:nReactors
        fData(i) = figure;
    end
    
    for iter = 1:opt.numIter
        
        % Create batch 
        [y0Batch, y0RowBatch, tBatch, uBatch, tuRowBatch, yBatch] = createMiniBatch_NODE(opt.perBatchSize, opt.nNodeSteps, t, u, y, tuyRow);
    
        % Evaluate network and compute loss and gradients
        [loss,gradients] = dlfeval(@modelLoss_NODE, y0Batch, y0RowBatch, tBatch, uBatch, tuRowBatch, yBatch, vol, neuralOdeParameters, ...
            opt.colTTime, opt.colLComp, opt.colO2, opt.colTemp, opt.colUQ, opt.colUKla);
        
        % Update network 
        [neuralOdeParameters,averageGrad,averageSqGrad] = adamupdate(neuralOdeParameters,gradients,averageGrad,averageSqGrad,iter,...
            opt.learnRate,opt.gradDecay,opt.sqGradDecay);
        
        % Plot loss   
        titleTxt = "NODE Training | Batch percent=" + num2str(100*opt.perBatchSize,'%g%%') + " | Step=" + num2str(opt.nNodeSteps) + ...
            " | Elapsed: " + string(duration(0,0,toc(start),Format="hh:mm:ss"));
        plot_loss_update(fLoss,lineLossTrain,loss,iter,titleTxt);
        
        % Plot predicted vs. real 
        if mod(iter, opt.plotFrequency) == 0  || iter == 1

            yPred = prediction(y0, t, u, tuyRow, vol, neuralOdeParameters);
            plot_data(tuyRow, yRName, t, y, yPred, f=fData, yLabels={'yTrain','yPred'}, figureTitle="NODE training | ", showRMSE=true,showR2=true,showUnit=true);
            drawnow
        end
    end
end

%% NODE Create Mini-Batches Function

function [y0Batch, y0RowBatch, tBatch, uBatch, tuRowBatch, yBatch] = createMiniBatch_NODE(perBSize, nNodeSteps, t, u, y, tuyRow)

    nReactors = size(tuyRow,1);

    tuRowBatch  = zeros(size(tuyRow));
    yRowBatch  = zeros(size(tuyRow));
    y0RowBatch = zeros(size(tuyRow));
    
    nAllSize = zeros(nReactors,1);
    nBSize = zeros(nReactors,1);

    uk=0;
    yk=0;
    y0k =0;
    for i = 1:nReactors
        nAllSize(i) = tuyRow(i,2)-tuyRow(i,1)+1;
        nBSize(i) = floor(nAllSize(i)*perBSize);
        if nBSize(i) > (nAllSize(i)-nNodeSteps)
            nBSize(i) = nAllSize(i) - nNodeSteps;
        elseif nBSize(i)<1
            nBSize(i) = 1;
        end

        tuRowBatch(i,1) = uk+1;
        tuRowBatch(i,2) = uk+nBSize(i)*(nNodeSteps+1);
        uk = tuRowBatch(i,2);

        yRowBatch(i,1) = yk+1;
        yRowBatch(i,2) = yk+nBSize(i)*nNodeSteps;
        yk = yRowBatch(i,2);

        y0RowBatch(i,1) = y0k+1;
        y0RowBatch(i,2) = y0k+nBSize(i);
        y0k = y0RowBatch(i,2);
    end 

    tBatch  = dlarray(zeros([tuRowBatch(nReactors,2) size(t,2)]));
    uBatch  = dlarray(zeros([tuRowBatch(nReactors,2) size(u,2)]));
    yBatch  = dlarray(zeros([yRowBatch(nReactors,2),size(y,2)]));    
    y0Batch = dlarray(zeros([y0RowBatch(nReactors,2), size(y,2)]));

    for i = 1:nReactors
        s = randperm(nAllSize(i) - nNodeSteps, nBSize(i));
        y0Batch(y0RowBatch(i,1):y0RowBatch(i,2),:) = y(s+tuyRow(i,1)-1,:);
        for m = 1:nBSize(i)
            tBatch(tuRowBatch(i,1)+(m-1)*(nNodeSteps+1),:) = t(s(m)+tuyRow(i,1)-1,:); 
            uBatch(tuRowBatch(i,1)+(m-1)*(nNodeSteps+1),:) = u(s(m)+tuyRow(i,1)-1,:);            
        
            for j = 1:nNodeSteps
                tBatch(tuRowBatch(i,1)+(m-1)*(nNodeSteps+1)+j,:) = t(s(m)+j+tuyRow(i,1)-1,:); 
                uBatch(tuRowBatch(i,1)+(m-1)*(nNodeSteps+1)+j,:) = u(s(m)+j+tuyRow(i,1)-1,:); 
                yBatch(yRowBatch(i,1)+(m-1)*(nNodeSteps)+j-1,:) = y(s(m)+j+tuyRow(i,1)-1,:); 
            end
        end
    end
end

%% NODE Model Loss Function

function [loss,gradients] = modelLoss_NODE(y0, y0Row, t, u, tuRow, yTarget, vol, neuralOdeParameters, colTTime, colLComp, colO2, colTemp, colUQ, colUKla)

    global yMean
    % Compute predictions.
    [yPred,~] = model_NODE_varied_step(y0, y0Row, t, u, tuRow, vol, neuralOdeParameters, colTTime, colLComp, colO2, colTemp, colUQ, colUKla);
    % [yPred,yRowPred] = model_NODE_varied_step(y0, y0Row, t, u, tuRow, vol, neuralOdeParameters, colTTime, colLComp, colO2, colTemp, colUQ, colUKla);

    % % Scaled L1 loss (need yMean)
    % yWeight = zeros(size(yRowPred,1),size(yPred,2));
    % yWeight(1,:)= [1 yMean(2)/10 1 1 1 1 1 yMean(8)/10 1 1 yMean(11)/10 1 1 yMean(14)/10 yMean(15)/10 yMean(16)/10 1 1 1];
    % yWeight(2,:)= [1 yMean(2)/10 1 1 1 1 1 1 1 1 yMean(11)/10 1 1 yMean(14)/10 yMean(15)/10 yMean(16)/10 1 1 1];
    % 
    % yW = zeros(size(yPred));
    % 
    % for i=1:size(yRowPred,1)
    %     for j=yRowPred(i,1):yRowPred(i,2)
    %         yW(j,:) = yWeight(i,:);
    %     end
    % end
    % 
    % loss = l1loss(yPred./yW,dlarray(yTarget)./yW,NormalizationFactor="all-elements",DataFormat="BC");

    % Compute L1 loss.
    % loss = l1loss(yPred,yTarget,NormalizationFactor="all-elements",DataFormat="BC");

    % Scaled L1 loss (need yMean)
    yWeight = [1 yMean(2)/10 1 1 1 1 1 yMean(8)/10 1 1 yMean(11)/10 1 1 yMean(14)/10 yMean(15)/10 yMean(16)/10 1 1 1];
    loss = l1loss(yPred./yWeight,dlarray(yTarget)./yWeight,NormalizationFactor="all-elements",DataFormat="BC");
        
    % Compute L2 loss.
    % loss = l2loss(X,targets,NormalizationFactor="batch-size",DataFormat="BC");
    
    % Compute Huber loss.
    % loss = huber(X,targets,NormalizationFactor="batch-size",DataFormat="BC");
    
    % Compute gradients.
    gradients = dlgradient(loss,neuralOdeParameters);

end

