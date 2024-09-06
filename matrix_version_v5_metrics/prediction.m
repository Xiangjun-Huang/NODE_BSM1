%% Evaluate trained model, produce a prediction with a given initial condition by current neuralOdeParameters.

function y0Pred = prediction(y0, t, u, tuRow, vol, theta, opt)
    arguments
        y0 (:,:) double {mustBeNumeric};
        % y0Row (:,2) uint64;
        t (:,1) double {mustBeNumeric};
        u (:,:) double {mustBeNumeric};
        tuRow (:,2) uint64;
        vol (1,:) double;
        theta;
    
        % options for number of column of specified variables in t, U (u, y
        % are same at first 19 components)
        opt.colTTime (1,1) = 1   % col of time in t
        opt.colLComp (1,1) = 19; % col of last component in u and y
        opt.colO2 (1,1) = 8;     % col of O2 in u and y
        opt.colTemp (1,1) = 19;  % col of temp in u and y
        opt.colUQ (1,1) = 21;    % col of Q (flow) in u
        opt.colUKla (1,1) = 22;  % col of Kla in u
    end

    y0Pred = [];
    nReactors = size(tuRow,1);
    for i = 1:nReactors
        [yPred,~] = model_NODE_varied_step(y0(i,:), [1,1], t(tuRow(i,1):tuRow(i,2),:), ...
            u(tuRow(i,1):tuRow(i,2),:), [1,tuRow(i,2)-tuRow(i,1)+1], vol(i), ...
            theta, opt.colTTime, opt.colLComp, opt.colO2, opt.colTemp, opt.colUQ, opt.colUKla);
        % add y0 to yPred    
        y0PredThis = zeros(size(u(tuRow(i,1):tuRow(i,2),:),1),size(y0(i,:),2));
        y0PredThis(1,:) = y0(i,:);
        y0PredThis(2:tuRow(i,2)-tuRow(i,1)+1,:) = yPred;
        y0Pred = [y0Pred; y0PredThis];
    end
end
