%% NODE model function

function [yPred, yRowPred] = model_NODE_varied_step(y0, y0Row, t, u, tuRow, vol, theta, colTTime, colLComp, colO2, colTemp, colUQ, colUKla)

    nReactors = size(tuRow, 1);
    nBSize = zeros(nReactors,1);
    nNodeSteps = zeros(nReactors,1);
    yRowPred = zeros(size(tuRow));
    yk =0;
    for i=1:nReactors
        nBSize(i) = y0Row(i,2)-y0Row(i,1) +1;
        nNodeSteps(i) = round((tuRow(i,2)-tuRow(i,1)+1)/(y0Row(i,2)-y0Row(i,1)+1))-1;

        yRowPred(i,1) = yk+1;
        yRowPred(i,2) = yk+nBSize(i)*nNodeSteps(i);
        yk = yRowPred(i,2); 
    end

    yPred = dlarray(zeros(yRowPred(nReactors,2),size(y0,2)));   
    
    for i=1:nReactors
        for j=1:nBSize(i)
            
            tBatch = t(tuRow(i,1)+(j-1)*(nNodeSteps(i)+1):tuRow(i,1)+(j-1)*(nNodeSteps(i)+1)+nNodeSteps(i),:);
            uBatch = u(tuRow(i,1)+(j-1)*(nNodeSteps(i)+1):tuRow(i,1)+(j-1)*(nNodeSteps(i)+1)+nNodeSteps(i),:);
        
            t_u = tBatch(:, colTTime)- tBatch(1,colTTime);
            y_u = uBatch(:, 1:colLComp);
            temp_u = uBatch(:,colTemp);
            Q_u = uBatch(:, colUQ);
            kla_u = uBatch(:, colUKla);
            
            yP = dlode45(@odeModel,t_u,dlarray(y0(y0Row(i,1)+j-1,:)),theta,DataFormat="BC"); 
            yP = transpose(squeeze(yP));
            yP(yP<0) = 0.0; % nonNegtive restriction
            yPred(yRowPred(i,1)+(j-1)*(nNodeSteps(i)):yRowPred(i,1)+(j-1)*(nNodeSteps(i))+nNodeSteps(i)-1,:) = yP;  
        end
    end

    function dy = odeModel(t, y, theta)

    dyNN = ode_model(y,theta);

    % ========== uBatch input ==============
    
    y_u_t = interp1(t_u,y_u,t,'linear','extrap');
    temp_u_t = interp1(t_u,temp_u,t,'linear','extrap');
    Q_u_t = interp1(t_u,Q_u,t,'linear','extrap');
    kla_u_t = interp1(t_u,kla_u,t,'linear','extrap');
    
    % rate = in - out + reaction rate
    dy = Q_u_t*(y_u_t-y)/vol(i) + dyNN;
    
    % for oxygen, assume saturation concentration is 8 g/m3
    dy(colO2) = dy(colO2) + kla_u_t*(8-y(colO2));

    % for temperature, assume direct through
    dy(colTemp) = temp_u_t-y(colTemp);
    end
end