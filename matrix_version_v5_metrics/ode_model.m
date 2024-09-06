function dy = ode_model(y,theta)
% ODE Model
% y is a MATRIX! The |odeModel| function is the learnable right-hand side used 
% in the call to deep learning for collocation method and dlode45 for NODE method. 

global yMean yStd dyMean dyStd
y(y<0)=0;

idx = (yStd ~= 0);
y(:, idx) = (y(:, idx) - yMean(idx))./yStd(idx);

layerNames = fieldnames(theta); 
numTotalLayer = numel(layerNames);
for i = 1:numTotalLayer-1 
        y = gelu(y*theta.(layerNames{i}).Weights + theta.(layerNames{i}).Bias);
end 
y = y*theta.(layerNames{numTotalLayer}).Weights + theta.(layerNames{numTotalLayer}).Bias;
dy = y .*dyStd + dyMean;

end