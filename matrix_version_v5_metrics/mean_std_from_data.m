%% Initialize yMean, yStd, dyMean, dyStd parameters for normalisation method from train data by difference quotient  
% estimate mean and sd of y and dy of NODE model based on state data y

% default keyword arguments:
% t: timestep
% y: (19 components)
% tyRow: (nReactors,2)

function [yMean, yStd, dyMean, dyStd, dyGrad] = mean_std_from_data(t, y, tyRow)
    arguments
        t (:,1) double;
        y (:,:) double;
        tyRow (:,2) uint64; 
    end
   
    nReactors = size(tyRow,1);
    
    yMean = mean(y, 1);
    yStd = std(y,1,1);

    dyGrad = zeros(size(y));
    for i=1:nReactors 
        [~, dy] = gradient(y(tyRow(i,1):tyRow(i,2), :));
        dy = dy./gradient(t(tyRow(i,1):tyRow(i,2), 1));
        
        dyGrad(tyRow(i,1):tyRow(i,2),:) = dy;
    end
    dyMean = mean(dyGrad, 1);
    dyStd = std(dyGrad,1,1);
end