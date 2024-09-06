%% Initialize yMean, yStd, dyMean, dyStd parameters for normalisation method from collocated dataset. 

% default keyword arguments:
% yColl:  (19 components) 
% dyColl: (19 components)
% fromSaved: false

function [yMean, yStd, dyMean, dyStd] = mean_std_from_collocated(yColl, dyColl)
    arguments
        yColl (:,:) double {mustBeNumeric};
        dyColl (:,:) double {mustBeNumeric};
    end

    yMean = mean(yColl,1);
    yStd = std(yColl,1,1);
    dyMean = mean(dyColl,1);
    dyStd = std(dyColl,1,1);
end