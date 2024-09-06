%% scan file for data
% file structure: time series matrix, column definition is shown below:
% day, input 19 components, input TSS, input flow, input Kla, state 19 components.

% based on the above selections, generate training data
% t matrix (nTimesteps)
% u matrix: input, (nTimesteps, nFeatures = 22) column: 19 Components, TSS, flow, Kla 
% y matrix: state, (nTimesteps, nComponents = 19)
% Components column: Si, Ss,Xi,	Xs,	XBH, XAOB, XP, SO, SNO3, SNH, SND, XND, SALK, SNO2, SNO, SN2O, SN2, XNOB, Temperature
% y0 matrix: state, first line of y
% tuyRow matrix (nReactors, 2), columns: start row, end row;
% rName: train reactor names
% v: train reactor volume

function [t, u, y, y0, tuyRow, r, v] = get_mat_data(info, opt)
    arguments
        info (:,4) cell;
        opt.loadPath string = "../data/"
    end
    
    vol = [1000 1000 1333 1333 1333];
    rIndex = find([info{:,4}]==true);
    nReactors = length(rIndex);

    t = [];
    u = [];
    y = [];
    y0 = [];
    tuyRow = [];
    r = [];
    v = [];

    nRow = 0;
    
    for i = 1:nReactors

        rFileName = info{rIndex(i),1};
        rTxt = extractBetween(rFileName,"_r",".mat");
        rNum = str2double(rTxt);
        rType = extractBefore(rFileName,"_r");
        rType = upper(extractBefore(rType,2)) + extractAfter(rType,1);
        if upper(rType) == upper("Dyn")
            rName = "Dynamic | Reactor " + rTxt;
        else
            rName = rType + " weather | Reactor " + rTxt;
        end

        rFileStruct = load(opt.loadPath + rFileName);
        rFileStructNames = fieldnames(rFileStruct);
        rMat = rFileStruct.(rFileStructNames{1});
            
        tFull = rMat(:,1); 
        [row, ~] = find(tFull>=info{rIndex(i),2} & tFull<info{rIndex(i),3});
        startRow = row(1);
        endRow = row(end);

        t = [t; rMat(startRow:endRow,1)];
        u = [u; rMat(startRow:endRow,2:23)];
        y = [y; rMat(startRow:endRow,24:42)];
        y0= [y0; rMat(startRow,24:42)];
        tuyRow = [tuyRow; [nRow+1,size(y,1)]];
        nRow = size(y,1);
        r = [r; rName];
        v = [v; vol(rNum)];
    end
end