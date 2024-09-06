%% Plot ty data function
% t: timestep
% tyRow: (nReactors,2), col: startRow, endrow, of reactors in yData structure
% rName: reactor names
% repeated tyData, (nTimesteps,nFeatures = 19 nComponents)

function plot_partial_data(tyRow, rName, t, yData, opt)
    arguments
        tyRow (:,2) uint64;
        rName (:,1) string;
        t (:,1) double {mustBeNumeric}; 
    end
    arguments (Repeating)
        yData (:,:) double {mustBeNumeric}; 
    end
    arguments
        opt.selectedCompNo = [2,4,5,6,8,10,14,16]; 
        opt.nReactors (1,1)  = size(tyRow,1);
        opt.nComponents (1,1)  = size(yData{1},2);
        opt.f = create_f(size(tyRow,1)); 
        opt.numRow (1,1) = 2;
        opt.numCol (1,1) = 4;
        opt.yLabels string = {"Train","Pred"}; % must be paired with the input ydata
        opt.xLables string = " d";
        opt.yName (1,19) string = {"S_I", "S_S", "X_I", "X_S", "X_{BH}", "X_{BA}", "X_P", "S_O", "S_{NO_3}",...
                 "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}", "S_{NO_2}", "S_{NO}", "S_{N_2O}", "S_{N_2}", "X_{BA2}", "Temp"};
        opt.yUnit (1,19) string = {"mg COD/l", "mg COD/l", "mg COD/l", "mg COD/l", "mg COD/l", "mg COD/l", "mg COD/l", ... 
            "mg COD/l", "mg N/l", "mg N/l", "mg N/l", "mg N/l", "mole ALK/m_3", "mg N/l", "mg N/l", "mg N/l", "mg N/l", "mg COD/l", "°C"};
        opt.figureTitle string ;
        opt.showUnit (1,1) logical {mustBeNumericOrLogical} = true;
        opt.showRMSE (1,1) logical {mustBeNumericOrLogical} = true;
        opt.showR2 (1,1) logical {mustBeNumericOrLogical} = true;
        opt.showWMAPE (1,1) logical {mustBeNumericOrLogical} = false;
    end

    nYData = numel(yData);

    strOverallRMSE = "";
    strOverallR2 = "";
    strOverallWMAPE = "";
    if nYData >= 2
        yTrue = yData{1}(:,:);
        yPred = yData{2}(:,:);
        if opt.showRMSE
            % calcuate RMSE of first two inputs of ydata only            
            strOverallRMSE = sprintf('%s %.2f ', " | Overall RMSE=", rmse(yTrue,yPred,'all'));
        end
        if opt.showR2
            % calcuate R2 of first two inputs of ydata only            
            strOverallR2 = sprintf('%s %.1f%% ', " | Overall R^2=", 100*(1-norm(yTrue-yPred)^2/norm(yTrue-mean(yTrue))^2));
        end
        if opt.showWMAPE
            % calcuate wMAPE of first two inputs of ydata only            
            strOverallWMAPE = sprintf('%s %.1f%% ', " | Overall wMAPE=", 100*(sum(abs(yTrue-yPred))/sum(abs(yTrue))));
        end
    end

    numPlot = length(opt.selectedCompNo);

    for i = 1:opt.nReactors
        f = figure(opt.f(i));
        f.Position = [0 200 1539 260];
        colororder(["#0072BD" "#A2142F" "#EDB120" ]);
        linestyles = ["-","-.",":"];
        linestyleorder(linestyles);
        tiledlayout(opt.numRow,opt.numCol,TileSpacing="compact",Padding="compact");
        for k = 1:numPlot
            nexttile
            hold on
            for j = 1:nYData
                plot(t(tyRow(i,1):tyRow(i,2),1), yData{j}(tyRow(i,1):tyRow(i,2),opt.selectedCompNo(k)), LineWidth=1)
            end
            hold off
                        
            strSubTitle = opt.yName(opt.selectedCompNo(k));
            if nYData >= 2
                yIKTrue = yData{1}(tyRow(i,1):tyRow(i,2),opt.selectedCompNo(k));
                yIKPred = yData{2}(tyRow(i,1):tyRow(i,2),opt.selectedCompNo(k));
                if opt.showRMSE
                    strSubTitle = strSubTitle + " | RMSE=" + sprintf('%.2f', rmse(yIKTrue, yIKPred,"all"));
                end            
                if opt.showR2
                    if norm(yIKTrue-mean(yIKTrue))~=0 
                        strSubTitle = strSubTitle + " | R^2=" + sprintf('%.1f%%', 100*(1-norm(yIKTrue-yIKPred)^2/norm(yIKTrue-mean(yIKTrue))^2));
                    else
                        strSubTitle = strSubTitle + " | R^2=100.0%";
                    end
                end
                if opt.showWMAPE
                    strSubTitle = strSubTitle + " | wMAPE=" + sprintf('%.1f%%', 100*(sum(abs(yIKTrue-yIKPred))/sum(abs(yIKTrue))));
                end
            end
            title(strSubTitle);

            if upper(extractBefore(rName(i)," ")) == upper("Dynamic")
                xlmStep = 60;
            else
                xlmStep = 1;
            end  
            xlmMin = floor(t(tyRow(i,1),1));
            xlmMax = ceil(t(tyRow(i,2),1));
            xlim([xlmMin,xlmMax]);
            xticks(xlmMin:xlmStep:xlmMax);

            xtl = xticklabels;
            xtl{end} = xtl{end}+opt.xLables;
            xticklabels(xtl);
            if opt.selectedCompNo(k) == 1 || opt.selectedCompNo(k) == 19
                ytickformat('%.1g');
            end
            if opt.showUnit
                ylabel(opt.yUnit(opt.selectedCompNo(k)))
            end
            grid on
        end

        % show legend in the bottom right of the figure
        if i==1
            lgd = legend(opt.yLabels,'Orientation','horizontal');
            lgd.Position(1) = 0.05;
            lgd.Position(2) = 0.89;
        end

        strRMSE = "";
        strR2 = "";
        strWMAPE = "";
        if nYData >= 2
            yITrue = yData{1}(tyRow(i,1):tyRow(i,2),:);
            yIPred = yData{2}(tyRow(i,1):tyRow(i,2),:);
            if opt.showRMSE
                % calcuate RMSE of first two inputs of ydata only for reactor i          
                strRMSE = sprintf('%s %.2f ', " | RMSE=", rmse(yITrue, yIPred,'all'));
            end
            if opt.showR2 
                % calcuate R2 of first two inputs of ydata only for reactor i           
                strR2 = sprintf('%s %.1f%% ', " | R^2=",  100*(1-norm(yITrue-yIPred)^2/norm(yITrue-mean(yITrue))^2)); 
            end
            if opt.showWMAPE
                % calcuate wMAPE of first two inputs of ydata only for reactor i          
                strWMAPE = sprintf('%s %.1f%% ', " | wMAPE=", 100*(sum(abs(yITrue-yIPred))/sum(abs(yITrue))));
            end
        end

        strTitle = opt.figureTitle + rName(i) + strRMSE + strR2 + strWMAPE + strOverallRMSE + strOverallR2 + strOverallWMAPE; 
        sgtitle(strTitle);
    end
end

function f = create_f(n)
    f = gobjects(1,n);
    for i = 1:n
        f(i) = figure;
    end
end