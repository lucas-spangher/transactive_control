% This is the code for clustering the profile library that contains
% everything (across consumers and days)

%%%%%%%%% INPUT
% profLib: Profile Library (n*m matrix. n is the number of daily load shapes and there are m-3 features for load shapes(e.g., 96 for 15-min resolution data). Last three columns are the load signature: house ID, day of the year, datatype (aggregate or appliance level))
% type: Clustering method: type = 1 (SOM), type = 2: Two stage
% (k-means-hierarchical), type 3 = k-means
% norm: Normalization option: 1 = perform normalization, 2 = don't perform
% normalization
% filt = Moving average filter: 1 = perform filtering, 2 = don't perform
% filtering

%%%%%%%%% OUTPUT
% dataTempLib = clustering output (last column in the cluster index)
% dataTempSignLib = clustering output with load shape signature and unique identifers for information retrieval(last four columns are house ID, day of the year, and datatype, and cluster index)


function [dataTempLib, dataTempSignLib,dataTemp, dataTempSign, eval] = clusterEverything(profLib,type,kRange,norm,filt)
close all;

lastTime = 96; % Profile resolution

consumInfoAll = [];

%%%%%%%%% Divide all profiles into L,M,H energy level

for tt = 1:size(profLib,1)
    consum=(24/lastTime)*trapz(profLib(tt,1:lastTime));
    consumInfoAll = [consumInfoAll;consum,tt];
end


firstQuart = quantile(consumInfoAll(:,1),0.25);
thirdQuart = quantile(consumInfoAll(:,1),0.75);

firstIdx = find(consumInfoAll(:,1)<firstQuart);
thirdIdx = find(consumInfoAll(:,1)>thirdQuart);
secondIdx = setxor([1:size(profLib)],[firstIdx;thirdIdx]);

C{1} = profLib(firstIdx,:);
C{2} = profLib(secondIdx,:);
C{3} = profLib(thirdIdx,:);

for iii = 1:3
    profLibTar = C{iii};
    lastTime = 96;
    
    % Check normalization option
    %%%%%%%%%%%%
    if norm==1
        
        for i = 1:size(profLibTar,1)
            profLibTar(i,1:lastTime) = profLibTar(i,1:lastTime)./max(profLibTar(i,1:lastTime));
        end
    end
    %%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Removing empty observations for appliance data
    if profLibTar(1,end)~= 3
        thre = 0.3; % consumption threshold for removing the observation from clustering
        idxRemove = []; % observations that should be removed (lower than threshold)
        
        for i = 1:size(profLibTar,1)
            if max(profLibTar(i,1:lastTime))<thre
                idxRemove = [idxRemove;i];
            end
        end
        
        profLibTar(idxRemove,:) = []; % cleaning the data for clustering (removing non-usage days)
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    profLibCop = profLibTar;
    % Check filtering option
    %%%%%%%%%%%%
    if filt==1
        
        filtData = [];
        windowSize = 4; % Filter parameter
        b = (1/windowSize)*ones(1,windowSize);
        a = 1;
        
        for n = 1 : size(profLibTar,1)
            filtData = [filtData;filter(b,a,profLibTar(n,1:lastTime))];
        end
        
        profLibTar = []; profLibTar = [filtData,profLibCop(:,end-2:end)];
        
        
        %     d1 = designfilt('lowpassiir','FilterOrder',12, ...
        %     'HalfPowerFrequency',0.15,'DesignMethod','butter');
        %
        %
        %     filtData = [];
        %
        %
        %     for n = 1 : size(profLib,1)
        %         filtData = [filtData;filtfilt(d1,profLib(n,1:lastTime))];
        %     end
        %
        %     profLib = []; profLib = [filtData,profLibCop(:,end-2:end)];
        
        
    end
    
    %%%%% if flitering is performed, remove first (window) elements from
    %%%%% clustering
    if filt==1
        profLibTar(:,1:windowSize-1) = [];
        lastTime = lastTime - (windowSize-1);
    end
    
    
    % Clustering type: SOM
    if type == 1
        
        eval = [];
        for kAvg = kRange(1):kRange(end)
            dataTemp = mySOM(profLibTar(:,1:lastTime),kAvg); % select the clustering method
            
            % DaviesBouldin
            evaDB = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'DaviesBouldin');
            % silhouette
            evaS = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'silhouette');
            % CalinskiHarabasz (select the maximum)
            evaCH = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'CalinskiHarabasz');
            
            SSE = myInternalValidation(dataTemp(:,end),dataTemp(:,1:end-1),kAvg);
            
            eval = [eval;evaDB.CriterionValues,evaS.CriterionValues,evaCH.CriterionValues, SSE]
        end
        
        % Clustering type: SOM
    else if type ==2
            eval = [];
            for kAvg = kRange(1):kRange(end)
                clustLib = [];
                
                theta = 10; % Range for theta threshold
                %          theta = 18; % Range for theta threshold
                
                kLib = []; % k-means result for 1st stage
                for j = 1:length(theta)
                    dataTemp = myKmeans(profLibTar(:,1:lastTime),kAvg);
                    dataTempSign = [dataTemp(:,1:end-1),profLibTar(:,end-2:end),dataTemp(:,end)]; % Including ID info
                    
                    
                    stopFlag = false;
                    cnt = 0;
                    output = [];
                    
                    while ~stopFlag
                        preTarg = dataTemp(dataTemp(:,end) < cnt+1,1:end);
                        preTargSign = dataTempSign(dataTempSign(:,end) < cnt+1,end-3:end-1);
                        targ = dataTemp(dataTemp(:,end) == cnt+1,1:end-1);
                        targSign = dataTempSign(dataTempSign(:,end) == cnt+1,end-3:end-1);
                        targComplement = dataTemp(dataTemp(:,end) > cnt+1,1:end);
                        targComplementSign = dataTempSign(dataTempSign(:,end) > cnt+1,end-3:end-1);
                        
                        if size(targ,1)>1
                            clustCent = mean(targ);
                        else
                            clustCent = targ;
                        end
                        
                        sumClustCent = sum(clustCent.^2);
                        sumTemp = 0;
                        
                        for i = 1:size(targ,1)
                            
                            for k =1:lastTime
                                sumTemp = sumTemp + (targ(i,k)- clustCent(k))^2;
                            end
                            
                        end
                        
                        %%%%%%%%%%%% CAUTION !!!! THETA(1)
                        if sumTemp > theta(j)*sumClustCent
                            newClust = myKmeans(targ(:,1:lastTime),2);
                            newClust(:,end) = newClust(:,end) + cnt;
                            targComplement(:,end) = targComplement(:,end) + 1;
                            dataTemp = [preTarg;newClust;targComplement];
                            dataTempSign = [preTarg(:,1:end-1),preTargSign,preTarg(:,end); newClust(:,1:end-1),targSign,newClust(:,end);targComplement(:,1:end-1),targComplementSign,targComplement(:,end)];
                        else
                            cnt = cnt + 1;
                            if cnt == max(unique(dataTemp(:,end)))
                                stopFlag = true;
                            end
                        end
                        
                    end
                    
                    kLib = [kLib;length(unique(dataTemp(:,end)))];
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                end
                % 2nd stage: Hierarchical clustering
                dataTempNew = [];
                
                % Find centroid of k-means clustering for 2nd stage
                for i = 1:length(unique(dataTemp(:,end)))
                    temp = dataTemp(dataTemp(:,end)==i,:);
                    
                    if size(temp,1)>1
                        dataTempNew = [dataTempNew;mean(temp)];
                    else
                        dataTempNew = [dataTempNew;temp];
                        
                    end
                end
                
                Z = linkage(dataTempNew(:,1:lastTime),'ward','euclidean');
                
                dissimilarity = pdist(dataTempNew(:,1:lastTime),@dtwDist);
                Z = linkage(dissimilarity,'weighted');
                
                c = cluster(Z,'maxclust',kAvg);
                
                
                
                dataH = [dataTempNew,c];
                for i = 1:size(dataTemp,1)
                    
                    idx = dataTemp(i,end);
                    ans = find(dataH(:,end-1)==idx);
                    newIdx = dataH(ans,end);
                    dataTemp(i,end) = newIdx;
                    
                end
                
                dataTempSign = [dataTemp(:,1:lastTime),dataTempSign(:,end-3:end-1),dataTemp(:,end)];
                
                % DaviesBouldin
                evaDB = evalclusters(dataTempSign(:,1:lastTime),dataTempSign(:,end),'DaviesBouldin');
                % silhouette
                evaS = evalclusters(dataTempSign(:,1:lastTime),dataTempSign(:,end),'silhouette');
                % CalinskiHarabasz (select the maximum)
                evaCH = evalclusters(dataTempSign(:,1:lastTime),dataTempSign(:,end),'CalinskiHarabasz');
                
                SSE = myInternalValidation(dataTemp(:,end),dataTemp(:,1:end-1),kAvg);
                
                eval = [eval;evaDB.CriterionValues,evaS.CriterionValues,evaCH.CriterionValues,SSE]
                
            end
            % Clustering type : k-means
        else if type==3
                
                eval = [];
                for kAvg = kRange(1):kRange(end)
                    dataTemp = myKmeans(profLibTar(:,1:lastTime),kAvg);
                    
                    % DaviesBouldin
                    evaDB = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'DaviesBouldin');
                    % silhouette
                    evaS = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'silhouette');
                    % CalinskiHarabasz (select the maximum)
                    evaCH = evalclusters(profLibTar(:,1:lastTime),dataTemp(:,end),'CalinskiHarabasz');
                    
                    SSE = myInternalValidation(dataTemp(:,end),dataTemp(:,1:end-1),kAvg);
                    
                    
                    eval = [eval;evaDB.CriterionValues,evaS.CriterionValues,evaCH.CriterionValues,SSE]        
                    
                end
                
            else
                
                eval = [];
                for kAvg = kRange(1):kRange(end)
                    
                    Z = linkage(profLibTar(:,1:lastTime),'weighted','euclidean');
                    
                    c = cluster(Z,'maxclust',kAvg);
                    dataTemp = [profLibTar(:,1:lastTime),c];
                    
                    
                    % DaviesBouldin
                    evaDB = evalclusters(dataTemp(:,1:lastTime),dataTemp(:,end),'DaviesBouldin');
                    % silhouette
                    evaS = evalclusters(dataTemp(:,1:lastTime),dataTemp(:,end),'silhouette');
                    % CalinskiHarabasz (select the maximum)
                    evaCH = evalclusters(dataTemp(:,1:lastTime),dataTemp(:,end),'CalinskiHarabasz');
                    
                    SSE = myInternalValidation(dataTemp(:,end),dataTemp(:,1:end-1),kAvg);
                         
                    eval = [eval;evaDB.CriterionValues,evaS.CriterionValues,evaCH.CriterionValues,SSE]
                    
                end
                
            end
            
        end
            
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Sort cluster number based on quantity (1st cluster with highest number of observations)
    
    if type~=2
        dataTempSign = [dataTemp(:,1:end-1),profLibCop(:,lastTime+1:end),dataTemp(:,end)];
    end
    
    
    clusInfo = [];
    
    for i = 1:length(unique(dataTemp(:,end)))
        
        temp = dataTemp(dataTemp(:,end)==i,1:lastTime);
        countNo = size(temp,1);
        clusInfo = [clusInfo; i,countNo];
        
    end
    clusInfo = sortrows(clusInfo,-2);
    clusInfo = [clusInfo,[1:length(unique(dataTemp(:,end)))]'];
    
    
    dataTemp2=[]; dataTempSign2 = [];
    for j = 1:length(unique(dataTemp(:,end)))
        temp = dataTemp(dataTemp(:,end)==j,1:end);
        temp2 = dataTempSign(dataTempSign(:,end)==j,1:end);
        idx=clusInfo(find(clusInfo(:,1)==j),3); temp(:,end)=idx; temp2(:,end)=idx
        dataTemp2 = [dataTemp2;temp];
        dataTempSign2 = [dataTempSign2;temp2];
        
    end
    dataTemp = dataTemp2;
    dataTempSign = dataTempSign2;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    %%%%%%%%%%%%%% Plot
    for i = 1:length(unique(dataTemp(:,end)))
        
        if iii == 1
            figure(1);
        elseif iii == 2
            figure(4)
        else
            figure(7)
        end
        
        subplot(2,ceil(kAvg/2),i);
        imagesc(dataTemp(dataTemp(:,end)==i,1:lastTime))
        colorbar
        
        %%%% Change color map
        maxVal = max(max(dataTemp(dataTemp(:,end)==i,1:lastTime)));
        newmap = jet;
        ncol = size(newmap,1);           %how big is it?
        caxis([0, 0.5*maxVal]);
        colormap(newmap);                %activate it
        
        %%%% Change color map
        temp = dataTemp(dataTemp(:,end)==i,1:lastTime);
        centroid = temp; % cluster centroid
        
        if size(temp,1)>1
            centroid = mean(temp);
        end
        
        temp = dataTemp(dataTemp(:,end)==i,1:lastTime);
        
        centroid = temp; % cluster centroid
        if size(temp,1)>1
            centroid = mean(temp);
        end
        
        if iii == 1
            figure(2);
        elseif iii == 2
            figure(5)
        else
            figure(8)
        end
        
        subplot(2,ceil(kAvg/2),i);
        plot1 = plot(temp');
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if size(temp,1)<50
            col = 0.08;
        elseif size(temp,1)<200
            col = 0.04;
        else
            col = 0.015;
        end
        
        set(gca,'XTick',[0; 24;  48;  72;  96])
        set(gca,'XTickLabel',[0;6;12;18;24])
        set(gca,'fontsize',14)
        xlabel('Hour')
        ylabel('Power(kW)')
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for j = 1:size(plot1,1)
            plot1(j).Color=[0,0,0,col];
        end
        
        hold on;
        plot(centroid','r','LineWidth',2.5);
        title(num2str(i));
        
        
        if iii == 1
            figure(3);
        elseif iii == 2
            figure(6)
        else
            figure(9)
        end
        
        subplot(4,ceil(kAvg/4),i);
        h = boxplot(temp)
        
        %    Without outlier
        h = boxplot(temp,'symbol','')
        
        set(gca,'XTick',[0; 24;  48;  72;  96])
        set(gca,'XTickLabel',[0;6;12;18;24])
        
        xlabel('Hour')
        ylabel('Power(kW)')
        
    end
    
    dataTempLib{iii} = dataTemp;
    dataTempSignLib{iii} = dataTempSign;
end
end
