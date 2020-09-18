%%% This code gets the entire clusterd data as the input and returns the semantic features.
%%% 1) plots the average of clusters
%%% 2) Finds the peak of each cluster/ finds the location of peak/ find the
%%% magnitude (high, medium, low)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT
%%% Cluster library (dataTempSignLib obtained from clusterEverything.m)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT
%%% CL = Cell structure with each element containing semantic features for each cluster


%%%%%%%%%%%%%% This code labels the load shape according to:
%%% 1: Level of consumption (L,M,H)> low, medium, high
%%% 2: Peak distribution pattern (unimodal, bimodal, multimodal)
%%%	3: Temporal peak assignment (M, N, AN, E, NT, MN)> morning, noon,
%%%	afternoon, evening, night, midnight
%%% 4: Intensity of peak consumption (NS,MS,S)> not significant, moderately
%%% significant, significant
%%% 5: Duration of peak consumption (SH,H,MH)> subhour, hour, multihour


% Performance metrics: errLib, EMDLib

close all; clc; clear;

CL = {}; % CL is the code library

%%%%% 1st Feature: Consumption level: L,M,H is already calculated in
%%%%% dataTempSignLib

%%%%% 2nd Feature: Distribution pattern: Calculated based on the number of
%%%%% peaks (columns in peakInfo)

%%%%% 3rd Feature: Temporal peak assignment: morning = 6-11/ noon:11-13/ afternoon:13-17/ evening:17-20/night:20-24/midnight:24-6 
%%%%% Timebins are arbitrarily defined here and can be adjusted based on user
%%%%% preference
mT = 24:43; nT = 44:55; anT = 56:67; eT = 68:79; ntT = 80:96; mnT = 1:23;

%%%%% 4th Feature: Intensity of peak consumption (NS,MS,S)> The value of
%%%%% peak (first row of peakInfo) is compared with Mu+-nSigma of the
%%%%% average of all profiles associated with the cluster


%%%% 5th Feature: Duration of peak consumption (values presented for 15-min data)
threLow = 8; % low width 
threMid = 16; % medium width 


% Load clustering results (from clusterEverything.m)
load('G:\My Drive\PhD Research\Power Data & Code\i3CE - Extended\Result\clusSOM_JulAug_3.mat');

for iii = 1:size(dataTempLib,2)
    
    
CLgroup = {}; % code library for each group

dataTempSign = dataTempSignLib{iii};
noClus = unique(dataTempSign(:,end)); % no of clusters
avgClus = []; % library containing cluster centroid
lastTime=96; % number of datapoint in each load shape (e.g. 96 for 15-min data)

for ii = 1:numel(noClus)
   
    temp = dataTempSign(dataTempSign(:,end)==ii,:);
    avgClus = [avgClus;mean(temp(:,1:96),1)];
    
end


%%%%%%%% plot cluster centroid
leg = {};
figure;
col = jet(size(avgClus,1));
for ii = 1:size(avgClus,1)
    plot(avgClus(ii,:),'color',col(ii,:)); hold on;
    leg = [leg;num2str(ii)];
end
legend(leg)



peakInfo = {};
%%%%%%%%% find peaks 
for ii = 1:size(avgClus)
    
[pks,locs,w,p] = findpeaks(avgClus(ii,:));   
peakInfo{ii} = [pks;locs;w;p];    % pks is the value at peak point; locs is the location; w is the width;
% p: Peak prominences, returned as a vector of real numbers. 
% The prominence of a peak is the minimum vertical distance that the signal must descend on either side of the peak before either climbing back to a level higher than the peak or reaching an endpoint. See Prominence for more information.

end

%%%%% modify peaks; set prominence threshold to accept or reject peak
promThre1 = 0.06; % prominence threshold group L (low energy)
promThre2 = 0.15; % prominence threshold group M (medium energy)
promThre3 = 0.5; % prominence threshold group H (high energy)
peakThre1 = 1; % magnitude threshold group L
peakThre2 = 4; % magnitude threshold group M

for ii = 1:size(peakInfo,2)
    rmvIdx = [];
    
    if max(peakInfo{1,ii}(1,:))<peakThre1
        promThreTar =  promThre1;
    elseif max(peakInfo{1,ii}(1,:))<peakThre2
        promThreTar = promThre2;
    else
        promThreTar = promThre3;        
    end
        

    for jj = 1:size(peakInfo{1,ii},2)
        if peakInfo{1,ii}(4,jj) < promThreTar
            rmvIdx = [rmvIdx;jj];
        end 
    end
    peakInfo{1,ii}(:,rmvIdx) = [];  
end



xRange = 1:lastTime;

% for ii = 1:size(avgClus)
%     figure;
%     plot(xRange,avgClus(ii,:),xRange(peakInfo{1,ii}(2,:)),peakInfo{1,ii}(1,:),'or')
%     title(num2str(ii))
% end

peakInfoLib{iii} = peakInfo; 


%%%%%%%%%%%%% Extract features for labeling
for jjj = 1:size(peakInfo,2)
    
    % 1st feature: consumption level (1, 2, 3 stands for low, medium, high)
    if iii==1
        CLgroup{1,jjj}(1,1)=1;
    elseif iii==2
        CLgroup{1,jjj}(1,1)=2;
    else
        CLgroup{1,jjj}(1,1)=3;
    end
    
     % 2nd feature: peak number (0, 1, 2, 3 stands for no mode, unimodal, bimodal, multimodal)
     if size(peakInfo{1,jjj},2)==0
         CLgroup{1,jjj}(2,1)=0;
     elseif size(peakInfo{1,jjj},2)==1
        CLgroup{1,jjj}(2,1)=1;
    elseif size(peakInfo{1,jjj},2)==2
        CLgroup{1,jjj}(2,1)=2;
    else
        CLgroup{1,jjj}(2,1)=3;
    end
    
    % 3rd feature: temporal peak assignment (1:5 stands for midnight, morning, noon,
	% afternoon, evening, night)
    
    infThirdFt = [];
    for kkk = 1:size(peakInfo{1,jjj},2)
        idxLoc = peakInfo{1,jjj}(2,kkk);
        locRes1 = [ismember(idxLoc,mnT);ismember(idxLoc,mT); ismember(idxLoc,nT);ismember(idxLoc,anT);ismember(idxLoc,eT);ismember(idxLoc,ntT)];
        infThirdFt = [infThirdFt;find(locRes1==1)];
    end
    
    thirdFt = [];
    for kkk=1:numel(infThirdFt)
        thirdFt = [thirdFt,strcat(num2str(infThirdFt(kkk)))];
    end
    
    if isempty(thirdFt)
        thirdFt = '0';
    end
    
    CLgroup{1,jjj}(3,1) = str2num(thirdFt); 
    
    % 4th feature: Intensity of peak consumption (NS,MS,S as 1 2 3/ 0 means no peak)
    fourthFt = [];
    for kkk = 1:size(peakInfo{1,jjj},2)
        meanP = mean(avgClus(jjj,:));  stdP = std(avgClus(jjj,:));
        if peakInfo{1,jjj}(1,kkk)<meanP+stdP
            fourthFt = [fourthFt,strcat(num2str(1))];
        elseif peakInfo{1,jjj}(1,kkk)<meanP+2*stdP
            fourthFt = [fourthFt,strcat(num2str(2))];
        else
            fourthFt = [fourthFt,strcat(num2str(3))];
        end
        
    end
    if isempty(fourthFt)
        fourthFt = '0';
    end
    CLgroup{1,jjj}(4,1) = str2num(fourthFt);
    
    
    % 5th feature: width of peaks
    fifthFt = [];
    
    for kkk = 1:size(peakInfo{1,jjj},2)
        
        if peakInfo{1,jjj}(3,kkk)<threLow
            fifthFt = [fifthFt,strcat(num2str(1))];
        elseif peakInfo{1,jjj}(3,kkk)<threMid
            fifthFt = [fifthFt,strcat(num2str(2))];
        else
            fifthFt = [fifthFt,strcat(num2str(3))];
        end

    end
    
    if isempty(fifthFt)
        fifthFt = '0';
    end
    CLgroup{1,jjj}(5,1) = str2num(fifthFt); 
    
end

CL{iii} = CLgroup;

end

CL = [CL{1},CL{2},CL{3}];
    
[uniqueLbl, idxUniqueLbl ,idxOrigLbl] = uniquecell(CL)



%%%% Re-label dataTempLib based on merging results
dataTempLib2 = [];
for ii = 2:size(dataTempLib,2)
    dataTempLib{ii}(:,end) = max(dataTempLib{ii-1}(:,end)) + dataTempLib{ii}(:,end);
end

for ii = 1:size(dataTempLib,2)
    dataTempLib2 = [dataTempLib2;dataTempLib{ii}];
end

idxEnd = 1:size(dataTempLib2,1);
for ii = 1:numel(unique(dataTempLib2(:,end)))
    idx = find(dataTempLib2(:,end)==ii);
    idxEnd(idx) = idxOrigLbl(ii);
end

uniqueLbl = unique(idxOrigLbl);
for ii = 1:numel(unique(idxOrigLbl))
    
    idxSim = find(idxOrigLbl == uniqueLbl(ii));
    
%     figure;
%     for jj = 1:numel(idxSim)
%         hold on; plot(mean(dataTempLib2(dataTempLib2(:,end)==idxSim(jj),1:end-1)))
%     end
%     title(num2str(uniqueLbl(ii)));
end

%%%%%%%%% Merge clusters based on labeling
dataTempLib2(:,end) = idxEnd;




%%%%%%%%%%%%%%%%%%%%%% Quantify energy deviation and EMD (earth mover distance metric)

%  %%%%%%%%% EMD dist
%  
%  EMDLib = {}; % library of EMD distance 
%  errLib = []; % library of average error in energy
%  
% for jj = 1:numel(unique(dataTempLib2(:,end)))    
%      
%     clusTar = dataTempLib2(dataTempLib2(:,end) == jj,1:end);
%     clusAvg = mean(clusTar(:,1:end-1));
%     
%     mDist = zeros(size(clusTar,1),1); % EMD dist for each cluster
%     
%     
%     consumInfoAll = [];
%     
%     %%%%%%%%%%%%%%%% Error estimation
%     for kk = 1:size(clusTar,1)
%         
%           consum=(24/lastTime)*trapz(clusTar(kk,1:size(clusTar,2)-1));
%           consumInfoAll = [consumInfoAll;consum,kk];  
%         
%     end
%     
%     consumAvg=(24/lastTime)*trapz(clusAvg);
%     
%     %%% exception case
%     idxZ = find(consumInfoAll(:,1)==0);
%     consumInfoAll(idxZ,:) = [];
%     %%%
%     
%     consum2 = abs((consumInfoAll(:,1) - consumAvg)./consumInfoAll(:,1));
%     errLib = [errLib;mean(consum2)];
%     
%     %%%%%%%%%%%%%%%%
%     
%    for kk = 1:size(clusTar,1)
%         clusTar(kk,1:end-1) = clusTar(kk,1:end-1)./sum(clusTar(kk,1:end-1));
%    end
%    
%    clusAvg = clusAvg./sum(clusAvg);
%    
%     
%    for ii = 1:size(clusTar,1)
%        EMD = zeros(1,size(clusTar,2));
%        EMD(1) = 0;
%        for jjj = 1:size(clusTar,2)-1
%            
%            EMD(jjj+1) = clusTar(ii,jjj) + EMD(jjj) - clusAvg(jjj);
%            
%        end
%        
%        mDist(ii) = sum(abs(EMD));
%    end
%    
%    EMDLib{jj} = sum(mDist);
%    
% end