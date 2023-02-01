function temp = excelwrite2(hypos,hyposCard,clusters,clustersCard,probLogHypos,meaHistCol,trackFile,predX,inCol,k)

% Need maximum number of columns
% (or perhaps I don't really need that)

nT = size(meaHistCol,2);
maxCol = max([nT+1,size(hypos,2),size(meaHistCol, 2)]);


% row1 = zeros(1,0);
% row2 = zeros(1,0);
% row3 = zeros(1,0);
% row4 = zeros(1,0);
% 
% row5 = 'hypos';
% row6 = hypos;
% begsH = tCloud2BegInd(hyposCard);
% 
% hc = zeros(1,length(hypos));
% hc(begsH) = hyposCard;
% row7 = hc;
% 
% 
% ac = {row1,row2,row3,row4,row5,row6,row7}';
% 
% 
% 
% 
% writecell(ac,'testac.xlsx')

temp = 0;

probabilitiesCell = probLogs2Probabilities(probLogHypos,clusters,clustersCard);
probabilities = zeros(1,length(hyposCard));
begsC = tCloud2BegInd(clustersCard);
endsC = tCloud2EndInd(clustersCard);
for iC=1:length(probabilitiesCell)
    probabilities(clusters(begsC(iC):endsC(iC))) = probabilitiesCell{iC};
end

row1 = cell(1,maxCol);
row2 = cell(1,maxCol);
row3 = cell(1,maxCol);
row4 = cell(1,maxCol);
row5 = cell(1,maxCol);
row5{1} = 'hypos';

row5a = cell(1,maxCol);
row6 = cell(1,maxCol);
for ii=1:length(hypos)
   row6{ii} = hypos(ii); 
end
row7 = cell(1,maxCol);
row8 = cell(1,maxCol);
row9 = cell(1,maxCol);

begsH = tCloud2BegInd(hyposCard);
hc = zeros(1,length(hypos));
hc(begsH) = hyposCard;

lc = zeros(1,length(hypos));
lc(begsH) = 1:length(hyposCard);

pc = zeros(1,length(hypos));
pc(begsH) = probabilities;

[bH,cH] = a2bc(1:size(hypos,2),hyposCard);

for ii=1:length(hc)
    if(hc(ii) > 0)
       row7{ii} = hc(ii); 
       row8{ii} = lc(ii); 
       row9{ii} = pc(ii); 
       row5a{ii} = ['H',num2str(cH(ii))];
    else 
    end    
end

row10 = cell(1,maxCol);
row11 = cell(1,maxCol);
row11{1} = 'clusters';
row12 = cell(1,maxCol);
row13 = cell(1,maxCol);
row14 = cell(1,maxCol);
row15 = cell(1,maxCol);

row11a = cell(1,maxCol);

for ii=1:length(clusters)
   row12{ii} = clusters(ii); 
   row14{ii} = hyposCard(ii); 
   row15{ii} = probabilities(ii);  
end
cc = zeros(1,length(clusters));
cc(begsC) = clustersCard;
ld = zeros(1,length(clusters));
ld(begsC) = 1:length(clustersCard);
pd = probabilities;

[bClu,cClu] = a2bc(1:size(clusters,2),clustersCard);

for ii=1:length(cc)
    if(cc(ii) > 0)
       row13{ii} = cc(ii); 
       row11a{ii} = ['C',num2str(cClu(ii))];
       %row15{ii} = pd(ii);        
    else 
    end    
end

row16 = cell(1,maxCol);
row17 = cell(1,maxCol);
row17{1} = 'meaHistCol';
row19 = cell(1,maxCol);
nT = size(meaHistCol,2);
for ii=1:nT
   row19{ii} = ii;  
end
row20 = cell(1,maxCol);

maxLag = size(meaHistCol,1);
nKUse = min([maxLag,k]);
meaHistColRelevant = meaHistCol(maxLag+1-nKUse:end,:);
cellMeaHist = cell(nKUse,maxCol);

for ii=1:size(meaHistCol,2)
    for jj=1:nKUse
        if(~isnan(meaHistCol(maxLag-nKUse+jj,ii)))
            cellMeaHist{jj,ii} = meaHistCol(maxLag-nKUse+jj,ii);
        else
            cellMeaHist{jj,ii} = 'NaN';
        end
    end
end
row21 = cell(1,maxCol);
row22 = cell(1,maxCol);
row22{nT+1} = 'Existence probabilities';
for ii=1:size(meaHistCol,2)
    row22{ii} = trackFile(inCol.exi,ii);
end

[trackSumProbs,trackTotalProbs,probabilitiesCell] = trackProbAccumulatePure(trackFile,inCol,hypos,hyposCard,clusters,clustersCard,probLogHypos);

row23 = cell(1,maxCol);
row23{nT+1} = 'Total track probabilities';
for ii=1:size(meaHistCol,2)
    row23{ii} = trackTotalProbs(ii);
end

row24 = cell(1,maxCol);
row24{nT+1} = 'Labels';
for ii=1:size(meaHistCol,2)
    row24{ii} = trackFile(inCol.label,ii);
end

cellTrackStates = cell(4,maxCol);
for ii=1:size(meaHistCol,2)
    for jj=1:4
        %cellTrackStates{jj,ii} = trackFile(inCol.tarX(jj),ii);
        cellTrackStates{jj,ii} = predX(jj,ii);
    end
end

% Also identify the cluster membership of tracks
row25 = cell(1,maxCol);
[clusterNumbers] = track2Cluster(1:nT,hypos,hyposCard,clusters,clustersCard);
for ii=1:size(meaHistCol,2)
    row25{ii} = ['C',num2str(clusterNumbers(ii))];
end



ac = [row1; row2; row3; row4; row5; row5a; row6; row7; row8; row9;row10;row11;row11a;row12;row13;row14;row15;row16;row17;row19;row20;cellMeaHist;row1;row21 ; row22; row23; row24; row21; cellTrackStates; row21; row25;];

%writecell(ac,'mechiTrackingk9-AfterDM-Unsorted.xlsx')
%writecell(ac,'mechiLowFALowPInitVelk611C-AfterRC-Unsorted.xlsx')
%writecell(ac,'mechiLowFALowPInitVelk612e-AfterRC-Unsorted.xlsx')
%writecell(ac,'TAESk4-AfterCS-Unsorted.xlsx')
writecell(ac,'mechiSep-k612BeforeDM.xlsx')
%writecell(ac,'mechiEdmund-k30.xlsx')
%writecell(ac,'initscenario2-k30ARC.xlsx')
%writecell(ac,'initscenario-dd1-iMC2-k23ACS.xlsx')
%writecell(ac,'mechi-5M-k874BDM.xlsx')