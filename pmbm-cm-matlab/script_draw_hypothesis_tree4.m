%load 'pq33';
%load 'pq874';
%load 'pqworked';
clear all;
%load 'pqstuff220221b';
%load 'pq45';
load 'pqWorkedMarch.mat';

% Version 2: Make something that I can use in an Excel table instead

parentList = vertcat(pq.parentInSearchTree);
depthForEachNode = treedepth(parentList);

childrenSurvey = struct('nChildren',{},'children',{});

for ii=1:length(pq)
   
    % Identify all children of pq(ii)
    
    ix = find(parentList == ii);
    
    childrenSurvey(ii).nChildren = length(ix);
    children = struct('ch',{},'switch',{},'expand',{},'clusterBase',{},'trackBase',{},'label',{},'parentInBestList',{},'bound',{},'score',{},'depth',{});
    for jj=1:length(ix)
        children(jj).ch = ix(jj);
        children(jj).switch = pq(ix(jj)).switchOrExpand.switch;
        children(jj).expand = pq(ix(jj)).switchOrExpand.expand;
        children(jj).clusterBase = pq(ix(jj)).switchOrExpand.clusterBase;
        children(jj).trackBase = pq(ix(jj)).switchOrExpand.trackBase;
        children(jj).label = pq(ix(jj)).labelHypo;
        children(jj).score = pq(ix(jj)).score;
        children(jj).bound = pq(ix(jj)).bound;
        children(jj).depth = depthForEachNode(ix(jj));
        if(~isempty(children(jj).clusterBase))
        children(jj).parentInBestList = pq(ix(jj)).clustercontrib(children(jj).clusterBase).parentInBestList; 
        else
            
        end
    end
    childrenSurvey(ii).children = children;
   
end

maxCol = max(vertcat(childrenSurvey.nChildren));
maxRow = length(pq)*4;

ac = cell(maxRow,maxCol);

[depthSorted,dsOrder] = sort(depthForEachNode,'ascend');


for ii=1:length(pq)
   
    firstRowIndex = ii*4-3;
    secondRowIndex = ii*4-2;
    thirdRowIndex = ii*4-1;
    fourthRowIndex = ii*4;
    
    ac{firstRowIndex,1} = dsOrder(ii);
    firstCol = 2;
    lastCol = childrenSurvey(dsOrder(ii)).nChildren*2 + 1;
    for jj=1:childrenSurvey(dsOrder(ii)).nChildren
        
       ac{firstRowIndex,2*jj} = childrenSurvey(dsOrder(ii)).children(jj).ch;
       if(childrenSurvey(dsOrder(ii)).children(jj).switch == true)
           ac{secondRowIndex,2*jj} = 'Switch';
           
           ac{secondRowIndex,2*jj+1} = childrenSurvey(dsOrder(ii)).children(jj).bound;
           ac{thirdRowIndex,2*jj+1} = childrenSurvey(dsOrder(ii)).children(jj).score;
           
           % Can I also include parentInBestList here?
           
           
           pib = childrenSurvey(dsOrder(ii)).children(jj).parentInBestList;
           
           ac{thirdRowIndex,2*jj} = ['C-',num2str(childrenSurvey(dsOrder(ii)).children(jj).clusterBase),' (',num2str(pib),')'];
           
       elseif(childrenSurvey(dsOrder(ii)).children(jj).expand == true) 
           ac{secondRowIndex,2*jj} = 'Expand';
           
           ac{secondRowIndex,2*jj+1} = childrenSurvey(dsOrder(ii)).children(jj).bound;
           ac{thirdRowIndex,2*jj+1} = childrenSurvey(dsOrder(ii)).children(jj).score;           
           
           ac{thirdRowIndex,2*jj} = ['C-',num2str(childrenSurvey(dsOrder(ii)).children(jj).clusterBase),' T-',num2str(childrenSurvey(dsOrder(ii)).children(jj).trackBase)];
       end
       ac{fourthRowIndex,2*jj} = childrenSurvey(dsOrder(ii)).children(jj).label;
       ac{fourthRowIndex,2*jj+1} = ['Depth ',num2str(childrenSurvey(dsOrder(ii)).children(jj).depth)];
       
       
       % Squeeze clusterBase and trackBase into single text cell so that I
       % can put label in fourth row

       
    end
end

toBeDeleted = false(size(ac,1),1);
for ii=1:size(ac,1)
    
    testAnyContent = false;
    for jj=2:size(ac,2)
        if(~isempty(ac{ii,jj}))
           testAnyContent = true; 
        end
    end
    if(~testAnyContent)
       toBeDeleted(ii) = true; 
    end
end
ac = ac(~toBeDeleted,:);
%writecell(ac,'testDisplayTreeWorked.xlsx')
writecell(ac,'pqWorkedMarch.xlsx')
