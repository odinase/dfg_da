function [newBranches,trackTreeNodesSurvive,parentNodes] = branchExpandDevijver(measurements,tracksCol,inCol,trackTreeNodes,trackTreeNodesCard,trackPruneThreshold,gammaVar,pD,lambdaClutter,lag,generateZS,generateXP,k)

% Function to expand track tree after reception of measurement scan in MDA
% This function includes Kalman filter updates and track-cost evaluations
% Written by Edmund Brekke during October 2016
% Revised by Edmund Brekke during February 2018 to get rid of navigation states
%@measurements:             Current measurement scan with m measurements
%@tracksCol:                Track file
%@inCol:                    Struct that contains tarXInCol,tarPInCol,meaRefInCol,contribInCol,costInCol
%@trackTreeNodes:           Track tree: Parents, Labels and Lags
%@trackTreeNodesCard
%@xNav:                     Navigation state estimate
%@pNav:                     Navigation uncertainty
%@trackPruneThreshold
%@gammaVar:                 Gating threshold
%@pD:                       Probability of detection
%@lambdaClutter:            Clutter intensity
%>newBranches:              This becomes our new track collection.
%>trackTreeNodesSurvive:    Nodes may not survive if no leaf descendant survives gating and cost thresholds.
%>parentNodes:              
%$generateZS:               Anonymous function to generate sBar and zBar
%$generateXP:               Anonymous function to generate xHat and pHat (and perhaps Kalman gain as well)



% The anonymous functions are used to perform KF updates in a way that enables seamless integration of ordinary KFs, compensated KFs, SKFs and body-frame KFs.

m = size(measurements,2);

newBranches = zeros(size(tracksCol,1),0);
parentNodes = zeros(1,size(newBranches,2)); % How to keep track of parent nodes?
begsTTN = tCloud2BegInd(trackTreeNodesCard);
trackTreeNodesSurvive = false(1,sum(trackTreeNodesCard));

for iA=1:size(tracksCol,2)
    xStuff = tracksCol(inCol.tarX,iA);
    xBar = xStuff;
    pStuff = tracksCol(inCol.tarP,iA);
    
    % Calculate predicted measurement, corresponding covariance and also extract predicted covariance.
    
    [zBar,sBar,pBar] = generateZS(xStuff,pStuff);
   
    zeroBranch = zeros(size(tracksCol,1),1);
    zeroBranch(inCol.tarX) = xBar;
    zeroBranch(inCol.tarP) = tracksCol(inCol.tarP,iA);
    zeroBranch(inCol.meaRef(1:lag-1)) = tracksCol(inCol.meaRef(1:lag-1),iA);
    zeroBranch(inCol.meaRef(lag)) = 0;
    zeroBranch(inCol.contrib(1:(lag-1))) = tracksCol(inCol.contrib(1:(lag-1)),iA);
    zeroBranch(inCol.contrib(lag)) = - log(1-pD);
    newCost = tracksCol(inCol.cost,iA) - log(1-pD);
    zeroBranch(inCol.cost) = newCost;
    
    % Zero-branch should always be included, since pruning this could make the MDA problem unsolvable.
    
    newBranches = [newBranches,zeroBranch];
    parent = begsTTN(lag-1)+iA-1;
    parentNodes = [parentNodes,parent];
    
    an = treeAncestors(trackTreeNodes,parent);
    trackTreeNodesSurvive(an) = true;

    
                  %  if(k==1250 && iA==2)
                  %     error('samsala in branchExpand'); 
                  %  end                    
    
    for jj=1:m
        
        % Calculate innovation and corresponding Mahalanobis distance.
        
        nuM = measurements(:,jj) - zBar;
        d = nuM'*(sBar\nuM);
              
        if(d < gammaVar)
            
            % Successful gating: Perform KF update.
            % This is done inside the anonymous function generateXP.
            
            [xHat,pStuffHat] = generateXP(xBar,sBar,pBar,nuM);
            
            % Cost evaluation for new track
            
            costContribution = - log(pD) + 0.5*d + log(lambdaClutter*sqrt(2*pi*det(sBar)));
            newCost = tracksCol(inCol.cost,iA) + costContribution;
            
                    %if(k==1500 && iA==2 && jj==5)
                   % if(k==1250 && iA==2)
                   %    error('samsala in branchExpand'); 
                   % end                 
            
                    
            if(newCost < trackPruneThreshold)
                
                % Make the new branch
                
                ijBranch = zeros(size(tracksCol,1),1);
                ijBranch(inCol.tarX) = xHat;
                ijBranch(inCol.tarP) = pStuffHat;
                ijBranch(inCol.meaRef(1:lag-1)) = tracksCol(inCol.meaRef(1:lag-1),iA);
                ijBranch(inCol.meaRef(lag)) = jj;
                ijBranch(inCol.contrib(1:(lag-1))) = tracksCol(inCol.contrib(1:(lag-1)),iA);
                ijBranch(inCol.contrib(lag)) = costContribution;
                ijBranch(inCol.cost) = newCost;
                newBranches = [newBranches,ijBranch];
                parentNodes = [parentNodes,parent];         % Keep track of which node is parent of the new branch.
                an = treeAncestors(trackTreeNodes,parent);  % Find all the ancestor nodes of this branch,...
                trackTreeNodesSurvive(an) = true;           % ... and make sure that all ancestor nodes are kept in track tree.
            end
        end
    end
end