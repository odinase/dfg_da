clear all;
clusters = struct('clustercontrib',{});

tracks = 1:6;
measurements = 1:2; % Can I avoid dummy measurements?
m = length(measurements);
mSuper = m + 2;
n = length(tracks);

rewMat = [4.3,4.77,-2.3; -0.19,3.38, -2.3; 3.85,1.13,-2.4; 1.65,-1.25,-2.3; 4.66,4.62,-2.3; 1.77,4.13,-2.3];

clusters(1).clustercontrib(1) = struct('hypo',[2,5],'priorscore',8.04);
clusters(1).clustercontrib(2) = struct('hypo',[1,6],'priorscore',2.29);
clusters(2).clustercontrib(1) = struct('hypo',[3],'priorscore',3.85);
clusters(2).clustercontrib(2) = struct('hypo',[4],'priorscore',-4.08);
nC = length(clusters);

% Then I Have all the info needed to construct the graph
% Should I also directly implement the nodes and factors?

% Types: 1 = hypo, 2 = track, 3 = measurement
% All nodes are scalar integer valued
% m+1 = misdetection
% m+2 = non-existence


nodes = struct('type',{},'number',{},'numberInType',{},'outcomespace',{});
nodes(1) = struct('type',1,'number',1,'numberInType',1,'outcomespace',[1,2]);
nodes(2) = struct('type',1,'number',2,'numberInType',2,'outcomespace',[1,2]); % Should this be 3 and 4 instead?
nodes(3) = struct('type',2,'number',3,'numberInType',1,'outcomespace',[1,2,m+1,m+2]);
nodes(4) = struct('type',2,'number',4,'numberInType',2,'outcomespace',[1,2,m+1,m+2]);
nodes(5) = struct('type',2,'number',5,'numberInType',3,'outcomespace',[1,2,m+1,m+2]);
nodes(6) = struct('type',2,'number',6,'numberInType',4,'outcomespace',[1,2,m+1,m+2]);
nodes(7) = struct('type',2,'number',7,'numberInType',5,'outcomespace',[1,2,m+1,m+2]);
nodes(8) = struct('type',2,'number',8,'numberInType',6,'outcomespace',[1,2,m+1,m+2]);
nodes(9) = struct('type',3,'number',9,'numberInType',1,'outcomespace',1:6);
nodes(10) = struct('type',3,'number',10,'numberInType',2,'outcomespace',1:6);

% Can all the factors then be encoded as matrices?
% What would the factor between nodes 1 and 3 be?

firstFMat = zeros(2,4);
firstFMat(1,m+2) = 1;
firstFMat(2,1) = 1;
firstFMat(2,2) = 1;
firstFMat(2,m+1) = 1; % Isnt that all? % I could make this factors with a for loop, couldnt I?

factors = struct('nodeIn',{},'nodeOut',{},'matrix',{});
%factors(1) = struct('nodeIn',1,'nodeOut',3,'matrix',firstFMat);


f = 1;

% Pure prior-hypothesis factors

for c=1:nC
    
    preProb = exp(vertcat(clusters(c).clustercontrib.priorscore));
    prob = preProb/sum(preProb);
    factors(f) = struct('nodeIn',c,'nodeOut',c,'matrix',prob);
    f = f + 1;
end


% Hypothesis-track factors - these include the pure track factors

for ii=1:n
   
    
    
    % Which cluster does the track belong to?
 
    cBelong = NaN;
    for c=1:nC
       tracksInC = horzcat(clusters(c).clustercontrib.hypo); 
        if(ismember(ii,tracksInC))
           cBelong = c;
        end
    end
    nodeIn = cBelong;
    
    
    
    nH = length(clusters(cBelong).clustercontrib);
    nodeOut = nC + ii;
    fMatrix = zeros(nH,mSuper);
    for h=1:nH
       if(ismember(ii,clusters(cBelong).clustercontrib(h).hypo))
          preProbs = exp(rewMat(ii,:));
          probs = preProbs/sum(preProbs);
          fMatrix(h,1:m+1) = probs; 
       else
          fMatrix(h,m+2) = 1; 
       end
    end
    factors(f) = struct('nodeIn',nodeIn,'nodeOut',nodeOut,'matrix',fMatrix);

    f = f + 1;
end

% Track-measurement factors


for ii=1:n
    
    for jj=1:m
       if(~isinf(rewMat(ii,jj)))
          
           fMatrix = zeros(mSuper,n);
           fMatrix(jj,ii) = 1;
           factors(f) = struct('nodeIn',nC+ii,'nodeOut',nC+n+jj,'matrix',fMatrix);
           f = f + 1;
       end
    end
end

% How do I initialize LBP?
% Define an order of the nodes
% Or identify all factor-to-node connections and conversely
% But I should be able to reduce it to MRF for simplicity

for ii=1:length(nodes)
   
    % Find all incoming messages?
    
    
    
end




