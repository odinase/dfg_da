function piCol = genPiCol(h,m,maxM)

k = size(h,1)-1;
nSlots = size(h,2);

% Given nSlots and m, generate all omegas

collection = NaN*zeros(1,nSlots);
for s=1:nSlots
    accumulator = zeros(0,nSlots);
    for ii=0:m
        colS = collection;
        colS(:,s) = ii;
        iBool = colS == ii;
        iBoolSum = sum(iBool,2);
        if(ii>0)
            toBeRemoved = iBoolSum > 1;
        else
            toBeRemoved = false(size(iBoolSum));
        end
        accumulator = [accumulator;colS(~toBeRemoved,:)];
    end
    collection = accumulator;
end


% Split the collection of omega's into pi's
% This is essentially a clustering operation

labelGen = 0;
labels = zeros(size(collection,1),1);
piCol = zeros(k+1,nSlots,0);

for ii=1:size(collection,1)
    omega = collection(ii,:);
    pi = [h(1:(end-1),:);omega];
    test = false;
    for jj=1:size(piCol,3)
        labels(ii) = labels(jj);
        piPrev = piCol(:,:,jj);
        x1 = encodeHypo(piPrev,maxM);
        x2 = encodeHypo(pi,maxM);
        if(all(sort(x1) == sort(x2)))
            test = true;
            break;
        end
    end
    
    % Check whether the columns in piPrev are identical to the columns in pi
    
    if(~test)
        labelGen = labelGen + 1;
        labels(ii) = labelGen;
        piCol = cat(3,piCol,pi);
    end
end