function testBool = checkDuplicates(hyposWill,hyposCardWill,clustersWill,clustersCardWill)

% Function to check for duplicate hypotheses in cluster-based PMBM. 
% Duplicates within same clusters always trigger the test.
% Duplicates between clusters trigger the test if the hypotheses are non-empty. 

begsHS = tCloud2BegInd(hyposCardWill);
endsHS = tCloud2EndInd(hyposCardWill);

testBool = false;

for i1=1:length(begsHS)
    for i2=(i1+1):length(begsHS)
        h1 = hyposWill(:,begsHS(i1):endsHS(i1));
        h2 = hyposWill(:,begsHS(i2):endsHS(i2));
        cluster1 = hypo2cluster(i1,clustersWill,clustersCardWill);
        cluster2 = hypo2cluster(i2,clustersWill,clustersCardWill);
        if(size(h1,2) == size(h2,2))
            test1 = isnan(h1);
            test2 = isnan(h2);
            nanTest = all(vec(test1 == test2));
            if(nanTest)
                notNan = ~test1;
                h1Vec = h1(notNan);
                h2Vec = h2(notNan);
                hyposEqual = all(h1Vec == h2Vec);
                if(isempty(h1Vec))
                    if(cluster1 == cluster2 && hyposEqual)
                        testBool = true;
                    end
                else
                    if(hyposEqual)
                        testBool = true;
                    end
                end
            end
        end
    end
end

