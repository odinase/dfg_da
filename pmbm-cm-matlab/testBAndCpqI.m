function bool = testBAndCpqI(pqI)

bool = zeros(1,max(pqI.tracksCLevel));
for ii=1:max(pqI.tracksCLevel)
    bEntries = pqI.tracksBLevel(pqI.tracksCLevel == ii);
    if(~isempty(bEntries))
        bool(ii) = max(bEntries) == length(bEntries);
    else
        bool(ii) = true; 
    end
end
