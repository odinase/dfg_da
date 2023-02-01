nTest = 10000;
load('sMarkedStuff.mat');
t0 = unique(sMarked.tracksEntire);
for ii=1:nTest
   q = randperm(15);
   tE = t0(q);
   aE = a(q,:);
    [personToItemCr,~,optRewardCr]=assign2D(-a);
    [personToItemCrE,~,optRewardCrE]=assign2D(-aE);

    seventyone0 = find(t0 == 71);
    seventyoneE = find(tE == 71);
    if(personToItemCr(seventyone0) ~= personToItemCrE(seventyoneE))
       error('lll') 
    end
    
    
    
end
