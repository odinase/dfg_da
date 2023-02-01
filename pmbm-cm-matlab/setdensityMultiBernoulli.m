function val = setdensityMultiBernoulli(x,exiList,muList,cvList)


card = size(x,2);
maxCard = size(muList,2);

% Given cardinality, what are the combinations I need to evaluate? How do I generate them?

a = perms(1:maxCard);
b = unique(a(:,1:card),'rows');
nSP = size(b,1);

val = 0;

for ii=1:nSP
   
    muI = muList(:,b(ii,:));
    cvI = cvList(:,:,b(ii,:));
    exiI = exiList(b(ii,:));
    exiNonI = exiList(setdiff(1:maxCard,b(ii,:)));
    
    pdfProdLog = 0;
    for jj=1:card
        pdfProdLog = pdfProdLog + normpdfLog(x(:,jj),muI(:,jj),cvI(:,:,jj));
    end
    pdfPart = exp(pdfProdLog);
    exiPart = prod(exiI);
    nonExiPart = prod(1-exiNonI);
    val = val + pdfPart*exiPart*nonExiPart;
end

