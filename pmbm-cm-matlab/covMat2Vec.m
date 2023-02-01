function covVecArr = covMat2Vec(covMatArr)



nI = size(covMatArr,3);
M_length=size(covMatArr,1);
lowerPart = zeros(M_length*(M_length-1)/2,nI);
diagPart = zeros(M_length,nI);
for ii=1:nI
    insert_start=1;
    for kk=1:M_length-1,
        insert_end=insert_start+M_length-kk-1;
        lowerPart(insert_start:insert_end,ii)=covMatArr(kk+1:M_length,kk,ii);
        insert_start=insert_end+1;
    end
    diagPart(:,ii) = diag(squeeze(covMatArr(:,:,ii)));
end
covVecArr = [diagPart;lowerPart];