function covMatArr = covVec2Mat(covVecArr)

nI = size(covVecArr,2);
dV = size(covVecArr,1);
dM = (sqrt(8*dV+1)-1)/2;

covMatArr = zeros(dM,dM,nI);
for ii=1:nI 
    covMat = zeros(dM,dM);
    insert_start=dM+1;
    for kk=1:dM-1
        insert_end=insert_start+dM-kk-1;
        covMat(kk+1:dM,kk)=covVecArr(insert_start:insert_end,ii);
        insert_start=insert_end+1;
    end
    covMat = covMat + covMat' + diag(covVecArr(1:dM,ii));
    covMatArr(:,:,ii) = covMat;
end



