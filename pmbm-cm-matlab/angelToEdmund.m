function [hypos,hyposCard,meaHistCol,trackWNumbers,trackHypoNumbers,ttps,probs,trackFile] = angelToEdmund(filter_upd,zCard,k,inCol,maxLag)

% I should even be able to evaluate TTPs in this function


zBegs = tCloud2BegInd(zCard);
zEnds = tCloud2EndInd(zCard);

if(k < maxLag)
    lag = k;
else
    lag = maxLag;
end



meaHistCol = zeros(maxLag,0);
trackFile = zeros(inCol.last,0);
meaHistCounter = 1;

trackWNumbers = zeros(1,0);
trackHypoNumbers = zeros(1,0);
exiProbs = zeros(1,0);

hypos = zeros(1,0);
hyposCard = zeros(1,0);


nTracksW = length(filter_upd.tracks);

for ii=1:nTracksW
    
    
    nHLocal = length(filter_upd.tracks{ii}.aHis);
    for jj=1:nHLocal
        
        track = filter_upd.tracks{ii}.aHis{jj}';
        
        % Last index in track should be current measurement
        
        
        
        
        
        trackDummy = max(maxLag-length(track),0);
        mhc = [zeros(trackDummy,1); track(max(end-lag+1,1):end)];
        
        
%         if(isempty(mhc) || isempty(meaHistCol))
%            error('kkew'); 
%         end
        
        
        meaHistColNew = [meaHistCol,mhc];
        meaHistCol = meaHistColNew;
        
        trackWNumbers = [trackWNumbers,ii];
        trackHypoNumbers = [trackHypoNumbers,jj];
        exiProbs = [exiProbs,filter_upd.tracks{ii}.eB(jj)];

        
        % Then construct trackFile entry
        
        trackFileEntry = NaN*zeros(inCol.last,1);
        trackFileEntry(inCol.tarX) = filter_upd.tracks{ii}.meanB{jj};
        
        trackFileEntry(inCol.tarP) = covMat2Vec(filter_upd.tracks{ii}.covB{jj});
        trackFileEntry(inCol.meaLast) = track(end);
        trackFileEntry(inCol.exi) = filter_upd.tracks{ii}.eB(jj);
        trackFileEntry(inCol.label) = ii;
        
        trackFile = [trackFile, trackFileEntry];
    end
end



ttps = zeros(1,size(exiProbs,2));
probs = filter_upd.globHypWeight/sum(filter_upd.globHypWeight);

for ii=1:size(filter_upd.globHyp,1)
   hypoI = filter_upd.globHyp(ii,:); 
   registeredTracksI = zeros(size(hypoI));
   for twn=1:length(hypoI)
       
      % Identify all tracks in meaHistCol that corresponds to this W-track 
      % Then identify the one track out of these that has hypoI(jj) in trackHypoNumbers 
      % If multiple tracks satisfy this, give error.
      
      thn = hypoI(twn);
      
      ixW = trackWNumbers == twn;
      ixH = trackHypoNumbers == thn;
      
      ix = ixW & ixH;
      if(sum(ix) > 1)
         error('got more than one possible track'); 
      end
      
      tInMCH = find(ix);
      
      % The I need to discriminate: Is this a track that is non-trivial?
      % Trivial tracks are thos with only one potential detection.
      

      
      if(thn > 0 && sum(filter_upd.tracks{twn}.aHis{thn} > 0) > 1)
          
          registeredTracksI(twn) = tInMCH;
          
      end
      
      ttps(tInMCH) = ttps(tInMCH) + probs(ii)*exiProbs(tInMCH);
     
       
   end
   hypoNew = registeredTracksI(registeredTracksI > 0);
   hypos = [hypos,hypoNew];
   hyposCard = [hyposCard,size(hypoNew,2)];
   
   
    
end

% So how about the TTPs? 
% It would be interesting to compare Angels TTPs with my TTPs





