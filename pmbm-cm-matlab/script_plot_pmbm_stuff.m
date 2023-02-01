fig3 = figure;set(fig3,'position',[200,200,1220,510]);   

% Visualize PHD component
                
                phdGrid = muPPP*ones(size(phdGrid));
                
                for ii=1:nTPHD
                    muI = system.hMat*phdTracks(inCol.tarX,ii);
                    coI = system.hMat*covVec2Mat(phdTracks(inCol.tarP,ii))*system.hMat';
                    wI = phdTracks(inCol.exi,ii);
                    
                    % Can I vectorize the Gaussian evaluation?
                    
                    
                    vals = wI*exp(normpdfLog(pMonB,muI,coI));
                    phdGrid = phdGrid + reshape(vals,[nXB,nYB]);
                    
                end
                
                imagesc(xB,yB,log(phdGrid'));axis xy;
                
                %caxis([0,6e-3]);
                caxis([-11.5,-5]);
                hold on;
                
                % Visualize MBM component
                
                dimXVis = 1;
                dimYVis = 2;
                nXVis = 340;
                nYVis = 335;
                lims = [xMinDisp,xMaxDisp,yMinDisp,yMaxDisp];
                %             [gridX,gridY,gridVals] = mbm2phd(dimXVis,dimYVis,lims,nXVis,nYVis,clusters,clustersCard,hypos,hyposCard,probLogHypos,trackFile,inCol);
                %             imagesc(gridX,gridY,gridVals');axis xy;
                %colorbar;
                
                nH = length(hyposCard);
                nC = length(clustersCard);
                
                begsC = tCloud2BegInd(clustersCard);
                endsC = tCloud2EndInd(clustersCard);
                
                begsH = tCloud2BegInd(hyposCard);
                endsH = tCloud2EndInd(hyposCard);
                
                
                nTracks = size(meaHistCol,2);
                for iT=1:nTracks
                    %showTrack(iT,k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,colorArr(mod(iT-1,18)+1,:));
                    showTrack(iT,k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,[1,1,1]);
                    hold on;
                end
                
                
                for iC=1:nC % Loop through all clusters
                    
                    hInC = clusters(:,begsC(iC):endsC(iC));
                    [probs,aSel,b,c] = probLogs2ProbabilitiesForHypos(hInC,probLogHypos,clusters,clustersCard,size(hyposCard,2));
                    
                    % Only display 10 topmost hypotheses from each cluster
                    
                    nHDisp = min(15,clustersCard(iC));
                    [pValsSorted,ix] = sort(probs,'descend');
                    probsSorted = probs(ix);
                    selHInC = hInC(ix(1:nHDisp)); % These should be bona-fide hypothesis numbers
                    for iH=1:length(selHInC)
                        tracksInH = hypos(:,begsH(selHInC(iH)):endsH(selHInC(iH)));
                        probsWithExi = probsSorted(iH)*trackFile(inCol.exi,tracksInH);
                        toBeDisplayed = find(probsWithExi > trackProbLimitDisp);
                        for ii=1:length(toBeDisplayed)
                            %hold on;
                            %showTrack(toBeDisplayed(ii),k,meaHistCol,trackFileShadow,scenario(iMC,dd).zList,scenario(iMC,dd).zCard,inCol,[1,1,1]);
                            hold on;
                            x = trackFile(inCol.tarX(1:2),tracksInH(toBeDisplayed(ii)));
                            pRaw = trackFile(inCol.tarP,tracksInH(toBeDisplayed(ii)));
                            pMat = covVec2Mat(pRaw);
                            pMat = pMat(1:2,1:2);
                            plot(x(1,:),x(2,:),'w*');
                            %plot(x(1,:),x(2,:),'*','color',colorsClusters(iC,:));
                            hold on;
                            %el = getellipse03(x,pMat,1,100);
                            el = getellipse03(x,pMat*nSigma^2,1,100);
                            %plot(el(1,:),el(2,:),'w');
                            %plot(el(1,:),el(2,:),'color',colorsClusters(iC,:));
                            
                            
                            
                            
                        end
                        
                        
                    end
                end
                
                
                hold on;
                
                %clusterHulls(hypos,hyposCard,clusters,clustersCard,trackFile,inCol,colorArr,k);
                
                
                %             for ii=1:size(xPostRunn,2)
                %                 hold on;
                %                 plot(xPostRunn(1,ii),xPostRunn(2,ii),'dc');
                %                 el = getellipse03(xPostRunn(1:2,ii),pPostRunn(1:2,1:2,ii),1,200);
                %                 plot(el(1,:),el(2,:),'c');
                %             end
                %             hold on;
                
                for t=1:length(scenario(iMC,dd).targetsTrue)
                    kTList = scenario(iMC,dd).targetsTrue(t).k;
                    kI = find(scenario(iMC,dd).targetsTrue(t).k == k);
                    %plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'g','linewidth',2);
                    plot(scenario(iMC,dd).targetsTrue(t).x(1,1:kI),scenario(iMC,dd).targetsTrue(t).x(2,1:kI),'color',colorArr(t+1,:),'linewidth',2);
                    hold on;
                end
                
                plot(measurements(1,:),measurements(2,:),'ro','linewidth',2);
                hold on;
                
                hold off;
                axis([xMinDisp,xMaxDisp,yMinDisp,yMaxDisp]);
                %
                title(['Logarithm of PHD at k = ',num2str(k)]);