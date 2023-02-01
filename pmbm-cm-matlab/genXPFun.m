function [xHat,pStuff,pHat,kalmanGain] = genXPFun(xBar,sBar,pBar,nuM,hTar2TarMea)

kalmanGain = pBar*hTar2TarMea'/sBar;
pHat = pBar - pBar*hTar2TarMea'*(sBar\(hTar2TarMea*pBar));
xHat = xBar + kalmanGain*nuM;
pStuff = covMat2Vec(pHat);

