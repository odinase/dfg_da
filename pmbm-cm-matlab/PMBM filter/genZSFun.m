function [zBar,sBar,pBar] = genZSFun(xStuff,pStuff,hTar2TarMea,rTar)

xBar = xStuff;
pBar = covVec2Mat(pStuff); % This would be more complicated for an SKF
sBar =  hTar2TarMea*pBar*hTar2TarMea' + rTar;
zBar = hTar2TarMea*xBar ;