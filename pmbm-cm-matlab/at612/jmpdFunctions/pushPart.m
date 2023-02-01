function [xUpdated,tUpdated] = pushPart(xOld,tOld,xNew,tNew)

xTemp = [xOld,xNew];
tTemp = [tOld,tNew];

[indicesRemaining,tUpdated] = removeC(1,tTemp);
xUpdated = xTemp(:,indicesRemaining);
