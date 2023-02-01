function [xMerged,tMerged] = insertPart(xCloud,tCloud,xNew,tNew)

xMerged = [xCloud,xNew];
tMerged = [tCloud,tNew];