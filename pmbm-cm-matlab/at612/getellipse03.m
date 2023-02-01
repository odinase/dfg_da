function h = getellipse03(x,P,nSigma,nSamples,thetaMin,thetaMax,connect,direction)

% Function to obtain coordinates of arc of ellipse
% Original version written by Matt Walter, MIT OE
% Substantially amended by Edmund Brekke, NUS TMSI during January 2011
% @x:           Centre of ellipse
% @P:           Covariance matrix for which ellipse is nSigma contour
% @nSigma:      Scaling factor (e.g. size of validation gate)
% @nSamples:    Number of samples
% @thetaMin:    (Optional) Where path begins
% @thetaMax:    (Optional) Where path ends
% @connect:     (Optional) +1 for closed path, otherwise path will be open
% @direction:   (Optional) -1 for clockwise, otherwise counterclockwise
% >h:           2 x nSamples array of ellipse contour coordinates

% Caution: If thetaMin and thetaMax are specified this function returns an
% ellipse segment covering all angles from thetaMin to thetaMax in a
% COUNTERCLOCKWISE manner. The optional variables thetaMin and thetaMax
% must therefore be specified with this in mind. The optional variable
% direction must be used in one wants a clockwise oriented segment.

if(nargin < 4)
   error('getellipse03 must at least be provided 4 arguments'); 
end
if(nargin < 7)
   connect = true; 
end
nSamples = nSamples - connect;
if(nargin == 4)
    dTheta = 2*pi/(nSamples-1);
    thetaVec = (0:(nSamples-1))*dTheta;
else
    controlZ = mod(thetaMin-0.1/nSamples,2*pi);
    deltaTheta = angularDifference(controlZ,thetaMax,thetaMin);
    dTheta = (deltaTheta)/(nSamples-1);
    thetaVec = thetaMin + (0:(nSamples-1))*dTheta;
end
if(nargin == 7 && direction == -1)
    thetaVec = fliplr(thetaVec);
end

if(det(P) > 0)
    P = (P+P')/2;
    [V,D] = eigs(P);
    y = nSigma*[cos(thetaVec);sin(thetaVec)];
    el = V*sqrtm(D)*y;
    if(nargin >= 7 && ~connect)   
        el = el+repmat(x,1,size(el,2));
    else
        el = [el el(:,1)]+repmat(x,1,size(el,2)+1);
    end
    h = el;
    return;
else
   error('Matrix must be positive definite for ellipse to be constructed');
end

function zOut = angularDifference(controlZ,minuend,subtrahend)

% Function for calculating differences between angles
% Written by Edmund Brekke during Autumn 2010
% Edited March 2011, May 2011 and May 2014
% @controlZ:    A point such that the intervals of interest dont intersect it
% @minuend:     x in x-y
% @subtrahend:  y in x-y
% >zOut:        Angular difference

c = mod(controlZ,2*pi);
minuendShifted = mod(minuend - c,2*pi);
subtrahendShifted = mod(subtrahend - c,2*pi);
zOut = minuendShifted - subtrahendShifted;
z2 = subtrahendShifted - minuendShifted;
otherwayBool = zOut > minuendShifted;
zOut(otherwayBool) = z2(otherwayBool);

