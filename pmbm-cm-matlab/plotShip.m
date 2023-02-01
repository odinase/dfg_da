function plotShip(sv,color)

shipBasePoints = 4*[1,0; 0.5,0.65; -1,0.65; -1,-0.65; 0.5,-0.65; 1,0]';
shipBasePoints(2,:) = shipBasePoints(2,:)*0.75;
nP = size(shipBasePoints,2);
shipBaseExt = [shipBasePoints; zeros(1,nP)];
shipBase = repmat(sv(1:3),[1,nP]) + quatRotate(shipBaseExt,sv(4:7));

%fill(shipBase(2,:),shipBase(1,:),color);
fill(shipBase(1,:),shipBase(2,:),color);

