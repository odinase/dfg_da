function val = setdensityPoisson(x,lambda,mu,cv)

card = size(x,2);
pdfProdLog = sum(normpdfLog(x,mu,cv),2);
pdfProd = exp(pdfProdLog);
preFactor = exp(-lambda)*lambda^card;
val = preFactor*pdfProd;