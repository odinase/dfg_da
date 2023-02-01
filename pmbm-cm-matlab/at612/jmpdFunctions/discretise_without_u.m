function [Phi,S,Ga] = discretise_without_u(F,G,Q,d)

%Function to discretise continuous process model
%Written by Edmund Brekke July 2006
%@ F:       Continuous system matrix
%@ G:       Continuous noise control matrix
%@ Q:       Continuous noise covariance matrix
%@ s:       Sampling interval
%> Phi:     Discrete flow matrix
%> S:       Discrete noise covariance matrix
%> Ga:      Discrete cholesky factorisation of noise covariance


%Flow matrix Phi

Phi = expm(F*d);

%Control matrix Lambda - not to be used

L = zeros(size(F,1),1);
sf = size(F);
sl = size(L);

A = [F,L;zeros(sl(2),sf(2)+sl(2))];

Loan1 = expm(A*d);
Lambda = Loan1(1:sl(1),sf(2)+1:sf(2)+sl(2));


%Use the machinery of Loan (1978) to evaluate integral.

Qc = G*Q*G';
Qc = (Qc+Qc')/2;
dim = size(F,1);
Loan2 = expm([-F,G*Q*G';zeros(dim),F']*d);
G2 = Loan2(1:dim,dim+1:2*dim);
F3 = Loan2(dim+1:2*dim,dim+1:2*dim);

%Calculate Gamma*Gamma'.
A = F3'*G2;
S =(A+A')/2;
if(nargout == 4)
    %Cholesky factorization to obtain Gamma.
    %If not possible use only S.
    Ga = chol(S)';
end