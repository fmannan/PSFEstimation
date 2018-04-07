function [Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA] = EstPSFQuadProg(Is, Ib, KSIZE, LAMBDAS, bAllowNegVal, options, bOpenPool, PadType)
% lambda1 * ||grad x||^2 + lambda2 * ||x||^2 + lambda3 * ||x * T||_1
if(~exist('LAMBDAS', 'var'))
    LAMBDAS = zeros(1, 4);
end

if(~exist('bOpenPool', 'var'))
    bOpenPool = 0;
end

if(~exist('PadType', 'var'))
  PadType = 'replicate';
end

% pad both images appropriately based on the kernel size
Is = padarray(Is, (KSIZE - 1)/2, PadType);
Ib = padarray(Ib, (KSIZE - 1)/2, PadType);

% smooth Is and Ib for gradient computation
H = fspecial('gaussian', 7, 1);
IsG = conv2(Is, H, 'same');
IbG = conv2(Ib, H, 'same');

X0 = ones(KSIZE); %zeros(KSIZE);
%X0((R-1)/2 + 1, (C-1)/2 + 1) = 1;
X0 = X0 / sum(X0(:));

[R, C] = size(X0);

A = im2convmtx(Is, R, C);

% Use gradient image for data cost computation
if(LAMBDAS(4) > 0)
    [IDx, IDy] = gradient(IsG);
    AIDx = im2convmtx(IDx, R, C);
    AIDy = im2convmtx(IDy, R, C);
end

if(~exist('options', 'var'))
    options = optimoptions(@quadprog, 'MaxIter', 10000, 'Algorithm', 'interior-point-convex');
end

Aeq = ones(1, numel(X0));
beq = 1;

% %nPool = matlabpool('size');
% nPool = getPoolSize();
% 
% if(nPool == 0 && bOpenPool)
%     matlabpool open
% end

[rr, cc] = meshgrid(-(R-1)/2:(R - 1)/2, -(C-1)/2:(C-1)/2);
T = (rr.^2 + cc.^2);
Tn = T / max(T(:));

tmp = 1e6 * ones(size(Tn));
tmp(2:end-1,2:end-1) = Tn(2:end-1,2:end-1);
Tn = tmp;

H = A' * A + diag(LAMBDAS(2) * sparse(ones(1, numel(X0))));
if(LAMBDAS(4) > 0)
    H = H + LAMBDAS(4) * (AIDx' * AIDx + AIDy' * AIDy);
end

if(abs(LAMBDAS(1)) > 1e-8)
    Dx = [-1, 1];
    Dy = [-1;1];
    ADx = im2convmtx(Dx, R, C);
    ADy = im2convmtx(Dy, R, C);
    H = H + LAMBDAS(1) * (ADx' * ADx + ADy' * ADy);
end

f = - A' * Ib(:) + LAMBDAS(3) * Tn(:);

if(LAMBDAS(4) > 0)
    [IbDx, IbDy] = gradient(IbG);
    f = f - LAMBDAS(4) * (AIDx' * IbDx(:) + AIDy' * IbDy(:));
end

LB = zeros(numel(X0), 1);
if(exist('bAllowNegVal', 'var')  && bAllowNegVal)
    LB = [];
end
[HestV,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(H,f,[],[], Aeq, beq, LB, [], X0, options);
Hest = reshape(HestV, size(X0));
