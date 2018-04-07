function [Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA] = EstPSFQuadProgPartition(IsStack, IbStack, KSIZE, LAMBDAS, options, bOpenPool)
% Based on EstPSFQuadProg but partitions the image to keep the problem
% tractable. Here Is and Ib are stack of images
% lambda1 * ||grad x||^2 + lambda2 * ||x||^2 + lambda3 * ||x * T||_1
if(~exist('LAMBDAS', 'var'))
    LAMBDAS = zeros(1, 4);
end

if(~exist('bOpenPool', 'var'))
    bOpenPool = 1;
end
X0 = ones(KSIZE); %zeros(KSIZE);
%X0((R-1)/2 + 1, (C-1)/2 + 1) = 1;
X0 = X0 / sum(X0(:));

if(~exist('options', 'var'))
    options = optimoptions(@quadprog, 'MaxIter', 10000, 'Algorithm', 'interior-point-convex');
end

Aeq = ones(1, numel(X0));
beq = 1;

% %nPool = matlabpool('size');
% if(bOpenPool)
%     nPool = getPoolSize();
% 
%     if(nPool == 0)
%         matlabpool open
%     end
% end

[R, C] = size(X0);
[rr, cc] = meshgrid(-(R-1)/2:(R - 1)/2, -(C-1)/2:(C-1)/2);
T = (rr.^2 + cc.^2);
Tn = T / max(T(:));

H = diag(LAMBDAS(2) * sparse(ones(1, numel(X0))));
f = LAMBDAS(3) * Tn(:);

for ch = 1:size(IsStack, 3)
    Is = IsStack(:,:,ch);
    Ib = IbStack(:,:,ch);
    A = im2convmtx(Is, R, C);

    % Use gradient image for data cost computation
    if(LAMBDAS(4) > 0)
        [IDx, IDy] = gradient(Is);
        AIDx = im2convmtx(IDx, R, C);
        AIDy = im2convmtx(IDy, R, C);
    end

    H = H + A' * A;
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

    f = f - A' * Ib(:) ;

    if(LAMBDAS(4) > 0)
        [IbDx, IbDy] = gradient(Ib);
        f = f - LAMBDAS(4) * (AIDx' * IbDx(:) + AIDy' * IbDy(:));
    end
end
[HestV,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(H,f,[],[], Aeq, beq, zeros(numel(X0), 1), [], X0, options);
Hest = reshape(HestV, size(X0));
