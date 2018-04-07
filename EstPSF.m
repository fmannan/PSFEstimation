function [Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = EstPSF(Is, Ib, KSIZE, OptObj, LAMBDAS, bGradCheck, options, EPS)

X0 = ones(KSIZE); %zeros(KSIZE);
%X0((R-1)/2 + 1, (C-1)/2 + 1) = 1;
X0 = X0 / sum(X0(:));

[R, C] = size(X0);

%X0 = fspecial('gaussian', size(X0), 1);

A = im2convmtx(Is, size(X0,1), size(X0, 2));
if(~exist('options', 'var') || isempty(options))
    if(exist('bGradCheck', 'var') && bGradCheck ~= 0)
        rng(0,'twister'); 
        options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'DerivativeCheck','on', 'FinDiffType', 'central', 'GradObj', 'on', 'UseParallel', 'always');
    else
        options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'off', 'UseParallel', 'always');            
        %options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'on', 'Hessian', 'user-supplied', 'HessFcn',@(x, lambda) (A'*A), 'UseParallel', 'always');
    end
end

Aeq = ones(1, numel(X0));
beq = 1;

nPool = matlabpool('size');
if(nPool == 0)
    matlabpool open
end
if(all(LAMBDAS == 0))
    H = A' * A;
    f = - A' * Ib(:);
    options = optimoptions(@quadprog, 'MaxIter', 10000, 'Algorithm', 'interior-point-convex');            
    [HestV,FVAL,EXITFLAG,OUTPUT,LAMBDA] = quadprog(H,f,[],[], Aeq, beq, zeros(numel(X0), 1), [], X0, options);
    Hest = reshape(HestV, size(X0));
    GRAD = nan;
    HESSIAN = nan;
else
    [Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) OptObj(x, Ib, Is, LAMBDAS, A, A'*A, A'*Ib(:), EPS),X0,[],[],Aeq,beq, zeros(numel(X0), 1), [], [], options);
end
