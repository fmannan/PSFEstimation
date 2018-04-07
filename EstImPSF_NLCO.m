function [Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = EstImPSF_NLCO(Is, Ib, KSIZE, OptObj, LAMBDAS, bGradCheck, options, EPS)

X0 = ones(KSIZE); %zeros(KSIZE);
%X0((R-1)/2 + 1, (C-1)/2 + 1) = 1;
X0 = X0 / sum(X0(:));

[R, C] = size(X0);

%X0 = fspecial('gaussian', size(X0), 1);

% A = im2convmtx(Is, size(X0,1), size(X0, 2));
% if(~exist('options', 'var') || isempty(options))
%     if(exist('bGradCheck', 'var') && bGradCheck ~= 0)
%         rng(0,'twister'); 
%         options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'DerivativeCheck','on', 'FinDiffType', 'central', 'GradObj', 'on', 'UseParallel', 'always');
%     else
%         options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'on', 'UseParallel', 'always');            
%         %options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'on', 'Hessian', 'user-supplied', 'HessFcn',@(x, lambda) (A'*A), 'UseParallel', 'always');
%     end
% end
% 
Aeq = ones(1, numel(X0));
beq = 1;
options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'off',  'FinDiffType', 'central',  'UseParallel', 'always');

nPool = matlabpool('size');
if(nPool == 0)
    matlabpool open
end
LB = zeros(size(X0));
UB = [];

[Hest,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) OptObj(x, Is, Ib, LAMBDAS, 0),X0,[],[], Aeq, beq, LB, UB, [], options);

