function [SigEst, H1est, H2est, result] = EstBETGaussianPSF(I1, I2, params)
% Jul 11, 2015: implemented initial version.
% The current version doesn't have any constraints on sig1 and sig2. So it
% will find any (sig1, sig2) that has the right sigR. Need to put the
% constraint (sig2 - alpha * sig1 - beta) = 0 (TODO)
alpha = params.alpha;
beta = params.beta;
lambda = params.lambda;
UB = 20;
boundary_cond = 'symmetric';
CostWindowMask = [];

if(isfield(params, 'UB'))
    UB = params.UB;
end
if(isfield(params, 'boundary_cond'))
    boundary_cond = params.boundary_cond;
end
if(isfield(params, 'CostWindowMask'))
    CostWindowMask = params.CostWindowMask; % empty means adaptive window i.e. only the valid region of the cost function is considered
end
% 
% if(~exist('SigmaC', 'var'))
%      SigmaC = 0;
% end
I1 = im2double(I1);
I2 = im2double(I2);

% 'DerivativeCheck','on', 'FinDiffType', 'central',
options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'on', 'DerivativeCheck','on', 'FinDiffType', 'central', 'UseParallel', 'always');
X0 = [UB/2, UB/2];
[SigEst,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) fnBETGaussObjL2(x, I1, I2, alpha, beta, lambda, boundary_cond, CostWindowMask),X0,[],[], [], [], [0, 0], [UB, UB], [], options);
% if(SigmaC > 0) 
%     if(SigEst < 0) % |sig1| > |sig2|
%         I1 = blurImg(I1, SigmaC, 'gaussian');
%     else
%         I2 = blurImg(I2, SigmaC, 'gaussian');
%     end
%     [SigEst,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) fnGaussObjSignedBlur(x, I1, I2, boundary_cond, CostWindowMask),sign(SigEst) * sqrt(abs(SigEst^2 + SigmaC^2)),[],[], [], [], -UB, UB, [], options);
%     PrevSgn = sign(SigEst);
%     SigEst = PrevSgn * sqrt(abs(SigEst^2 - SigmaC^2));
% end

H1est = GaussianKernel(SigEst(1));
H2est = GaussianKernel(SigEst(2));

result.FVAL = FVAL;
result.H1est = H1est;
result.H2est = H2est;
result.SigEst = SigEst;
result.ExitFlag = EXITFLAG;
result.Output = OUTPUT;
result.Lambda = LAMBDA;
result.Grad = GRAD;
result.Hessian = HESSIAN;
