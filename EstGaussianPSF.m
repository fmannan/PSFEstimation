function [Hest, SigEst, result] = EstGaussianPSF(I1, I2, LB, UB, ...
                                                 boundary_cond, CostWindowMask, SigmaC, GradObj)
if(~exist('LB', 'var'))
    LB = -20;
end
if(~exist('UB', 'var'))
    UB = 20;
end
if(~exist('boundary_cond', 'var'))
    boundary_cond = 'symmetric';
end
if(~exist('CostWindowMask', 'var'))
    CostWindowMask = []; % empty means adaptive window i.e. only the valid region of the cost function is considered
end

if(~exist('SigmaC', 'var'))
     SigmaC = 0;
end
I1 = im2double(I1);
I2 = im2double(I2);

fnGaussianKernel = @GaussianKernel;

X0 = min((LB + UB)/2 + 2, UB);

% 'DerivativeCheck','on',
if(~exist('GradObj', 'var'))
    GradObj = 'off';  %
end

options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', GradObj,  'FinDiffType', 'central', 'UseParallel', 'always');
%tic
[SigEst,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) fnGaussObjSignedBlur(x, I1, I2, SigmaC, boundary_cond, CostWindowMask, fnGaussianKernel),X0,[],[], [], [], LB, UB, [], options);
%toc

% LAMBDA = nan;
% options = optimoptions('fminunc', 'MaxIter', 10000, 'Algorithm', 'quasi-newton', 'MaxFunEvals', 40000, 'GradObj', 'off', 'FinDiffType', 'central');
% %tic
% [SigEst,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(@(x) fnGaussObjSignedBlur(x, I1, I2, SigmaC, boundary_cond, CostWindowMask, fnGaussianKernel), X0, options);
% %toc

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

Hest = GaussianKernel(abs(SigEst));

result.FVAL = FVAL;
result.Hest = Hest;
result.SigEst = SigEst;
result.ExitFlag = EXITFLAG;
result.Output = OUTPUT;
result.Lambda = LAMBDA;
result.Grad = GRAD;
result.Hessian = HESSIAN;

