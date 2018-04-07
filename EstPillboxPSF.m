function [Hest, SigEst, result] = EstPillboxPSF(I1, I2, LB, UB, boundary_cond, CostWindowMask, SigmaC)
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

fnKernel = @PillboxKernel;

X0 = min((LB + UB)/2 + 2, UB);

% 'DerivativeCheck','on',
GradObj = 'on'; % 'off';
options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', GradObj,  'FinDiffType', 'central', 'UseParallel', 'always');
%tic
[SigEst,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(@(x) fnPillboxObjSignedBlur(x, I1, I2, SigmaC, boundary_cond, CostWindowMask, fnKernel),X0,[],[], [], [], LB, UB, [], options);
%toc


Hest = fnKernel(abs(SigEst));

result.FVAL = FVAL;
result.Hest = Hest;
result.SigEst = SigEst;
result.ExitFlag = EXITFLAG;
result.Output = OUTPUT;
result.Lambda = LAMBDA;
result.Grad = GRAD;
result.Hessian = HESSIAN;

