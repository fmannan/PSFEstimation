function [Hest, SigEst, result] = EstGaussianPSFUnconstrained(Is, Ib, UB, boundary_cond, CostWindowMask, SigmaC)
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
Is = im2double(Is);
Ib = im2double(Ib);
if(SigmaC > 0) 
    Ib = defocusBlurImg(Ib, SigmaC, 'gaussian');
end
% 'DerivativeCheck','on', 'FinDiffType', 'central',
options = optimoptions(@fmincon, 'MaxIter', 10000, 'Algorithm', 'interior-point', 'MaxFunEvals', 40000, 'GradObj', 'on', 'UseParallel', 'always');
X0 = 2;
[SigEst,FVAL,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fminunc(@(x) GaussObj(x, Ib, Is, boundary_cond, CostWindowMask),X0,[],[], [], [], 0, UB, [], options);
if(SigmaC > 0) 
    SigEst = sqrt(SigEst^2 - SigmaC^2);
end
n = ceil((6 * SigEst - 1)/2);
[X, Y] = meshgrid(-n:n, -n:n);
Hest = 1/(2 * pi * SigEst^2) * exp(-(X.^2 + Y.^2)/(2 * SigEst^2));

result.FVAL = FVAL;
result.Hest = Hest;
result.SigEst = SigEst;
result.ExitFlag = EXITFLAG;
result.Output = OUTPUT;
result.Lambda = LAMBDA;
result.Grad = GRAD;
result.Hessian = HESSIAN;

function [g, gc] = GaussObj(sigma, Ib, Is, boundary_cond, CostWindowMask)

if(abs(sigma) < 1e-4)
   error('NOT IMPLEMENTED') 
end
n = ceil((6 * sigma - 1)/2);
[X, Y] = meshgrid(-n:n, -n:n);
H = 1/(2 * pi * sigma^2) * exp(-(X.^2 + Y.^2)/(2 * sigma^2));

if(isempty(CostWindowMask))
    CostWindowMask = zeros(size(Ib));
    CostWindowMask(n + 1:end - n, n + 1:end - n) = 1;
end
Ib_est = imfilter(Is, H, boundary_cond); %conv2(Is, H, 'same');
diff = (Ib_est(:) - Ib(:)) .* CostWindowMask(:);

g = 0.5 * sum(diff.^2);

Hgrad = H .* (X.^2 + Y.^2 - 2 * sigma^2) / sigma^3;
Ibgrad = imfilter(Is, Hgrad, boundary_cond); %conv2(Is, Hgrad, 'same');
gc = sum(diff .* Ibgrad(:));
