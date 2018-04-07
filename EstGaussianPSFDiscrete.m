function [SigEstMean, SigEst, result] = EstGaussianPSFDiscrete(I1, I2, params)
% created on Jul 9, 2015
% linear search 
% Estimate Gaussian PSF fit using discrete linear search
% similar to DFDBasic but searches over discrete rel blur space and doesn't
% do any processing on the input image. 
nDivs = 100;
UB = 20;
LB = -UB;
boundary_cond = 'symmetric';
CostSmoothingWindow = [];
CostWindowMask = ones(size(I1, 1), size(I1, 2));
SigmaC = 0;

if(isfield(params, 'nDivs'))
    nDivs = params.nDivs;
end

if(isfield(params, 'UB'))
    UB = params.UB;
end
if(isfield(params, 'LB'))
    LB = params.LB;
end

if(isfield(params, 'boundary_cond'))
    boundary_cond = params.boundary_cond;
end

if(isfield(params, 'CostSmoothingWindow'))
    CostSmoothingWindow = params.CostSmoothingWindow;
end

if(isfield(params, 'CostWindowMask'))
    CostWindowMask = params.CostWindowMask;
end

if(isfield(params, 'SigmaC'))
     SigmaC = params.SigmaC;
end
I1 = im2double(I1);
I2 = im2double(I2);


SigR = linspace(LB, UB, nDivs);
Cost = nan(size(I1, 1), size(I1, 2), nDivs);
for sigRIdx = 1:nDivs
    if(SigR(sigRIdx) < 0) % |sig1| > |sig2|
        Is = I2;
        Ib = I1;
    else
        Is = I1;
        Ib = I2;
    end
    if(SigmaC > 0)
       Ib = blurImg(Ib, SigmaC, 'gaussian', boundary_cond, 0); 
    end
    Iblurred = blurImg(Is, SigR(sigRIdx), 'gaussian', boundary_cond, 0);
    cost2D = abs(Ib - Iblurred).^params.expK; % TODO: use an evaluator function NCC, SSD, SAD, etc
    if(~isempty(CostSmoothingWindow))
        cost2D = conv2(cost2D, CostSmoothingWindow, 'same');
    end
    Cost(:,:,sigRIdx) = cost2D;
end
[Y, I] = min(Cost, [], 3);
result.FVAL = Y;
result.Idx = I;
result.SigRLabels = SigR;
result.Cost = Cost;
result.SigEst = SigR(I);
SigEst = result.SigEst;
SigEstMean = mean(mean(result.SigEst(logical(CostWindowMask))));