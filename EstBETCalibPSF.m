function [OptIdx, res] = EstBETCalibPSF(Inear, Ifar, Knear, Kfar, expK, alpha, beta, lambda, boundary_type, NearMagScale, CostWindow)
% find the right PSF pair from a set of calibrated PSFs
NDepths = length(Knear);
Cost = nan(1, NDepths); 
if(~exist('CostWindow', 'var'))
   CostWindow = ones(size(Inear)); 
end
parfor idx = 1:NDepths
    if(abs(NearMagScale - 1) < 1e-6)
        KnearScaled = Knear{idx};
    else
        KnearScaled = imresize(Knear{idx}, NearMagScale);
    end
    Cost(idx) = fnCostBET(Inear, Ifar, KnearScaled, Kfar{idx}, expK, alpha, beta, lambda, boundary_type, CostWindow);
end
[~, OptIdx] = min(Cost);
res.Cost = Cost;
res.OptIdx = OptIdx;