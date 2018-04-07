function Res = EstDefocusFilterUnc(ImageDepthSet, params, filterBankSize)
% ImageDepthSet is the set of defocused image patches
% of the form A' * A where A is the convolution matrix for an image.

lambda = 1e5;
if(isfield(params, 'lambda'))
    lambda = params.lambda;
end
NFilters = 1;
if(isfield(params, 'NFilters'))
    NFilters = params.NFilters;
end
if(~exist('filterBankSize', 'var'))
    filterBankSize = 2;
end
NDepth = length(ImageDepthSet);
MatSize = size(ImageDepthSet{1,1}, 1);
KSize = sqrt(MatSize/filterBankSize);
Q = [];
Filters = cell(NDepth, filterBankSize * NFilters);

for idx = 1:NDepth
    AA = NDepth * lambda * ImageDepthSet{idx};
    
    for idx2 = 1:NDepth
       if(idx ~= idx2)
          AA = AA - ImageDepthSet{idx2}; 
       end
    end
    if(idx == 1)
        Q = blkdiag(sparse(AA));
    else
        Q = blkdiag(Q, sparse(AA));
    end
        
end

[U, S, V] = svd(full(Q)); %svds(Q, 1, 0);
for idxFilter = NFilters:-1:1
    for idx = 1:NDepth
        StartIdx = (idx - 1) * MatSize;
        IDX = StartIdx + (1:MatSize);
        F = V(IDX, idxFilter);
        f1 = reshape(F(1:KSize * KSize), KSize, KSize);
        f2 = reshape(F(KSize * KSize + 1:end), KSize, KSize);

        Filters{idx, 1 + 2*(idxFilter - 1)} = f1;
        Filters{idx, 2 + 2*(idxFilter - 1)} = f2;
    end
end
Res.Filters = Filters;
Res.Q = Q;
Res.U = U;
Res.S = S;
Res.V = V;