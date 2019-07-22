% [r,c,linregr] = ENcorr(A)
% Pairwise Pearson's correlation coefficient and angle cosine for several
% data lists, plus linear regression coefficients
%
% For each pair of columns x, y of the matrix A, ENcorr computes Pearson's
% correlation coefficient r and the angle cosine c:
%
%           cov(x,y)                                x.y            ^
%   r = -----------------                   c = ----------- = cos(x,y)
%       stdev(x) stdev(y)                       ||x|| ||y||
%
% Both r and c are in [-1,1]; but if both x and y >=0 then c is in [0,1].
% Both r and c are invariant to scale changes; r is also invariant to
% translations, but c is not.
% If x and y are zero-mean, then r = c.
%
% linregr contains the coefficients of the linear regressions Y|X and X|Y
% (i.e., vertical and horizontal errors, respectively) and X,Y (i.e.,
% orthogonal errors), so the regression lines can be plotted.
%
% In:
%   A: NxM array of real numbers. Each of its M columns is taken as a
%      list of N elements. There are M(M-1)/2 pairwise combinations of
%      these lists.
% Out:
%   r: MxM matrix with r(i,j) = Pearson's correlation coefficient for
%      rows i, j. Since r(i,i) = 1 and r is symmetric, only the lower (or
%      upper) triangular part of A is useful.
%      If A(:,i) = constant, then r(i,:) = NaN.
%   c: ditto for the angle cosine.
%      If A(:,i) = 0, then c(i,:) = NaN.
%   linregr: MxM structure array. For i=2:M and j=1:i-1, linregr(i,j)
%      corresponds to the regression between variables x=A(i), y=A(j) and
%      has fields 'mean', 'cov', 'vxy' (director vector for the regression
%      i->j), 'vyx' (ditto j->i), 'v' (ditto orthogonal) and 'r'.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function [r,c,linregr] = v1Corr(A)
[N,M] = size(A);
As2 = sum(A.^2,1,'omitnan'); 
nn = sum(~isnan(A),1);
As = sqrt(As2); 
As(As==0) = NaN;
Am = mean(A,1,'omitnan'); 
Asd = sqrt(As2./nn-Am.^2); 
Asd(Asd==0) = NaN;
A01 = (A - Am(ones(N,1),:)) ./ Asd(ones(N,1),:);    % Zero-mean, unit-var
A1 = A ./ As(ones(N,1),:);    % Unit-norm

% r, c
r = ones(M,M); c = r;
for i=2:M
    for j=1:i-1
        r(i,j) = A01(:,i)'*A01(:,j)/N;
        r(j,i) = r(i,j);
        c(i,j) = A1(:,i)'*A1(:,j);
        c(j,i) = c(i,j);
    end
end

% linregr
linregr = struct('mean',[],'cov',[],'vxy',[],'vyx',[],'v',[],'r',[]);
for i=2:M
    for j=1:i-1
        linregr(i,j).mean = Am([i j])';
        tmpcov = cov(A(:,[i j]),1,'omitrows');
        linregr(i,j).cov = tmpcov;
        if tmpcov(1,2) == 0
            linregr(i,j).vxy = [0;1];
            linregr(i,j).vyx = [1;0];
            if det(tmpcov(1,1)) == 0
                linregr(i,j).r = NaN;
            else
                linregr(i,j).r = 0;
            end
        else
            linregr(i,j).vxy = [1;tmpcov(1,2)/tmpcov(1,1)];
            linregr(i,j).vyx = [tmpcov(1,2)/tmpcov(2,2);1];
            linregr(i,j).r = tmpcov(1,2)/sqrt(prod(diag(tmpcov)));
        end
        [tmpV,tmpD] = eig(tmpcov);
        [tmp1,tmp2] = max(abs(diag(tmpD)));
        v = tmpV(:,tmp2);
        [tmp1,tmp2] = max(abs(v));
        v = v/tmp1*sign(v(tmp2));
        linregr(i,j).v = v;
    end
end
