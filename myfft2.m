% [f,kM,lM,tM,km,lm] = ENfft2(G,x)
% 2D Fourier transform for a 2D elastic net map; dominant wave number,
% wavelength and stripe angle; and mean wave number and wavelength.
%
% We use the Fourier transform to obtain:
% - The power spectrum, with the zero-frequency in the centre.
% - The dominant wave numbers kM(1) (horizontal) and kM(2) (vertical).
% - The dominant wavelength lM in pixels of the stripes (defined as
%   orthogonal to the stripes), and
% - The angle tM that these stripes make with the vertical, defined so that
%   (when plotted with ENplot):
%   t =       -pi/2   -pi/4     0     pi/4    pi/2
%   stripes =   -       /       |       \       -
% - The mean wave number km.
% - The mean wavelength lm.
% To obtain the dominant wave number, the power of the zero-frequency
% component is disregarded.
%
% The 2D Fourier transform gives as a 2D wave number k such that (i = 1, 2)
% k(i) = # cycles in G(i) pixels = # cycles along one edge of the map.
% The polar angle of the k vector is NOT the angle of the stripes unless
% the grid is square. The stripe wavelength l and stripe angle t are
% computed as follows. Call l(i) = G(i)/k(i) the wavelength along each
% edge and m(i) = 1/l(i). Then:
%
%       l = 1 / sqrt(m²(1)+m²(2))     t = arctan(l(1)/l(2))
%
% t must be remapped to the [-pi/2,pi/2] interval.
%
% Note on the results:
% - Consider the following cases of waves:
%   . Plane waves of constant wavelength: the Fourier transform is a delta
%     function, and both the dominant wave number and stripe angle are well
%     defined.
%   . Isotropic waves (i.e., constant wavelength but not preferred
%     direction): the Fourier transform is a ring and the dominant wave
%     number is its radius. The stripe angle is not well defined. Both
%     ocular dominance and orientation maps usually fall in this category.
%   . More complex waves: there may not be a dominant wavelength and/or
%     stripe angle.
% - The Fourier transforms of x, sin(x) and cos(x) are NOT the same; in
%   particular, the pi/2 dephase between sin and cos gets lost depending
%   on what values x takes. Generally though, x and sin(x) have similar
%   2D Fourier transforms and that of cos(x) is rotated by +pi/2.
% - When either k(1) or k(2) is small (i.e., long waves in some direction),
%   the resulting l and t may be subject to numerical errors, since small
%   differences in k(i) translate into large differences in l and t.
%
% In:
%   G: 2D vector containing the list of grid lengths.
%   x: Mx1 vector containing a single map, with M = G(1)*G(2).
% Out:
%   f: Mx1 vector (with the same spatial structure as x) containing the
%      power of the 2D Fourier transform of x, shifted so that k = 0 is
%      the centre.
%   kM: 1x2 vector containing the dominant wave numbers along each
%      direction (k(1): horizontal; k(2): vertical). k(i) = # cycles in
%      G(i) pixels = # cycles along one edge of the map.
%   lM: positive number containing the dominant wavelength in pixels of the
%      stripes (defined as orthogonal to the stripes).
%   tM: the stripe angle in [-pi/2,pi/2], defined as above.
%   km: 1x2 vector containing the mean wave number modulus.
%   lm: positive number containing the mean wavelength in pixels.

% Copyright (c) 2002 by Miguel A. Carreira-Perpinan

function [f,kM,lM,tM,km,lm] = myfft2(G,x,Pi)

M = prod(G);
x(~logical(Pi)) = mean(x(logical(Pi)));
% ---------------------------- Fourier spectrum ----------------------------
% $$$ F = abs(fft2(reshape(x,G)));		% Square root of the power
F = fft2(reshape(x,G)); F = real(F.*conj(F));	% Power
% I prefer to use the power spectrum rather than its square root, because
% the latter overestimates the mean wave number (and so underestimates the
% mean wavelength; see below).
% For visualisation purposes the square-root power is better because it
% allows to see low-power values. However, this can also be achieved by
% inverting the colormap so that white = zero and black = maximum.

% Ditto but zero the DC component
F0 = F; 
F0(1,1) = 0;
% Shift k = 0 to centre for display
F = fftshift(F); F0 = fftshift(F0);
% Flatten it
f = reshape(F,M,1); f0 = reshape(F0,M,1);

[X,Y] = ndgrid(1:G(1),1:G(2));

% Normalised square-root spectra
p = f / sum(f);					% Original
p0 = f0 / sum(f0);				% Zero-DC

% Origin of the Fourier space: it can be computed as either the centre of
% the bitmap (the "geometric" origin) or the mean of the Fourier space
% with respect to the power spectrum. There is little practical difference
% between both, but I prefer the latter because it enforces the symmetry
% of the data.
% $$$ Fm = floor(G/2)+[1 1];		% "Geometric" origin of the bitmap
Fm = p'*[X(:) Y(:)];			% Mean of the Fourier space
% $$$ % Alternative way to compute the mean:
% $$$ sF1 = sum(F,1)'; sF2 = sum(F,2); sF = sum(sF1);
% $$$ Fm = [(1:G(1))*sF2 (1:G(2))*sF1] / sF;

X = X - Fm(1); Y = Y - Fm(2);		% Fourier space Cartesian coordinates

% --------------------------- Dominant wave number ---------------------------
% The dominant wave number is simply obtained as the (or any one of the)
% maxima of the power spectrum (making sure the zero-frequency component
% has no power):
[tmp1,tmp2] = max(f0); 
kM = [X(tmp2) Y(tmp2)];
% $$$ % Alternative way to compute the mode:
% $$$ [tmp1,tmp2] = max(f0); kM = [0 0]; [kM(1) kM(2)] = ind2sub(G,tmp2);
% $$$ kM = kM - Fm;			% Translate to origin

% Dominant wavelength 
lM = 1/norm(kM./G);

% Dominant stripe angle in [-pi/2,pi/2]
if kM(1) == 0
  tM = pi/2;
else
  tM = -atan(kM(2)*G(1)/(kM(1)*G(2)));
end

% In discrete variables, the value k = (k1,k2) that Matlab returns is the
% number of cycles along each direction (horizontal, vertical). Thus,
% it must be transformed into a wave number independent of the direction:
%   (k1/G1,k2/G2) = k./G.
X = X(:)/G(1); 
Y = Y(:)/G(2);
% These are the moduli of the transformed (k1,k2), i.e., k = sqrt{k1²+k2²}.
k = sqrt(sum([X Y].^2,2));

% ---------------------------- Mean wave number ----------------------------
% In the following discussion, k refers to the modulus k = sqrt{k1²+k2²}.
%
% The mean wave number is obtained as the average of the wave number
% modulus with respect to the power spectrum. In continuous variables,
% it is the following integral:
%   km = \int^{\infty}_{-\infty}{ p(k1,k2).k dk1.dk2 }
% in Cartesian coordinates (k1,k2), or
%   km = \int^{\infty}_0\int^{2\pi}_0{ p(k,t).k k.dk.dt }
% in polar coordinates (k,t).
% The power spectrum is assumed normalise to integrate to 1:
%   \int^{\infty}_{-\infty}{ p(k1,k2) dk1.dk2 } =
%   \int^{\infty}_0\int^{2\pi}_0{ p(k,t) k.dk.dt } = 1.
% Notes:
% - The eq. in the centre of p. 417 in Kaschub_00a is wrong (they have k
%   instead of k² in the polar coordinates integral).
% - I zero the DC component for data which don't sum to zero, since we are
%   not interested in the zero frequency.
% - One should average over k rather than over \lambda = 1/k because the
%   density is defined over k, not \lambda.
% - An issue that remains open is what density function to use, whether the
%   power, the square root of the power or something else. In practice, the
%   power gives better results than the square root of the power because it
%   downweights more the near-zero values -- but there is no theoretical
%   reason to prefer one over the other.
%
% However, the above integrals can't be applied directly since our sampling
% is much coarser for small k than for large k, which leads to
% overestimating km. This is because the discrete Fourier spectrum typically
% assigns small but nonzero values to all unrepresentative k values, and
% since there are many more values of t for large k in the Cartesian sample,
% the accumulated density is much higher for large k. In other words, if one
% plots the marginal distribution of k, it peaks at the dominant wave number
% but has a very heavy and long tail towards high k values; so the mean of k
% is far away from the dominant k, strongly biased towards high k (and so
% small wavelengths). The tail's effect is stronger:
% - For the square-root power than for the power, because the tail is
%   heavier.
% - When averaging on k than on \lambda = 1/k, because it quickly displaces
%   the \lambda-mean towards small values.
% - For non-plane wave maps (although for plane waves the estimates are
%   still slightly biased).
% You can check the distribution of k (weighted by the normalised, zero-DC
% power spectrum) as follows:
% $$$ [h,tmp,tmp,km] = ENhist(k,p0,100);
% $$$ figure(100); bar(h(:,2),h(:,1),1); title(['km = ' num2str(km,3)]);
% $$$ hold on; text(km,0,'km','VerticalAlignment','top'); hold off;
% ...and the corrected distribution as follows:
% $$$ [h,tmp,tmp,km] = ENhist(k,p0./k,100);
% $$$ figure(100); bar(h(:,2),h(:,1),1); title(['km = ' num2str(km,3)]);
% $$$ hold on; text(km,0,'km','VerticalAlignment','top'); hold off;
%
% Assuming p(k,t) has an elliptical shape, as typically happens for cortical
% maps, this problem can be corrected by one of the following methods:
% 1. Taking the mean of p(k,t) for every k and then using this density
%    q(k) = <p(k,t)> to average k over all k values.
% 2. Downweighting p(k,t) by k: q(k,t) = p(k,t)/k, and then using q(k,t) to
%    average k over all (k,t) values. The reason is that, assuming a
%    pixel has length d, then 2.pi.k/d pixels contribute to a value k
%    (the number of pixels in the circumference of radius k). Thus, the
%    average of p(k,t) for constant k will be \sum_t{p(k,t)} / (2.pi.k/d)
%    which is proportional to \sum_t{p(k,t)/k}, and this is the same as
%    using q(k,t) = p(k,t)/k (normalised to unit integral) as density.
%
% $$$ % This is without downweighting p(k,t) by k, which gives km very biased
% $$$ % towards high values:
% $$$ km = p0'*k;				% km is in cycles per pixel
% For the downweighted case, we make sure the k = 0 value doesn't interfere
% (since we discard the DC frequency).
k(find(k == 0)) = 1;				% Where k=0, p0=0 so p0/k = 0
km = 1 / sum(p0./k);				% km is in cycles per pixel
% $$$ % Since sum(p0) = 1, the former is the same as:
% $$$ km = (p0./k)'*k / sum(p0./k);

% Mean wavelength
lm = 1/km;					% Pixels

% Mean stripe angle in [-pi/2,pi/2]: it doesn't make sense (one could
% take it as the angle corresponding to the maximum value in the circle
% or radius km, but it isn't worth).

