% VF is in the form of (n,2) each entry is (polar, eccentricity)
function xy = VF2cortex(VF, a, b, k)
    if nargin < 4
        k = sqrt(140)*0.873145;
        if nargin < 3
            b = 96.7; 
            if nargin < 2
                a = 0.635; 
            end
        end
    end
	n = size(VF,1);
	xy = zeros(n,2);
	w = dipole(VF(:,2),VF(:,1),a,b,k)-k*log(a/b);
	xy(:,1) = real(w);
	xy(:,2) = imag(w);
end

function w = dipole(ecc, polar, a, b, k)
    w = k*log((ecc.*exp(1j*polar.*f(ecc,polar,a))+a)./(ecc.*exp(1j*polar.*f(ecc,polar,b))+b));
end

function anisotropy = f(ecc, polar, ab)
    s1 = 0.76;
    s2 = 0.1821;
    anisotropy = cosh(polar).^(-2*s2./((ecc/ab).^s1 + (ecc/ab).^(-s1)));
    anisotropy(isnan(anisotropy)) = 0;
end
