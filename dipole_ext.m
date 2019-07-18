function w = dipole_ext(ecc, polar, a, b, k)
    w = k*log((ecc.*exp(1j*polar.*f(ecc,polar,a))+a)./(ecc.*exp(1j*polar.*f(ecc,polar,b))+b));
end
function anisotropy = f(ecc, polar, ab)
    s1 = 0.76;
    s2 = 0.18;
    anisotropy = cosh(polar).^(-2*s2./((ecc/ab).^s1 + (ecc/ab).^(-s1)));
    anisotropy(isnan(anisotropy)) = 0;
end