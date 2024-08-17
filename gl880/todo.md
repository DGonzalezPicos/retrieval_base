## Retrievals of M dwarfs
- [ ] Fit orders 47, 48 with CO and H2O
- [ ] Recompute opacities to 5000 K
    - [ ] Na, K, ...?


## Gl 880
- [ ] J band orders: 30,31,32
    - Missing opacity: Fe, Mg, Cr, Sc
- [ ] K band orders: 45,46,47,48
    - Missing opacity: Isotopologues of CO, H2O? Fe, Mg, U
    - [ ] Implement matching vertical profile of H2(18)O to H2O

- ~~[ ] SPHINX retrievals: use PT and chemistry interpolators for SPHINX models~~
- [ ] Line Spread Function: Fit for FWHM and Gamma of Gaussian and Lorentzian profiles?
- [ ] Add  $\alpha(X)$ free parameters as log X = log X (SPHINX) + $\alpha(X)$
- [ ] Instrumental broadening of SPIRou: Gaussian of std. dev. 1.83 km/s (equiv. to FWHM = 4.3 km/s <-> R=69k, see Cristofari+2022)
    - [ ] Additional broadening due to macroturbulence? probably negligible...