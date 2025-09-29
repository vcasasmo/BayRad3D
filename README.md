# 3D Activity Reconstruction from Angular Gamma Scanning with Variational Bayes  

## Abstract  
This study serves as a **proof of concept** for a **Bayesian variational framework** enabling high-resolution 3D activity reconstruction in 220 L waste drums using **Angular Segmented Gamma Scanning (ASGS)** data and transmission-derived attenuation maps.  

Our proposed inference and uncertainty quantification approach is demonstrated using **virtual experiments** that simulate typical waste characterization scenarios. Computations are made tractable by using **stochastic variational inference (SVI)** together with a **multi-resolution spatial prior** to infer the spatial activity distribution.  

Results show that the approach can **recover the spatial activity distribution** within the considered drum, while also providing **more accurate total activity estimates** than conventional methods, thereby enhancing the accuracy of waste characterization.  

---

## Code Overview  
This repository provides the **implementation used to generate the results in the paper**.  
It leverages **NumPyro** (probabilistic programming on JAX) to perform Bayesian inference on simulated waste drums.  

Main features:
- Virtual **synthetic drum datasets** (Cs-137, Eu-152, Co-60).  
- **Hierarchical forward model** linking activity fields, detector efficiencies, and emission probabilities.  
- **Variational inference (SVI)** with an **AutoNormal guide**.  
- Posterior predictive checks and uncertainty quantification.  
- 2D and 3D visualization of reconstructed activity distributions.  

---

## Installation  
Dependencies (Python 3.10+):  
```bash
pip install jax jaxlib numpyro matplotlib seaborn pandas scipy SimpleITK
