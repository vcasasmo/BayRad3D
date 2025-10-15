# Code from the article "3D Activity Reconstruction from Angular Gamma Scanning via Variational Bayes: a proof of concept"

## Abstract  
This study serves as a **proof of concept** for a **Bayesian variational framework** enabling high-resolution 3D activity reconstruction in 220 L waste drums using **Angular Segmented Gamma Scanning (ASGS)** data and transmission-derived attenuation maps.  

Our proposed inference and uncertainty quantification approach is demonstrated using **virtual experiments** that simulate typical waste characterization scenarios. Computations are made tractable by using **stochastic variational inference (SVI)** together with a **multi-resolution spatial prior** to infer the spatial activity distribution.  

Results show that the approach can **recover the spatial activity distribution** within the considered drum, while also providing **more accurate total activity estimates** than conventional methods, thereby enhancing the accuracy of waste characterization.  

---

## Code Overview  
This repository provides the **implementation used to generate the results in the paper**.  
It leverages **NumPyro** (probabilistic programming on JAX) to perform Bayesian inference on simulated waste drums.  

Main features:
- Virtual **synthetic drum datasets** (Cs-137, Eu-152 and Co-60) available for testing.  
- **Variational inference (SVI)** with an **AutoNormal guide**.  
- Posterior predictive checks and other diagnostic tools included in the flow.  
- 2D and (optional) 3D visualization of reconstructed activity distributions.  

---

## Installation  
Dependencies (Python 3.10+):  
```bash
pip install jax jaxlib numpyro matplotlib seaborn pandas scipy SimpleITK
```
## Synthetic Data

The synthetic drum cases (`.pkl` files) are not included in this repository because of their size.

They can be downloaded from **Zenodo**:  
[https://zenodo.org/records/16736172](https://zenodo.org/records/16736172)

After downloading, place the files inside:
```bash
test/synthetic_cases/voxel_size_1.92/
```
The folder structure should look like:
```bash
test/
└── synthetic_cases/
└── voxel_size_1.92/
├── Cs137_1_1.pkl
├── Cs137_1_2.pkl
├── Eu152_1_2.pkl
└── ...
```

##Usage
1. Select a case
Inside `solver_svi_cartesian_HTP.py`, 
```bash
mockup_case = "Cs137_1_2"
isotope = "Cs137"
truth = "point"   # ground truth morphology: "point", "rod", or "homogeneous"
```
**Note:** Remember to adjust the parameters of the Beta distribution according to the drum nature:  
 - **Homogeneous drum:** `α = β = 10`  
 - **Hotspot drum:** `α = 0.1`, `β > 20`
2. Run Inference
```bash
python solver_svi_cartesian_HTP.py
```
## 3. Outputs

The script will produce outputs in the `results/` folder:

- **ELBO convergence plot** (`elbo_values.csv`, `elbo_plot.svg`)
- **Posterior predictive checks** (`post_pred_check_plot.svg`)
- **Recovered activity histograms** (`svi_hist_recovered_activity.svg`)
- **2D activity heatmaps** (`heatmap_XZ.svg`, `heatmap_XY.svg`)
-  **SVI Samples are stored in** (`f'samples_{mockup_case}.npy'`)
- **(Optional) 3D reconstructed volumes** (`.mhd`, via SimpleITK)

## Synthetic Drum Cases

| Name        | Activity (GBq) | RI    | Source Morphology                                                         | Matrix      | Materials        |
|------------ |----------------|-------|---------------------------------------------------------------------------|------------ |----------------|
| Cs137_1_1   | 3              | Cs137 | Homogeneously distributed                                                 | Homogeneous| Sand (1.52 g/cc)|
| Cs137_1_2   | 3              | Cs137 | Voxel @ [23,34,23], voxel_size = 1.92                                     | Homogeneous| Sand (1.52 g/cc)|
| Cs137_1_4   | 3              | Cs137 | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm                   | Homogeneous| Sand (1.52 g/cc)|
| Cs137_1_4_2 | 5              | Cs137 | Voxel @ [20,34,9], voxel_size = 1.92                                      | Homogeneous| Sand (1.52 g/cc)|
| EuCo_1_4    | 9              | Co60  | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm                   | Homogeneous| Sand (1.52 g/cc)|
|             | 5              | Eu152 | Voxel @ [20,34,9], voxel_size = 1.92                                      |            |                 |
| Eu152_1_1   | 5              | Eu152 | Homogeneously distributed                                                 | Homogeneous| Sand (1.52 g/cc)|
| Eu152_1_2   | 3              | Eu152 | Voxel @ [23,34,23], voxel_size = 1.92                                     | Homogeneous| Sand (1.52 g/cc)|
| Eu152_1_4   | 9              | Eu152 | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm                   | Homogeneous| Sand (1.52 g/cc)|



