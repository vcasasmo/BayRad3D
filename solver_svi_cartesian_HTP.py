import os
import numpyro
import numpyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
 


from jax import random
#jax.config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import pandas as pd
import pickle

import seaborn as sns 
from contextlib import redirect_stdout

from numpyro.infer.autoguide import AutoNormal
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam
"""Environment variables for device setting"""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
plt.rcParams['font.size'] = 12 
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = [ 'Times', 'DejaVu Serif']

class SynthWasteBatch:
    def __init__(self):
        self.mesh = None
        self.spectrum_analyzed = None
        self.energies_detected = [] 
        self.detector_efficiencies = None
        self.isotopes_detected = []
        self.transmission = False

    def save_waste_batch(self, filename):
        """
        Saves the current instance of WasteBatch to a file.
        :param filename: Path to save the file.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

def load_pickle_file(file_path):
    """
    Load a pickle file and return the data.
    
    :param file_path: Path to the .pkl file.
    :return: Data loaded from the file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        print(f"Successfully loaded data from {file_path}")
    # Print the type of data and inspect its structure if successfully loaded
    if data is not None:
        print(f"Data type: {type(data)}")
        if isinstance(data, SynthWasteBatch):
            print("Loaded SynthWasteBatch object:")
            print(f"Mesh shape: {data.mesh.shape if data.mesh is not None else None}")
            print(f"Spectrum analyzed: {data.spectrum_analyzed.shape if data.spectrum_analyzed is not None else None}")
            print(f"Energies detected: {data.energies_detected}")
            print(f"Detector efficiencies shape: {data.detector_efficiencies.shape if data.detector_efficiencies is not None else None}")
            print(f"Isotopes detected: {data.isotopes_detected}")
            print(f"Transmission shape: {data.transmission.shape if isinstance(data.transmission, np.ndarray) else None}")
        else:
            print("Unexpected data structure:", data)


    return data







def init_solver(waste_batch,isotope):
    isotopes = waste_batch.isotopes_detected

    #for isotope in isotopes:
  
    #get positions in lists for isotopes
    energy_indices = [i for i, element in enumerate(isotopes) if element == isotope]
    energies = np.array(waste_batch.energies_detected)[energy_indices]
    transmission = waste_batch.transmission[:,:,:,:,energy_indices]
    #det_efficiency =  abs(waste_batch.detector_efficiencies[:, :, :, :, energy_indices, 0])
    det_efficiency =  jnp.clip(waste_batch.detector_efficiencies[:, :, :, :, energy_indices, 0],a_min =0)
    det_efficiency_var = abs(waste_batch.detector_efficiencies[:, :, :, :, energy_indices,1])
    #det_efficiency= np.flip(det_efficiency,axis =(1,3))
    
    w = det_efficiency *transmission
 
    
     # The drum spins 180 deg during operation 
    #TODO: add time vector 
    t  = np.ones([len(energy_indices)]) #temporal fix
    #Load emission probabilities from the database
    emission_db_filepath = f'data/gammaspec/{isotope}.csv'
    gammaspec_data = pd.read_csv(emission_db_filepath)
    tolerance = 0.1  # Allowable difference
    matching_indices = gammaspec_data['energy'].apply(
    lambda x: any(abs(x - target) <= tolerance for target in energies))
    filtered_data = gammaspec_data[matching_indices]
    eta = filtered_data['intensity_%'].values/100
    obs = waste_batch.spectrum_analyzed[energy_indices,:]
    """ numpy2jax"""
    obs = jax.device_put(jnp.array(obs.flatten())) 
    # Send `obs` to the GPU
    w = jax.device_put(jnp.array(w))   

    eta = jax.device_put(jnp.array(eta)) 
    t = 1
    t   = jax.device_put(jnp.array(t)) 
    #TEMPORAL FOR NOISE FREE
    #obs = jax.device_put(jnp.load('obs_noise.npy').squeeze())  
    return obs,t,eta,w,det_efficiency_var,det_efficiency,transmission
    
def mask_generation(mesh):
    """Generates a mask for voxels inside or touching a cylinder of radius R and height H centered at the origin.
    Now it is hard coded but in the future can be modifyied to allow for flexibility"""
    # Cylinder dimensions
    R = 287  # Cylinder radius
    H = 885  # Cylinder height

    # Extract mesh coordinates and spacing
    x = mesh['x'].values
    y = mesh['y'].values
    z = mesh['z'].values

    # Create a grid of voxel centers (xw, y, z)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Condition for the cylinder:
    # - The radial distance sqrt(x^2 + z^2) must be less than or equal to R.
    # - The y-coordinate (vertical) must lie within [-H/2, H/2] (since the cylinder is centered at the origin).
    condition = (np.sqrt(xx**2 + zz**2) <= R) & (np.abs(yy) <= H / 2)
    mask = np.where(condition, 1, 0).astype(np.uint8)

    return mask
 

 

 
def posterior_predictive_check(y_pred, obs, results_path, Save='Yes'):

    
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # Calculate posterio stats
    y_pred_mean = jnp.mean(y_pred, axis=0)
    y_pred_lower = jnp.percentile(y_pred, 2.5, axis=0)
    y_pred_upper = jnp.percentile(y_pred, 97.5, axis=0)

    # Obs sort
    sorted_idx = jnp.argsort(obs)
    obs_sorted = obs[sorted_idx]
    x_mean_sorted = y_pred_mean[sorted_idx]
    x_lower_sorted = y_pred_lower[sorted_idx]
    x_upper_sorted = y_pred_upper[sorted_idx]

    # 95% CI shadowing
    plt.plot(x_mean_sorted, obs_sorted, 'o', color='blue', label='Predictions')
    plt.fill_betweenx(
        obs_sorted,
        x_lower_sorted,
        x_upper_sorted,
        color='blue',
        alpha=0.3,
        label='95% CI'
    )

    # Perfect Prediction Line (x = y)
    min_val = min(jnp.min(obs_sorted), jnp.min(x_mean_sorted))
    max_val = max(jnp.max(obs_sorted), jnp.max(x_mean_sorted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

  
    plt.xlabel("Posterior Predicted Counts (cps)")
    plt.ylabel("Observed Counts (cps)")
    plt.legend()

    # Save results
    if Save == 'Yes':
        plt.savefig(f'{results_path}/post_pred_check_plot.svg', format='svg', dpi=600)
        datasave = pd.DataFrame({
            'obs': obs,
            'y_pred_mean': y_pred_mean,
            'y_pred_lower': y_pred_lower,
            'y_pred_upper': y_pred_upper
        })
        datasave.to_csv(f'{results_path}/post_pred_check.csv', index=False)

    plt.show()

def total_recovered_activity(A_samples,prior_samples,results_path,Save='Yes',averaged_activity=None,actual_value=3.0):
    # Plot histogram
    #prior_activity = jnp.sum(prior_samples, axis=(1)) / 1e9
  
    activity_sum = jnp.sum(A_samples, axis=(1,2,3)) / 1e9
  
    
    plt.style.use("bmh")
    sns.histplot(activity_sum, bins=80, kde=True, stat="density", alpha=0.65, color="blue")
    #sns.histplot(prior_activity, bins=80, kde=True, stat="density", alpha=0.65, color="yellow")
    # Add a vertical line for the actual value
    plt.axvline(actual_value , color='red', linestyle='--', linewidth=2, label="Truth")
    plt.axvline(averaged_activity/1e9 , color='black', linestyle='--', linewidth=2, label="Averaged")
    plt.legend()
    
    # Customize plot
    plt.xlabel("Retrieved Activity (GBq)")
    plt.ylabel("Density")
 
    plt.grid(False)
    if Save=='Yes':
        plt.savefig(f'{results_path}/svi_hist_recovered_activity.svg',format = 'svg', dpi =1200)
        
def summary(results_path):
    with open(f"{results_path}/svi_summary.txt", "w") as f:
        with redirect_stdout(f):
            svi.print_summary()
            


 
 

 
def forward_model(w, eta, t,mask,obs=None):
   

    #total inferred parameters 49152
    
    
    #number of energy peaks matches w 4th dimension:
    num_peaks = w.shape[4]            
    # Define original grid size
    I_orig, J_orig, K_orig = 31, 48, 31
  
    # Define downsampled grid size with padding
    I_padded = I_orig + (I_orig % 3)
    J_padded = J_orig + (J_orig % 3)
    K_padded = K_orig + (K_orig % 3)
    
    I_coarse = I_padded // 2
    J_coarse = J_padded // 2
    K_coarse = K_padded // 2
     
    #magnitude inference for the overall drum
    magnitude =10**numpyro.sample("magnitude",dist.Uniform(4,12))
 
    # Coarse Level A field
    
    alpha = 0.1 # 0.1  #0.1
    beta  = 20 #20 #20
    

    A_flat = numpyro.sample("A_coarse", dist.Beta(alpha, beta).expand([I_coarse, J_coarse, K_coarse])) 
    
    A_coarse = A_flat.reshape((I_coarse, J_coarse, K_coarse))
    
    
    # Expand to fine grid with hierarchical inference
    I_fine, J_fine, K_fine = I_coarse * 2, J_coarse * 2, K_coarse * 2
    
   
    # Broadcast coarse values to fine grid shape
    A_coarse_broadcasted = jnp.kron(A_coarse, jnp.ones((2, 2, 2)))
    #Normalize Beta level
    A_coarse_broadcasted /= jnp.sum(A_coarse_broadcasted)
    
    # Sample fine voxels conditioned on their coarse parent:
    with numpyro.plate("fine_voxels", I_fine * J_fine * K_fine):
        A_fine_flat = numpyro.sample(
             "A_fine",  dist.HalfNormal(1)) *A_coarse_broadcasted.flatten()
        
   
   
    A_fine_full =A_fine_flat.reshape((I_fine, J_fine, K_fine))
    """
    Mask used for comparing ideal results vs. inferred by forcing inference only
    in voxels of the ground truth. For instance for Cs137_1_2.pkl:
    infer_mask = jnp.zeros((I_orig, J_orig, K_orig))
    infer_mask = infer_mask.at[23, 34, 23].set(1.0)
    """
    # Crop to original size
    A_fine = A_fine_full[:I_orig, :J_orig, :K_orig] * mask  #*infer_mask

    numpyro.deterministic('A_field', A_fine)
    #Scaling for inferred magnitude
    A_fine*=magnitude 
    
    numpyro.deterministic('magnit', magnitude)
    
    # Save fine-level field
    numpyro.deterministic("A", A_fine)
    # Stack multiplicity for  handling multiple peaks
    A_expanded = jnp.stack([A_fine] * num_peaks, axis=-1)  
 
    # Compute forward model
    
    s_nm = t * eta * jnp.sum(w * A_expanded, axis=(1, 2, 3))
    s_nm = s_nm.T.flatten()     
    s_nm = s_nm+0.01
    # Noise to account for model error
    noise =  numpyro.sample("noise",dist.Gamma(2,1))
    numpyro.deterministic('noise_check', noise)
    
    with numpyro.plate('data', s_nm.shape[0]):
        return numpyro.sample(
    "Y_obs",
     
    dist.TruncatedNormal(s_nm,jnp.sqrt(s_nm) * noise,low =0),
    obs=obs
)

 

       


"""data_preparation"""
#About mockup_drum filenames:# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Case Definitions
# Name         | Activity (GBq) | RI     | Source Morphology                               | Matrix       | Materials
# -----------------------------------------------------------------------------
# Cs137_1_1    | 3              | Cs137  | Homogeneously distributed                        | Homogeneous  | Sand (1.52 g/cc)
# Cs137_1_2    | 3              | Cs137  | Voxel @ [23,34,23] voxel_size = 1.92             | Homogeneous  | Sand (1.52 g/cc)
# Cs137_1_4    | 3              | Cs137  | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm | Homogeneous | Sand (1.52 g/cc)
# Cs137_1_4_2  | 5              | Cs137  | Voxel @ [20,34,9] voxel_size = 1.92              | Homogeneous  | Sand (1.52 g/cc) | 
# EuCo_1_4     | 9              | Co60   | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm | Homogeneous | Sand (1.52 g/cc)
#              | 5              | Eu152  | Voxel @ [20,34,9] voxel_size = 1.92              |              | 
# Eu152_1_1    | 5              | Eu152  | Homogeneously distributed                        | Homogeneous  | Sand (1.52 g/cc)
# Eu152_1_2    | 3              | Eu152  | Voxel @ [23,34,23] voxel_size = 1.92             | Homogeneous  | Sand (1.52 g/cc)
# Eu152_1_4    | 9              | Eu152  | Rod: R=1 cm, Rot[38,10,0]°, Pos[-4.0,3.0,-5.0], L=19 cm | Homogeneous | Sand (1.52 g/cc)
# ───────────────────────────────────────────────────────────────────────────────────────────────────────────


mockup_case='Cs137_1_2'
mockup_path= f'./test/synthetic_cases/voxel_size_1.92/{mockup_case}.pkl'
samples_file =f'samples_{mockup_case}.npy' #saves the samples for posterior analysis
results_path = './results'
isotope = 'Cs137'
truth='point' #rod, point,point2, homo
""" 'homo' tag in truth is used for no annotations in the xa/xy plots
'rod' will draw the rod location of the rod location cases
'point' is coordinates [23,34,23] for Hotspot 1 and 'point2' is [20,34,9] for Hotspot 2 
"""
actual_value =3
num_iterations = 160000

#mask = np.load('mask.npy')


mockup_drum =load_pickle_file(mockup_path)

obs,t,eta,w,det_efficiency_var,det_efficiency,transmission= init_solver(mockup_drum,isotope)
I, J, K = [31, 48, 31]  # Dimensions of the voxel grid
t *= 1 
obs = obs*t
mesh =mockup_drum.mesh
mask = mask_generation(mesh=mesh)
mask_indices=np.where(mask==1)


#"Classical" way of obtaining Average Total Activity
mask_broadcast = mask[None,:,:,:,None]
mask_broadcasted = jnp.broadcast_to(mask_broadcast, w.shape)

masked_w = w[mask_broadcasted == 1]
averaged_activity = jnp.mean(obs)/(t*jnp.mean(masked_w )*jnp.mean(eta))
print(f"traditional: {averaged_activity/1e9}")

"""SVI"""
#from numpyro.infer import init_to_mean,init_to_median,init_to_value,init_to_sample
 
guide = AutoNormal(forward_model )
# Initialize SVI

optimizer = Adam(step_size=0.001)  # You can tune the step size
svi = SVI(forward_model, guide, optimizer, loss=Trace_ELBO(num_particles=10))


# Run SVI
# Number of optimization steps
svi_result = svi.run(
    random.PRNGKey(0),
    num_iterations,
    obs=obs,
    w=w,
    eta=eta,
    t=t,
    mask=mask
  
)

# Extract variational parameters
params = svi_result.params
elbo_values = svi_result.losses
elbo_np = np.array(elbo_values)

# Guardar como CSV
df = pd.DataFrame(elbo_np)
df.to_csv("elbo_values.csv", index=False)
# ELBO check

import matplotlib.ticker as ticker
plt.plot(elbo_values)
plt.yscale("log")
plt.xlabel('Iteration')
plt.ylabel('ELBO')
# Use scientific notation
ax = plt.gca()

ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.xaxis.get_major_formatter().set_scientific(True)
ax.xaxis.get_major_formatter().set_powerlimits((-1, 1))

 
plt.tight_layout()
plt.show()


"""Sample collection""" 




# Posterior predictive check
predictive = numpyro.infer.Predictive(model=forward_model,guide=guide,num_samples=1000)
 
samples= predictive(random.PRNGKey(1), w=w, eta=eta, t=t, mask=mask,obs=None) 
prior_samples= samples['A']
# Posterior predictive sampling
predictive = numpyro.infer.Predictive(
    model=forward_model,  # Forward model
    guide=guide,          # Variational guide
    num_samples=1000,     # Number of posterior samples
    params=params         # Fitted parameters from SVI
)

# Generate posterior predictive samples
posterior_samples = predictive(
    random.PRNGKey(1), 
    w=w, 
    eta=eta, 
    t=t, 
    mask=mask

)
from scipy.stats import pearsonr
# Extract predictions and observed data
posterior_predictions = posterior_samples['Y_obs']  # Shape: (num_samples, data_points)
observed_data = obs  # Observed data
A_retrieved =posterior_samples['A']
magnitude = posterior_samples['magnit']
Activity_field = jnp.sum(posterior_samples['A_field'],axis=(1,2,3))
# Compute Pearson correlation on log-transformed values
log_A = np.log10(Activity_field)
log_mag = np.log10(magnitude)
r, p_value = pearsonr(log_A, log_mag)
# Plot
plt.figure(figsize=(8, 6))
plt.scatter(Activity_field, magnitude, alpha=0.5, s=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('A (log scale)')
plt.ylabel('Magnitude (log scale)')
plt.title('Posterior Samples: A vs Magnitude (log-log)')

# Annotate the correlation coefficient
plt.text(
    0.05, 0.95,
    f"$r$ = {r:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top'
)

plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()
# Compute summary statistics for predictions
posterior_mean = jnp.mean(posterior_predictions, axis=0)
posterior_std = jnp.std(posterior_predictions, axis=0)

# Plot observed vs predicted distributions
plt.figure(figsize=(10, 6))
sns.histplot(
    posterior_mean, 
    bins=50, 
    kde=True, 
    color="blue", 
    label="Posterior Mean Prediction"
)
sns.histplot(
    observed_data, 
    bins=50, 
    kde=True, 
    color="orange", 
    label="Observed Data", 
    alpha=0.7
)
plt.axvline(jnp.mean(observed_data), color='red', linestyle='--', label="Observed Mean")
plt.xlabel("Value")
plt.ylabel("Density")

plt.legend()
plt.show()

# Calculate and print discrepancies
mean_discrepancy = jnp.mean(jnp.abs(posterior_mean - observed_data))
std_discrepancy = jnp.mean(jnp.abs(posterior_std - jnp.std(observed_data)))
print(f"Mean Discrepancy: {mean_discrepancy:.4f}")
print(f"Standard Deviation Discrepancy: {std_discrepancy:.4f}")

# Plot predictive intervals (optional)
lower = jnp.percentile(posterior_predictions, 2.5, axis=0)
upper = jnp.percentile(posterior_predictions, 97.5, axis=0)

plt.figure(figsize=(12, 6))
plt.plot(observed_data, label="Observed Data", color="black", linewidth=2)
plt.fill_between(
    jnp.arange(len(lower)), lower, upper, color="blue", alpha=0.3, label="95% Predictive Interval"
)
plt.plot(posterior_mean, label="Posterior Mean", color="blue", linestyle="--")
plt.xlabel("Detector Index")
plt.ylabel("Counts (cps)")
 
plt.legend()
 
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.tight_layout()
plt.show()




A_samples = A_retrieved
A_samples = A_samples.reshape(1000,I,J,K)
A_median = jnp.median(A_samples, axis=0)
A_mean = jnp.mean(A_samples, axis=0)
 
 
np.save(samples_file,A_samples)

                
"""Diagnostics"""
"""Posterior Predictive Check"""
#predictive = numpyro.infer.util.Predictive(forward_model,posterior_samples_ungrouped)
y_pred =  predictive(random.PRNGKey(1), w, eta, t,mask)['Y_obs'] 
#summary(results_path)
#convergence(results_path)
posterior_predictive_check(y_pred, obs,results_path,'Yes')
e  = obs-y_pred
e = e.squeeze()
rmse = np.sqrt(np.sum(e**2)/len(e))
 
total_recovered_activity((A_retrieved),prior_samples, results_path,'Yes', averaged_activity,actual_value)
 




#Plotting
# Function to plot heatmap with a circle in the xz plane with major and minor ticks, inverted Y-axis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm
import jax.numpy as jnp
 
def plot_xz_heatmap(
    A_mean, circle_radius, x_range, grid_size, major_tick_interval,
    truth_type='point', cross_coords=None, cross_ends=None # 'cylinder', 'point', or 'both'
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.colors import LogNorm

    y_max_idx = jnp.argmax(jnp.mean(A_mean, axis=(0, 2)))  # Max Y index
    A_slice = A_mean[:, y_max_idx, :]  # XZ plane slice

    # Ground-truth coordinates (X, Y, Z) 
    #cross_coords = np.array([153.6,201.6,153.6])#Cs137 voxel
    #cross_coords = np.array([96,201.6,-115.2])#Eu152 voxel
    #cross_coords = np.array([-45, 90, -14]) #rod
    #cross_ends = np.array([-10, -90, 150])  #rod 
    thickness = 20  # mm, rod

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    c = ax.imshow(
        A_slice, cmap='viridis', aspect='auto',
        extent=[x_range[0], x_range[1], x_range[1], x_range[0]],
        norm=LogNorm(vmin=1e4,vmax=1e7)
    )
    plt.colorbar(c, label='Activity (Bq)')

    # Draw reference circle
    circle = plt.Circle((0, 0), circle_radius, color='black', fill=False, linewidth=2)
    ax.add_artist(circle)

    # Rectangle
    if truth_type in ('cylinder', 'both'):
        p_start = np.array([cross_coords[2], cross_coords[0]])  # (Z, X)
        p_end = np.array([cross_ends[2], cross_ends[0]])         # (Z, X)

        direction = p_end - p_start
        length = np.linalg.norm(direction)
        unit_dir = direction / length
        perp = np.array([-unit_dir[1], unit_dir[0]])
        offset = (thickness / 2) * perp

        rect_coords = np.array([
            p_start + offset,
            p_start - offset,
            p_end - offset,
            p_end + offset
        ])

        polygon = Polygon(rect_coords, closed=True, edgecolor='black', fill=False, linewidth=1.5)
        ax.add_patch(polygon)

    # Cross marker
    if truth_type in ('point', 'both'):
        z, x = cross_coords[2], cross_coords[0]
        ax.plot(z, x, marker='x', color='black', markersize=10, markeredgewidth=2)
    if truth_type == 'homo':
        pass
    # Grid and ticks
    major_ticks = np.arange(x_range[0], x_range[1], major_tick_interval)
    minor_ticks = np.arange(x_range[0], x_range[1], grid_size)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, which='both')
    ax.invert_yaxis()

    plt.title(f'XZ plane at Y index {y_max_idx}', fontsize=14)
    plt.xlabel('Z (mm)', fontsize=12)
    plt.ylabel('X (mm)', fontsize=12)
    plt.tight_layout()
    plt.show()

 
 
 
import numpy as np
import matplotlib.pyplot as plt

 
import jax.numpy as jnp
 
def plot_xy_heatmap(
    A_mean, rect_height, rect_width, x_range, y_range,
    grid_size, major_tick_interval, truth_type='point', save_path=None,cross_coords=None,cross_ends=None):
    # Rectangle
    thickness = 20
  
    z_max_idx = jnp.argmax(jnp.mean(A_mean, axis=(0, 1)))
    A_slice = A_mean[:, :, z_max_idx]
       

 
    # Direction and perpendicular vectors
    direction = cross_ends[0:2] - cross_coords[0:2]
    length = np.linalg.norm(direction)
    unit_dir = direction / length
    perp = np.array([-unit_dir[1], unit_dir[0]])
    offset = (thickness / 2) * perp

    rectangle_coords = np.array([
        cross_coords[0:2] + offset,
        cross_coords[0:2] - offset,
        cross_ends[0:2] - offset,
        cross_ends[0:2] + offset
    ])


    # Extent
    Nx, Ny = A_slice.shape
    dx = (x_range[1] - x_range[0]) / Nx
    dy = (y_range[1] - y_range[0]) / Ny
    extent = [
        x_range[0] - dx / 2, x_range[1] + dx / 2,
        y_range[1] + dy / 2, y_range[0] - dy / 2
    ]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    c = ax.imshow(
        A_slice.T, cmap='viridis', aspect='equal',
        extent=extent, norm=LogNorm(vmin=1e4,vmax=1e7)
    )
    cb = plt.colorbar(c, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Activity (Bq)', fontsize=12)
   
    if truth_type in ('cylinder', 'both'):

        polygon = Polygon(rectangle_coords, closed=True, edgecolor='black', fill=False, linewidth=1.5)
        ax.add_patch(polygon)


    if truth_type in ('point', 'both'):
        ax.plot(cross_coords[0], cross_coords[1], marker='x', color='black', markersize=10, markeredgewidth=2)
    if truth_type=='homo':
        pass

    # Ticks and grid
    def symmetric_ticks(axis_min, axis_max, interval):
        max_abs = max(abs(axis_min), abs(axis_max))
        tick_range = np.arange(-max_abs, max_abs + interval, interval)
        return tick_range[(tick_range >= axis_min) & (tick_range <= axis_max)]

    major_ticks_x = symmetric_ticks(*sorted(x_range), major_tick_interval)
    minor_ticks_x = symmetric_ticks(*sorted(x_range), grid_size)
    major_ticks_y = symmetric_ticks(*sorted(y_range), major_tick_interval)
    minor_ticks_y = symmetric_ticks(*sorted(y_range), grid_size)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(True, color='gray', linestyle='--', linewidth=0.4, which='both')
    ax.invert_yaxis()

    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_title(f'XY plane at Z index {z_max_idx}', fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=True)
    else:
        plt.show()

 
    
# Define the dimensions for the circle and rectangle    # Add grid

circle_radius = 287  # mm
rect_height = 885  # mm
rect_width = 287*2  # mm
grid_size = 19.2  # mm
x_range = (-288, 288)  # mm for X axis
y_range = (-451.2, 451.2)  # mm for Y axis

# Example usage:
 
if truth == 'rod' or isotope=='Co60':
    cross_coords = np.array([-45, 80, -14]) #rod
    cross_ends = np.array([-10, -80, 150])  #rod 
    truth_type= 'cylinder'

elif truth == 'point':
    truth_type= 'point'
    if isotope=='Eu152':
        truth_type= 'point'
        cross_coords = np.array([96,201.6,-115.2])#Eu152 voxel  
        cross_ends=np.array([0,0,0])
    elif isotope=='Cs137':
        truth_type= 'point'
        cross_coords = np.array([153.6,201.6,153.6])
        cross_ends = np.array([0,0,0])
elif truth == 'point2':
    if isotope =='Cs137':   
            truth_type= 'point'
            cross_coords = np.array([96,201.6,-115.2])#Eu152 voxel  
            cross_ends=np.array([0,0,0])
    elif isotope=='Eu152':
            truth_type= 'point'
            cross_coords = np.array([153.6,201.6,153.6])
            cross_ends = np.array([0,0,0])
        

elif truth == 'homo':
    truth_type= 'homo'
    cross_coords = np.array([0,0,0])
    cross_ends = np.array([0,0,0])


        
major_tick_interval = 96  # Define the interval for major ticks (96 mm)
plot_xz_heatmap(A_mean, circle_radius, x_range, grid_size, major_tick_interval,truth_type=truth_type,cross_coords=cross_coords,cross_ends = cross_ends)
plot_xy_heatmap(A_mean, rect_height, rect_width, x_range, y_range, grid_size, major_tick_interval,truth_type=truth_type,cross_coords=cross_coords,cross_ends=cross_ends)
"""
#If desired, a 3D render of the activity distribution can be generated uing SimpleITK:
#import SimpleITK as sitk
   
sitk_image = sitk.GetImageFromArray(A_mean)

sitk_image.SetOrigin( mockup_drum.mesh.attrs['origin'])
sitk_image.SetSpacing( mockup_drum.mesh.attrs['spacing'])
sitk_image.SetDirection( mockup_drum.mesh.attrs['direction'])
"""
 
#sitk.WriteImage(sitk_image, f'Retrieved_activity_{isotope}_{truth}_{actual_value}.mhd')

print(f'Average estimated Activity:  {jnp.mean(jnp.sum(A_samples,axis=(1,2,3)) )/1e9} GBq')


"""
This is for checking posterior densities for different results
"""
"""
z_samples_1 = guide.sample_posterior(random.PRNGKey(10101010), sample_shape=(2000,),params=params)
np.save("z_samples_config_good.npy", z_samples_1)

from numpyro.handlers import block, trace, condition

def compute_log_pz(model, z_samples, w, eta, t, mask, obs, hide_sites=None):
    # inject latent values
    conditioned_model = condition(model, data=z_samples)
    # hide likelihood if needed
    blocked_model = block(conditioned_model, hide=hide_sites or [])
    # compute trace
    tr = trace(blocked_model).get_trace(w, eta, t, mask, obs=obs)
    
    # sum log probs of latent sites
    log_pz = sum(
        site["fn"].log_prob(site["value"]).sum()
        for site in tr.values() if site["type"] == "sample"
    )
    return log_pz


z_samples_1 = np.load("z_samples_config_good.npy", allow_pickle=True).item()
z_samples_2 = np.load("z_samples_config_bad.npy", allow_pickle=True).item()
log_pz_list_1 = []
log_pz_list_2 = []

for i in range(2000):
  
    single_1 = {k: v[i] for k, v in z_samples_1.items()}
    single_2 = {k: v[i] for k, v in z_samples_2.items()}
    

    log_pz_list_1.append(
        compute_log_pz(forward_model, single_1, w, eta, t, mask, obs=obs, hide_sites=["Y_obs"])
    )
    log_pz_list_2.append(
        compute_log_pz(forward_model, single_2, w, eta, t, mask, obs=obs, hide_sites=["Y_obs"])
    )

log_pz_list_1 = np.array(log_pz_list_1)
log_pz_list_2 = np.array(log_pz_list_2)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.hist(log_pz_list_1, bins=50, alpha=0.5, label="Fixed voxel")
plt.hist(log_pz_list_2, bins=50, alpha=0.5, label="Failing model")
plt.xlabel("log p(z)")
plt.ylabel("Frequency")
plt.legend()
plt.show()

#summary
print("Config 1 mean log p(z):", log_pz_list_1.mean())
print("Config 2 mean log p(z):", log_pz_list_2.mean())
"""