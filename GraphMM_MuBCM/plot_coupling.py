import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set English fonts for Matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans' # Or Arial, Helvetica, etc. if available

# --- Configuration ---
coupling_file = './results/coupling_graph/coupling_graph_param_ISK_ICN_600s_16.csv'
icn_surrogate_file = './results/600s/surrogate_ICN_nohub_600s/surrogate_ICN_cell_16.csv'
isk_surrogate_file = './results/600s/surrogate_ISK_600s_nohub/surrogate_ISK_600s_cell_16.csv'


# Time steps from config.yaml (or deduced from data generation)
dt_isk = 1.0e-4  # ISK time step
dt_icn = 3.0e-2  # ICN time step

# Column indices (0-based) - assuming header row exists
# Coupling file: mean=col 0
coupling_mean_col = 0

# ISK file: Ca_ic.ISK is state index 2 ('Ca_ic.ISK') -> mean is col 2*2 = 4
isk_ca_ic_mean_col = 4

# ICN file: Ca.ICN is state index 3 ('Ca.ICN') -> mean is col 3*2 = 6
icn_ca_mean_col = 6

# --- Load Data ---
try:
    # Read coupling data
    coupling_data = np.genfromtxt(coupling_file, delimiter=',', skip_header=1)
    coupling_mean = coupling_data[:, coupling_mean_col]
    print(f"Read {len(coupling_mean)} data points from coupling file.")

    # Read ISK surrogate data
    isk_data = np.genfromtxt(isk_surrogate_file, delimiter=',', skip_header=1)
    # Handle potential different row counts by slicing to match coupling data length if necessary
    min_len_isk_coupling = min(len(isk_data), len(coupling_mean))
    isk_ca_ic = isk_data[:min_len_isk_coupling, isk_ca_ic_mean_col]
    coupling_mean = coupling_mean[:min_len_isk_coupling] # Adjust coupling data if ISK is shorter
    print(f"Read {len(isk_ca_ic)} data points from ISK file (adjusted to match coupling length if needed).")


    # Read ICN surrogate data
    icn_data = np.genfromtxt(icn_surrogate_file, delimiter=',', skip_header=1)
    icn_ca = icn_data[:, icn_ca_mean_col]
    print(f"Read {len(icn_ca)} data points from ICN file.")

except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit()
except IndexError as e:
    print(f"Error: Column index out of bounds. Check column definitions and file structure. Details: {e}")
    exit()
except Exception as e:
    print(f"An error occurred while reading the files: {e}")
    exit()

# --- Create Time Axes ---
# Coupling and ISK share the same time axis (faster time step)
time_coupling_isk = np.arange(len(coupling_mean)) * dt_isk
# ICN time axis
time_icn = np.arange(len(icn_ca)) * dt_icn

# --- Plotting ---
fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=False) # Adjust figsize if needed

# Plot 1: ICN Surrogate Ca.ICN
ax[0].plot(time_icn, icn_ca, label='Ca.ICN Mean')
ax[0].set_title('ICN Surrogate Model State (Ca.ICN)')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Ca.ICN Mean')
ax[0].grid(True)
ax[0].legend()

# Plot 2: ISK Surrogate Ca_ic.ISK
ax[1].plot(time_coupling_isk, isk_ca_ic, label='Ca_ic.ISK Mean', color='orange')
ax[1].set_title('ISK Surrogate Model State (Ca_ic.ISK)')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Ca_ic.ISK Mean')
ax[1].grid(True)
ax[1].legend()

# Plot 3: Coupling Variable Mean Only
ax[2].plot(time_coupling_isk, coupling_mean, label='Coupling Variable Mean', color='black')
ax[2].set_title('ICN-ISK Coupling Variable Mean')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Coupling Variable Mean')
ax[2].grid(True)
ax[2].legend()


fig.tight_layout()
plt.show()