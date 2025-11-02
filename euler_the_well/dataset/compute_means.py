from euler import EulerPeriodicDataset
import numpy as np

def compute_mean_fields(ds):
    """
    Computes the mean fields values over the entire dataset (all simulations, all timesteps).
    """
    mean_density = []
    mean_pressure = []
    mean_energy = []
    mean_momentum = []  # magnitude

    for i in range(ds.n_sims):
        # load all time possible steps for simulation i
        arrs = ds._load_time_window(sim_idx=i, t_idx=0)
        mean_density.append(arrs["density"].mean())
        mean_pressure.append(arrs["pressure"].mean())
        mean_energy.append(arrs["energy"].mean())

        # for momentum, compute magnitude first
        arrs["momentum"] = np.sqrt((arrs["momentum"]**2).sum(axis=-1))
        mean_momentum.append(arrs["momentum"].mean())

    # now compute global means
    mean_density = sum(mean_density) / ds.n_sims
    mean_pressure = sum(mean_pressure) / ds.n_sims
    mean_energy = sum(mean_energy) / ds.n_sims
    mean_momentum = sum(mean_momentum) / ds.n_sims

    return mean_density, mean_pressure, mean_energy, mean_momentum

if __name__ == "__main__":
    sims_paths = ["euler_multi_quadrants_periodicBC_gamma_1.13_C3H8_16.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.404_H2_100_Dry_air_-15.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.22_C2H6_15.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.453_H2_-76.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.33_H2O_20.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.4_Dry_air_20.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.365_Dry_air_1000.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.597_H2_-181.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.3_CO2_20.hdf5",
        "euler_multi_quadrants_periodicBC_gamma_1.76_Ar_-180.hdf5"]

    base_paths = ["/work/imos/datasets/euler_multi_quadrants_periodicBC/data/train/",
            "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/valid/",
            "/work/imos/datasets/euler_multi_quadrants_periodicBC/data/test/"]

    stats_path = "/work/imos/datasets/euler_multi_quadrants_periodicBC/stats.yaml"

    for sim_file in sims_paths:
        for base_path in base_paths:
            full_path = base_path + sim_file
            ds = EulerPeriodicDataset(full_path, stats_path=stats_path, time_window=1, patch_size=None, normalize=True)
            means = compute_mean_fields(ds)
            print(f"Dataset: {full_path}")
            print("Means (density, pressure, energy, momentum):", means)
            print(f"Gamma: {ds._static_cache['gamma']}")
            print("-----")
            