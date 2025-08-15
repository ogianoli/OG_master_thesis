""" Compilation of functions that are beyond the scope of SRM
    and do not specifically fit in the calc_functions or plot_functions
"""

import os
import copy
import time
#--- extra non-standard libraries---#
import pandas as pd
import numpy as np
from scipy.stats import qmc
from scipy.interpolate import interp1d
import psutil # for memory checks
from warnings import warn
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from tqdm import tqdm
#----------------------------------#
import srm_preliminary_design.src.calc_functions as calcfunc
import srm_preliminary_design.src.plot_functions as plotfunc
#-----------------------------------------------------------#
def generate_sobol_samples(optimization_variables, nb_sobol_samples):
    """
    Generates Sobol sequence samples scaled to the provided optimization variable ranges.

    Parameters:
    - optimization_variables (list): List of dictionaries with 'name' and 'bounds' keys.
    - n_pairs (int): Number of Sobol-generated samples.

    Returns:
    - list of tuples: Scaled Sobol pairs matching the variable bounds.
    """
    # Determine the number of dimensions from the optimization variables
    dimensions = len(optimization_variables)

    # Initialize Sobol sequence generator
    sobol_engine = qmc.Sobol(d=dimensions, scramble=True)
    sobol_samples = sobol_engine.random(nb_sobol_samples)

    # Scale Sobol samples to the ranges defined in the optimization variables
    scaled_samples = []
    for sample in sobol_samples:
        scaled_pair = tuple(
            round(var["bounds"][0] + sample[i] * (var["bounds"][1] - var["bounds"][0]), 4)
            for i, var in enumerate(optimization_variables)
        )
        scaled_samples.append(scaled_pair)

    return scaled_samples

def check_samples_within_bounds(samples, optimization_variables):
    """
    Checks if generated samples are within the bounds of the optimization variables.

    Parameters:
    - samples (list of tuples): Generated Sobol samples.
    - optimization_variables (list): List of dictionaries with 'name' and 'bounds'.

    Returns:
    - bool: True if all samples are within bounds, False otherwise.
    """
    for sample in samples:
        for i, value in enumerate(sample):
            lower, upper = optimization_variables[i]["bounds"]
            if not (lower <= value <= upper):
                return False
    return True

def get_next_results_folder(base_name="results_", path=".", max_iter=100):
    for index in range(max_iter):
        folder_name = f"{base_name}{index:03d}"
        folder_path = os.path.join(path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created new folder {folder_path}")
            return folder_path
    raise RuntimeError(f"Could not create a new results folder after {max_iter} attempts.")

def make_column_integer(samples, column_index):
    """
    Converts a specific column in the samples to integers.

    Parameters:
    - samples (list of tuples): Generated Sobol samples.
    - column_index (int): Index of the column to convert.

    Returns:
    - list of tuples: Samples with the specified column converted to integers.
    """
    return [tuple(int(value) if i == column_index else value for i, value in enumerate(sample)) 
            for sample in samples]

def check_for_duplicates(samples):
    """
    Removes duplicate samples.

    Parameters:
    - samples (list of tuples): Generated samples.

    Returns:
    - list of tuples: Samples with duplicates removed.
    """
    return list(set(samples))

def calculate_r2(target_values, predicted_values):
    """
    Calculates the coefficient of determination (R²) between target and predicted values.

    Parameters:
    -----------
    target_values : array-like
        Actual (true) values (list, numpy array, or pandas Series).
    predicted_values : array-like
        Predicted values (list, numpy array, or pandas Series).

    Returns:
    --------
    float
        The R² value, ranging from -∞ to 1.0.
    """
    # Convert inputs to numpy arrays for consistent operations
    target_values = np.asarray(target_values)
    predicted_values = np.asarray(predicted_values)

    # Ensure the shapes match
    if target_values.shape != predicted_values.shape:
        raise ValueError("Target and predicted values must have the same shape.")
    
    # Compute sum of squared residuals
    ss_res = np.sum((target_values - predicted_values) ** 2)
    # Compute total sum of squares
    ss_tot = np.sum((target_values - np.mean(target_values)) ** 2)
    
    # Calculate R²
    r2 = 1 - (ss_res / ss_tot)
    return r2

def create_one_dim_interpolator(xvals, yvals, method = 'piecewise_linear_extrapolate'):
    assert method in ['piecewise_linear_clip', 'piecewise_linear_extrapolate']

    if method == 'piecewise_linear_extrapolate':
        fill_param = 'extrapolate'
    if method == 'piecewise_linear_clip':
        fill_param = (None, 0)
    
    return interp1d(xvals, yvals, bounds_error = False, fill_value=fill_param)

def compute_mean_squared_error(vals_in, vals_tgt):
    return np.mean((vals_tgt - vals_in) ** 2)

def find_index_tgt_val(value, array):
    return np.where(array == value)[0]

def find_entries_from_criteria(array, nb_entries, criterion):
    """
    Return the indices and values of the `nb_entries` best entries in `array`,
    based on the provided `criterion` (e.g., np.max, np.min).

    Parameters:
    - array: 1D numpy array
    - nb_entries: int, number of top or bottom entries to find
    - criterion: function, e.g., np.max, np.min, or custom function

    Returns:
    - indices: np.ndarray of indices of the selected entries
    - values: np.ndarray of the corresponding values
    """
    array = np.asarray(array)
    if criterion == np.max:
        indices = np.argpartition(-array, nb_entries)[:nb_entries]
        sorted_indices = indices[np.argsort(-array[indices])]
    elif criterion == np.min:
        indices = np.argpartition(array, nb_entries)[:nb_entries]
        sorted_indices = indices[np.argsort(array[indices])]
    else:
        # For a custom function (e.g., absolute value), sort all and take best
        values = criterion(array)
        indices = np.argsort(values)[:nb_entries]
        sorted_indices = indices

    return sorted_indices, array[sorted_indices]


#------------------------------------------------------------#
#------------------------------------------------------------#
#------------------------------------------------------------#
#------------------------------------------------------------#
#---------------Memory related functions--------------#
# Function to monitor memory usage
def monitor_memory_usage():
    # Get total RAM
    total_memory = psutil.virtual_memory().total
    used_memory = psutil.virtual_memory().used
    print(f"Total RAM: {total_memory / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {used_memory / (1024 ** 3):.2f} GB")

def compute_list_memory_mb(array_list):
    """
    Compute the total memory usage of a list of NumPy arrays in megabytes (MB).

    Parameters:
        array_list (list): List containing NumPy arrays.

    Returns:
        float: Total memory usage in MB.
    """
    total_bytes = sum(arr.nbytes for arr in array_list if isinstance(arr, np.ndarray))
    print(f"Total memory usage of times_list: {total_bytes / 1024 / 1024:.2f} MB")
    
    return total_bytes / 1024 / 1024  # Convert bytes to megabytes

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 ** 2  # Resident Set Size in MB
    print(f"[{note}] Memory usage: {mem_mb:.2f} MB")






#-------End memory related functions------------------------#
#------------------------------------------------------------#
#------------------------------------------------------------#
#------------------------------------------------------------#
#------------------------------------------------------------#


def extract_index_colvals(ind_list, df_in, col_names = None):
    """ Based on a list of indices, returns those rows from 
        a pandas df
    """
    return copy.deepcopy(df_in.iloc[ind_list])

def compute_fmax(par_cols_names, df_in, BSA_array, OD_grain, L_prop, BR_comb):
    """ NOTE: order of par_cols_names is super important! and so dangerous actually
        should be: NoP, Ri, f, e, web  for BSA_star_shape_array()
    """
    tmp_list = [
        calcfunc.BSA_star_shape_array(
            BSA_array, *row, OD_grain, L_prop, BR_comb, f_max_return=True
        )
        for row in df_in[par_cols_names].itertuples(index=False, name=None)
    ]
    return tmp_list

def check_f_max_error(opt_variables, fmax, exp_sol = True):

    _, _, fill_radius, _ = opt_variables

    if exp_sol is True:
        
        r = fill_radius / fmax
        ro = 0.9
        alpha = 50
        p = 4
        if r <= ro: sol = 1
        else: sol = np.exp(alpha * (r ** p - ro ** p))

        return sol

    return True if fill_radius > fmax else False

def evaluate_fmax_error(param_col_names, df_in,
                               BSA_array, OD_grain, L_prop, BR_comb):
    """ method applied on complete input dataframe and useful for 
        postprocessing purposes. check_f_max_error is for single call
    """
    fmax_list = compute_fmax(param_col_names, df_in,
                               BSA_array, OD_grain, L_prop, BR_comb)

    fmax_err = [True if fill_rad > fmax else False 
                for fill_rad, fmax in zip(df_in['fill_radius'], fmax_list)]

    return fmax_err

def check_fill_radius_error(opt_variables, shape_int, ODgrain, web, exp_sol = True):

    if shape_int == 'star':
        NoB, Ri, fill_radius, _ = opt_variables
    elif shape_int == 'finocyl':
        NoB, _, fill_radius, _ = opt_variables
        Ri = ODgrain / 2 - web
    elif shape_int == 'truncated star':
        NoB, Ri, fill_radius = opt_variables
        
    tmp_arr = fill_radius / np.sin(np.pi / NoB)
    tmp_arr -= Ri

    if exp_sol is True:
        
        r = fill_radius / np.sin(np.pi / NoB) / Ri
        ro = 0.9
        alpha = 50
        p = 4
        if r <= ro: sol = 1
        else: sol = np.exp(alpha * (r ** p - ro ** p))

        return sol

    return True if tmp_arr > 0 else False

def check_branch_radius_error(df_in):
    tmp_arr = df_in['fill_radius'] / np.sin(np.pi / df_in['number_branches'])
    tmp_arr -= df_in['int_radius']

    return [True if val < 0 else False for val in tmp_arr]


def check_inner_radius_error(opt_variables, od_grain, web, exp_sol = True):
    
    NoB, Ri, fill_radius, _ = opt_variables

    tmp_arr = (od_grain / 2 - Ri) 
    tmp_arr -= (web + fill_radius)

    if exp_sol is True:

        r = (web + fill_radius) / (od_grain / 2 - Ri)
        ro = 0.9
        alpha = 50
        p = 4
        if r <= ro: sol = 1
        else: sol = np.exp(alpha * (r ** p - ro ** p))

        return sol

    return True if tmp_arr < 0 else False 

def check_diameter_radius_error(df_in, od_grain):
    tmp_arr = (od_grain / 2 - df_in['int_radius']) 
    tmp_arr -= (df_in['web_burned'] +df_in['fill_radius'])

    warn('This method is deprecated. Use check_inner_radius_error() instead.', DeprecationWarning, stacklevel=2)

    return [True if val < 0 else False for val in tmp_arr]

def check_web_error(opt_variables, web_max, exp_sol = True):

    _, web_star, _,  _ = opt_variables

    if exp_sol is True:
        
        r = web_star / web_max
        ro = 0.9
        alpha = 50
        p = 4
        if r <= ro: sol = 1
        else: sol = np.exp(alpha * (r ** p - ro ** p))

        return sol

    return True if web_star > web_max else False

def check_theta_error(od_grain, web, opt_variables, exp_sol = True):

    NoP, Ri, _, e = opt_variables

    # Calculate theta and Rp using the geometry function
    theta, Rp, _, _, _, _ = calcfunc.star_shape_derived_param(od_grain, web, opt_variables)

    if theta > 0:
        r = theta / np.pi
        error = theta > np.pi
    else:
        angle = np.pi * e / NoP
        r = (Rp * np.sin(angle)) / (Ri * np.tan(angle))
        error = (Rp * np.sin(angle)) > (Ri * np.tan(angle))

    if not exp_sol:
        return error

    # If exp_sol is True, return smoothed penalty instead of boolean
    ro = 0.9
    alpha = 50
    p = 4

    if r <= ro:
        return 1.0
    else:
        return np.exp(alpha * (r ** p - ro ** p))

def get_exp_penalty_from_geom_checks(opt_variables, od_grain, web, shape_int):

    if shape_int == 'star':
        fmax = calcfunc.star_shape_derived_param(od_grain, web, opt_variables, fmax_return = True)
        fmax_err = check_f_max_error(opt_variables, fmax, exp_sol = True)
        branch_radius_err = check_fill_radius_error(opt_variables, shape_int, od_grain, web, exp_sol = True)
        diam_rad_error = check_inner_radius_error(opt_variables, od_grain, web, exp_sol = True)
        theta_error = check_theta_error(od_grain, web, opt_variables, exp_sol = True)
        total_error = fmax_err * branch_radius_err * diam_rad_error * theta_error

    elif shape_int == 'finocyl':
        branch_radius_err = check_fill_radius_error(opt_variables, shape_int, od_grain, web, exp_sol = True)
        web_error = check_web_error(opt_variables, web, exp_sol = True)
        total_error = branch_radius_err * web_error

    elif shape_int == 'truncated star':
        branch_radius_err = check_fill_radius_error(opt_variables, shape_int, od_grain, web, exp_sol = True)
        total_error = branch_radius_err

    return total_error

def perform_geometry_checks(opt_variables, od_grain, web):
    """ Various validity checks deviced by Ambara.
        If not all False, the geometry is not valid
        NOTE: statefull function!
    """

    fmax = calcfunc.star_shape_derived_param(od_grain, web, opt_variables, fmax_return = True)
    fmax_err = check_f_max_error(opt_variables, fmax, exp_sol = False)
    branch_radius_err = check_fill_radius_error(opt_variables, exp_sol = False)
    diam_rad_error = check_inner_radius_error(opt_variables, od_grain, web, exp_sol = False)

    if fmax_err or branch_radius_err or diam_rad_error == False: sol = False
    else: sol = True

    return sol

def perform_validity_checks(df_in, param_col_names,
                               BSA_array, OD_grain, L_prop, BR_comb):
    """ Various validity checks deviced by Ambara.
        If not all False, the geometry is not valid
        NOTE: statefull function!
        These are custom for postprocessing in the form of input dataframes
    """

    df_in['fmax_err'] = evaluate_fmax_error(param_col_names, df_in,
                               BSA_array, OD_grain, L_prop, BR_comb)

    df_in['branch_radius_err'] = check_branch_radius_error(df_in)
    df_in['diameter_radius_err'] = check_diameter_radius_error(df_in, OD_grain)
    df_in['valid_geom'] = [True if np.unique(row).size == 1 and row[0] == False else False 
                            for row in df_in[['fmax_err', 'branch_radius_err',
                                'diameter_radius_err']].itertuples(index=False, name=None)
    ]

def find_sliver_start_index(time_calc, bsa_calc, method = 'ambara'):
    assert method in ['ambara','jj']
    # Can definitely be replaced by np.diff
    if method == 'ambara':
        dbsa_dt = np.ones(len(time_calc)-1)
        for ii in range(2,len(time_calc)-1):
            # Should be divided by time_calc instead!??
            dbsa_dt[ii] = (bsa_calc[ii] - bsa_calc[ii-1]) / (bsa_calc[ii] - bsa_calc[ii-1])

        idx_end_arr = np.where(dbsa_dt == np.min(dbsa_dt))[0] - 1
    if method == 'jj':
        dbsa_dt = np.diff(bsa_calc) / np.diff(time_calc)
        idx_end_arr = np.where(dbsa_dt == np.min(dbsa_dt))[0]

    if idx_end_arr.size == 1:
        idx_end = int(idx_end_arr[0])
    else:
        idx_end = len(bsa_calc)
    return idx_end


def compute_burn_time_error(actual_vals, tgt_val, method = 'relative_max'):
    """ Method is suitable for postprocessing purposes
    """
    assert method in ['absolute', 'relative_max']
    if method == 'absolute':
        return abs(tgt_val - np.array(actual_vals))
    
    if method == 'relative_max':
        return [abs(tgt_val - act_val)/max(tgt_val, act_val) for act_val in actual_vals]


def compute_scalar_error(actual_vals, tgt_val, method = 'relative_max'):
    assert method in ['absolute', 'relative_max', 'diff', 'diffinvert']
    if method == 'absolute':
        return abs(tgt_val - actual_vals)

    if method == 'diff':
        return tgt_val - actual_vals

    if method == 'diffinvert':
        return actual_vals - tgt_val 

    if method == 'relative_max':
        return abs(tgt_val - actual_vals)/max(tgt_val, actual_vals)

def compute_vector_shape_error(actual_vals_arr, actual_time_arr, tgt_vals_arr, target_time_arr,
                        method = 'mse'):
    """ Shape error applied to the section of the signal that falls within the
        desired burn time.
    """
    assert method in ['mse', 'R2 scikit-learn', 'R2', 'mean scaled', 'strong error']
    pred_interp = create_one_dim_interpolator(actual_time_arr, actual_vals_arr)
    # fill_value=-- consider this instead!
    pred_y_interp = pred_interp(target_time_arr)
    if method == 'mse':
        shape_error = compute_mean_squared_error(pred_y_interp,tgt_vals_arr)
        return shape_error
    if method == 'R2 scikit-learn':  # Scikit-learn R2 score
        return 1-r2_score(tgt_vals_arr, pred_y_interp)
    if method == 'R2':  # In-house R2_score
        return calculate_r2(pred_y_interp,tgt_vals_arr) # CHECK THIS!!
    if method == 'mean scaled':
        shape_error = np.mean((tgt_vals_arr - pred_y_interp) ** 2)
        return shape_error / (np.max(tgt_vals_arr) ** 2 + 1e-8)
    if method == 'strong error':
        
        if not np.all(np.isfinite(actual_vals_arr)):
            return 1e6
        
        dt = target_time_arr[1] - target_time_arr[0]

        # Component 1: Absolute Overshoot Area
        target_value = tgt_vals_arr[1]
        over = np.maximum(pred_y_interp - target_value, 0)
        under = np.maximum(target_value - pred_y_interp, 0)

        # Penalize overshoot more heavily
        overshoot_area = np.sum(over) * dt
        undershoot_area = np.sum(under) * dt

        # Component 2: Non-Flatness (Curvature)
        d2 = np.diff(pred_y_interp, n=2)
        flatness_penalty = np.sum(np.abs(d2)) / len(d2)

        # Optional: max deviation
        max_dev = np.max(np.abs(pred_y_interp - tgt_vals_arr))

        components = [overshoot_area, undershoot_area, flatness_penalty, max_dev]
        if not all(np.isfinite(x) for x in components):
            return 1e6

        # Combine
        loss = (
            10.0 * overshoot_area +     # critical penalty
            5.0 * undershoot_area +     # strong but less harsh
            1.0 * flatness_penalty +    # punishes spikes
            2.0 * max_dev               # prevent extreme peaks
        ) /max(tgt_vals_arr) # normalize by max target value if needed
        return loss

def sample_data_within_xrange(xvals_arr, yvals_arr, target_xvals_arr):
    pred_interp = create_one_dim_interpolator(xvals_arr, yvals_arr)
    pred_y_interp = pred_interp(target_xvals_arr)
    return pred_y_interp

def compute_within_band(yvals, ytarget, tol = 0.05):
    """ Calculates the nb of entries of an array that are
        within an error band of a target value.
    """
    within_band = np.abs(yvals - ytarget) < tol
    fraction_within_band = np.sum(within_band) / len(yvals)
    return fraction_within_band

def compute_metrics_from_df(df_in, runs_select, bsa_arr, time_arr):
    """ A collection of metrics to compute on an input df
    """
    valid_runs = [ind for ind in runs_select if ind in df_in.index]
    # -2 index in case smth weird at last index, had some NaN issues
    burn_times = [df_in['Time'][ind].iloc[-2] for ind in valid_runs] 
    # do -1 index in case we have suddenly a weird negative pressure which is also a
    #   negative time
    shape_error_mse = [compute_shape_error(df_in['Pressure'][ind][:-1], df_in['Time'][ind][:-1],
                bsa_arr, time_arr,
                method = 'mse')
                    for ind in valid_runs]

    shape_error_r2 = [compute_shape_error(df_in['Pressure'][ind][:-1], df_in['Time'][ind][:-1],
                bsa_arr, time_arr,
                method='R2')
                for ind in valid_runs]

    shape_error_mse_mean_scaled = [compute_shape_error(df_in['Pressure'][ind][:-1],
                                                                df_in['Time'][ind][:-1],
                            bsa_arr, time_arr, method = 'mean scaled')
                            for ind in valid_runs]

    return burn_times, shape_error_mse, shape_error_r2, shape_error_mse_mean_scaled



def combined_error_metric(frac_in_band, time_error, penalty_scale=10.0):
    """
    Compute a combined error metric based on time error and deviation from target value.
    Time error is used as a first main check, followed by the fraction in band

    Args:
        frac_in_band (list): of float of  pre-computed nb of signal entries that
                            are within a target error band 
        time_error (list): of float time difference from expected event (e.g., in seconds)
        penalty_scale (float): scalar factor to amplify the time error penalty

    Returns:
        (float): an error scalar metric
    """
    if time_error < 0:
        # Pre-event: penalize negative time error more heavily
        return abs(time_error) * penalty_scale
    else:
        # Post-event: emphasize how close y stays to target
        deviation_penalty = 1.0 - frac_in_band  # 0 is perfect, 1 is worst
        return deviation_penalty



def compute_shape_error(actual_vals_arr, actual_time_arr, tgt_vals_arr, target_time_arr,
                        method = 'mse'):
    """ Shape error applied to the section of the signal that falls within the
        desired burn time.  
    """

    warn('This method is deprecated. Use compute_vector_shape_error() instead.',
         DeprecationWarning, stacklevel=2)

    assert method in ['mse', 'R2', 'mean scaled']
    pred_interp = create_one_dim_interpolator(actual_time_arr, actual_vals_arr)
    # fill_value=-- consider this instead!
    # NOTE: should consider min between target time arr and desired time array
    #       otherwise we're in extrapolation mode
    pred_y_interp = pred_interp(target_time_arr)
    if method == 'mse':
        shape_error = compute_mean_squared_error(pred_y_interp,tgt_vals_arr)
        return shape_error
    if method == 'R2':
        return calculate_r2(pred_y_interp,tgt_vals_arr)
    if method == 'mean scaled':
        shape_error = compute_mean_squared_error(pred_y_interp,tgt_vals_arr)
        # shape_error = np.mean((tgt_vals_arr - pred_y_interp) ** 2)
        return shape_error / (np.max(tgt_vals_arr) ** 2 + 1e-12)


def compute_vlf_error(vlf_arr):
    return (1 - vlf_arr) ** 2

def select_subspace(df, columns, num_points=20, seed=None):
    """
    Selects a random small subspace of design parameters in a multi-dimensional sense.

    Parameters:
        df (pd.DataFrame): DataFrame containing the design parameters.
        columns (list): List of column names representing the design parameters.
        num_points (int): Number of points to select in the subspace (default is 20).
        seed (int, optional): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Subset of the DataFrame with selected points.
    """

    # Normalize the specified columns using min-max scaling
    df_normalized = df.copy()
    df_normalized[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())

    # Randomly select a point of interest
    random_point = df_normalized[columns].sample(n=1, random_state = seed).values[0]

    # Calculate Euclidean distance to the random point for each row
    distances = np.linalg.norm(df_normalized[columns].values - random_point, axis=1)

    # Sort by distance and select the closest num_points
    nearest_indices = np.argsort(distances)[:num_points]

    return df.iloc[nearest_indices].copy()


def dataframe_analytics(df: pd.DataFrame) -> dict:
    """
    Calculate min, max, mean, and radian (converted from degrees)
    for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with numeric values.

    Returns:
    dict: Dictionary with column names as keys and calculated values as values.
    """
    analytics = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            analytics[column] = {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean(),
                'median': df[column].median()
            }

    return analytics

def subset_check_function(df_in, input_params, branches_list,
                          random_seeds, df_pressure, time_arr, BSA_array,
                          output_folder = '.', save_shape_image = False,
                          OD_grain = None, save_subset_dfs = False,
                            xvals_plot= 'Time', yvals_plot = 'Pressure'):
    """
    Create and analyze multiple data subsets based on a list of random seeds for multiple branches.

    Parameters:
    - df_in (pd.DataFrame): Dataframe containing the full valid data.
    - input_params (list): List of input parameter names (cols of df to consider).
    - branches_list (list): List of branch values to process.
    - random_seeds (list): List of random seeds for each subset.
    - df_pressure (pd.DataFrame): Pressure dataframe for curve plotting.
    - time_arr (np.array): Time array for target values.
    - BSA_array (np.array): BSA array for target values.
    - output_folder (str): Directory to save plots.
    """

    for branches in branches_list:
        print(f"Processing {branches} branches...")
        subset_df = df_in[df_in['number_branches'] == branches]
        if subset_df.empty:
            print(f"No data available for {branches} branches. Skipping.")
            continue

        subset_list = []
        # Generate subsets and analytics
        analytics_results = []
        for _, seed in enumerate(random_seeds):
            subset = select_subspace(subset_df, input_params, seed=seed)
            subset_list.append(subset)

            tmp_subset = dataframe_analytics(subset[input_params])
            analytics_results.append(tmp_subset)

        branch_subfolder = f"subset_check_nbranch{branches}"
        # Complete original range
        tmp_original = dataframe_analytics(df_in[input_params])
        # Plot analytics
        plotfunc.plot_analytics(analytics_results, tmp_original,
                                output_folder, branch_subfolder)

        # Plot subset curves
        plotfunc.plot_subset_curves(subset_list, df_pressure,
                                    time_arr, BSA_array, output_folder,
                                    branch_subfolder,
                                    xvals= xvals_plot, yvals = yvals_plot)
        if save_shape_image:
            save_shape_image_for_df_lists(subset_list, input_params, branches, OD_grain,
                                  output_folder, branch_subfolder)
            # for seed_nb, seed_subset_nb in enumerate(subset_list):
            #     for ind, row_vals in seed_subset_nb[list(input_params)].iterrows():
            #         plotfunc.plot_star_geometry(
            #             branches,
            #             row_vals['int_radius'],
            #             OD_grain / 2,
            #             row_vals['web_burned'],
            #             row_vals['fill_radius'],
            #             row_vals['ang_coeff'],
            #             savefolder= os.path.join(output_folder, branch_subfolder,
            #                                     f"subset{seed_nb+1}"),
            #             savename=f"shape{ind}.png",
            #             show_plot=False
            #         )
        if save_subset_dfs:
            for seed_nb, seed_subset_df in enumerate(subset_list):
                seed_subset_df.to_pickle(os.path.join(output_folder, branch_subfolder,
                                        f"subset{seed_nb+1}", 'subset_df.pkl'))


def save_shape_image_for_df_lists(subset_list, input_params, branches, OD_grain,
                                  output_folder, branch_subfolder, userfolder = None):
    for seed_nb, seed_subset_nb in enumerate(subset_list):
        savefolder= os.path.join(output_folder, branch_subfolder,
                                        f"subset{seed_nb+1}")
        if userfolder:
            savefolder= os.path.join(output_folder, branch_subfolder,
                                        userfolder)
        os.makedirs(savefolder, exist_ok= True)
        for ind, row_vals in seed_subset_nb[list(input_params)].iterrows():
            plotfunc.plot_star_geometry(
                branches,
                row_vals['int_radius'],
                OD_grain / 2,
                row_vals['web_burned'],
                row_vals['fill_radius'],
                row_vals['ang_coeff'],
                savefolder=savefolder,
                savename=f"shape{ind}.png",
                show_plot=False
            )



def find_nans_in_df(df_in, keyword = 'Pressure'):
    """ Function goes through groups defined via 'Run' and find NaNs in the
        input keyword column

        Args:
            df_in (pandas df): input dataFrame
            keyword (str): column of Pandas Df where to find

        Returns:
            (Index list): with Run nbs where a NaN has been found
    """
    return df_in[keyword].isna().groupby(level='Run').any()


def find_nans_above_threshold(df_in, keyword='Pressure', max_allowed_nans=0):
    """
    Function checks per 'Run' group if the number of NaNs in the specified column
    exceeds `max_allowed_nans`.

    Args:
        df_in (pd.DataFrame): input DataFrame with 'Run' as index level.
        keyword (str): name of the column to check for NaNs.
        max_allowed_nans (int): maximum number of NaNs allowed per group.

    Returns:
        pd.Series: Boolean Series indexed by 'Run', True if NaNs exceed threshold.
    """
    nan_counts = df_in[keyword].isna().groupby(level='Run').sum()
    return nan_counts > max_allowed_nans
    # return nan_counts[nan_counts > max_allowed_nans].index


#----------Machine learning corner---------------------------#
#----------Machine learning corner---------------------------#

def extract_peak_features(xvals, yvals):
    peak_index = np.argmax(yvals)
    peak_value = yvals[peak_index]
    peak_time = xvals[peak_index]
    return peak_value, peak_time


def extract_mean_std(yvals):
    mean_value = np.mean(yvals)
    std_value = np.std(yvals)
    return mean_value, std_value


def extract_initial_slope(xvals, yvals):
    initial_slope = (yvals[1] - yvals[0]) / (xvals[1] - xvals[0])
    return initial_slope


def extract_range(yvals):
    value_range = np.max(yvals) - np.min(yvals)
    return value_range

def extract_value_at_xloc(xvals, yvals, xloc):
    if xloc in xvals:
        return yvals[np.where(xvals == xloc)[0][0]]
    else:
        return np.interp(xloc, xvals, yvals)
    
def get_curve_features(curves_df, x_col='xvals', y_col='yvals', index_range=None,
                       xloc = None):
    features = []
    indices = index_range if index_range is not None else curves_df.index

    for index in indices:
        xvals = curves_df[x_col][index]
        yvals = curves_df[y_col][index]

        peak_value, peak_time = extract_peak_features(xvals.values, yvals.values)
        mean_value, std_value = extract_mean_std(yvals)
        initial_slope = extract_initial_slope(xvals.values, yvals.values)
        value_range = extract_range(yvals)

        value_at_xloc = extract_value_at_xloc(xvals, yvals, xloc) if xloc is not None else np.nan

        features.append([peak_value, peak_time, mean_value, std_value, initial_slope, value_range, value_at_xloc])


    
    features_df = pd.DataFrame(features, columns=[
        'PeakValue', 'PeakTime', 'MeanValue', 'StdValue', 'InitialSlope', 'Range', 'ValueAtXloc'
    ], index=indices)

    if xloc is None:
        features_df = features_df.drop(columns=['ValueAtXloc'])

    return features_df


def classify_curve_shapes(features_df, n_clusters=2, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(features_df)
    return labels



#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# --------------- Parallelization  oriented functions ----------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
def make_compute_sliver_row_pandas(OD_grain, params_use):
    def compute_sliver_row_pandas(row):
        return calcfunc.compute_sliver_percentage(
            OD_grain,
            'star',
            row[params_use[-1]],
            row[params_use[:-1]].values
        )
    return compute_sliver_row_pandas


def make_compute_sliver_row_dict(OD_grain, params_use):
    def compute_sliver_row_dict(row):
        return calcfunc.compute_sliver_percentage(
            OD_grain,
            'star',
            row[params_use[-1]],
            [row[param] for param in params_use[:-1]]
        )
    return compute_sliver_row_dict

def run_parallel_joblib(df_in, params_use, input_function,
                        nprocs = 6, method = 'dict', show_progress=False,
                        batch_size = 500):
    compute_row = input_function # case when no partial function
    if params_use is not None: # additional argument, but should avoid this
        compute_row = input_function(params_use) # could also be already as partial argument
    if nprocs > 1:
        print("Starting parallel computation...")
    start = time.time()
    total = len(df_in)
    if method == 'dict':
        iterator = df_in.to_dict("records")
    else:
        iterator = list(df_in.iterrows())

    if show_progress:
        iterator = tqdm(iterator, total=total, desc="Processing", leave=True)

    if method == 'dict':
        results = Parallel(n_jobs= nprocs, batch_size = batch_size)(
            delayed(compute_row)(row_dict) for row_dict in iterator
        )
    else:
        results = Parallel(n_jobs= nprocs)(
            delayed(compute_row)(row) for _, row in iterator
        )
    end = time.time()
    print(f"Completed in {end - start:.2f} seconds.")
    return results




def make_perform_validity_row_dict(BSA_array, OD_grain, L_prop, BR_comb, param_col_names,
                                   L_prop_variable = None):
    def perform_validity_row_dict(row):
        L_use = L_prop
        if L_prop_variable is not None:
            L_use  = row[L_prop_variable]
        fmax_err = evaluate_fmax_error(param_col_names, pd.DataFrame([row]),
                                       BSA_array, OD_grain, L_use, BR_comb)[0]
        branch_err = check_branch_radius_error(pd.DataFrame([row]))[0]
        diam_err = check_diameter_radius_error(pd.DataFrame([row]), OD_grain)[0]

        web = row[param_col_names[-1]]
        opt_variables = [row[param] for param in param_col_names[:-1]]
        theta_err = check_theta_error(OD_grain, web, opt_variables, exp_sol=False)

        valid = not any([fmax_err, branch_err, diam_err, theta_err])

        return {
            'fmax_err': fmax_err,
            'branch_radius_err': branch_err,
            'diameter_radius_err': diam_err,
            'theta_err': theta_err,
            'valid_geom': valid
        }
    return perform_validity_row_dict



#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# --------------- END Parallelization  oriented functions ------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
