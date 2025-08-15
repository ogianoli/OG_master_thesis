import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os
import math

def plot_3d_objective_contour(f1_values, f2_values, f3_values,
                              title="",
                              label_one = None, label_two = None, label_three= None,
                              interp_method = 'cubic',
                              output_folder = '.',
                              savename = 'contour_plot.png',
                              show_plot = True,
                              xrange = None,
                              yrange = None):
    """
    Creates a 2D contour plot using three objective functions.

    Parameters:
        f1_values, f2_values, f3_values: np.ndarray
            1D arrays of the three objective functions (same length).
        title: str
            Title of the plot.
        label_one, label_two, label_three: str (optional)
            Labels for the x, y axes, and the color bar.
        interp_method: str
            Interpolation method for griddata ('linear', 'cubic', 'nearest').
        output_folder: str
            Folder where the plot will be saved.
        savename: str
            Name of the saved plot image.
        show_plot: bool
            Whether to display the plot.
        xrange, yrange: tuple (optional)
            Range for x and y axes in the format (min, max).
    """

    if len(f1_values) != len(f2_values) or len(f1_values) != len(f3_values):
        raise ValueError("All input arrays must have the same length.")

    # Create a grid for contouring
    grid_x, grid_y = np.linspace(min(f1_values),
                                 max(f1_values), 200),np.linspace(min(f2_values),
                                                                  max(f2_values), 200)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)

    # Interpolate f3 values on the grid
    grid_z = griddata((f1_values, f2_values), f3_values, (grid_x, grid_y), method=interp_method)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot filled contour for f3 values
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap='viridis')
    plt.colorbar(contour, ax=ax,
                 label="Objective 3 (Color)" if label_three is None else label_three)

    # Add contour lines
    ax.contour(grid_x, grid_y, grid_z, levels=10, colors='black', linewidths=0.5)

    # Set axis labels and title
    ax.set_xlabel("Objective 1" if label_one is None else label_one)
    ax.set_ylabel("Objective 2" if label_two is None else label_two)
    ax.set_title(title)

    # Apply axis ranges if specified
    if xrange is not None:
        ax.set_xlim(xrange)
    if yrange is not None:
        ax.set_ylim(yrange)
    # Show plot
    plt.tight_layout()
    if savename is not None:
        if not os.path.isdir(output_folder): os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, savename))
    if show_plot:
        plt.show()

    plt.close()

def get_color(index, start_idx, total, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=start_idx, vmax=start_idx + total - 1)
    return cmap(norm(index))

def plot_objective_function(ax, optimizer_result, times, time_vector, target_function, final_time, final_result,
                         best_x, best_f, pop_size, optimization_variables, objective_func_title, yellow_end=10, red_end=50):
    total_iters = len(optimizer_result)
    
    def count_in_range(start_gen, end_gen):
        return sum(start_gen <= (i // pop_size) <= end_gen for i in range(total_iters))
    
    yellow_count = count_in_range(0, yellow_end)
    red_count = count_in_range(yellow_end + 1, red_end)
    blue_count = total_iters - yellow_count - red_count

    for i, (pressure, time) in enumerate(zip(optimizer_result, times)):
        gen_num = i // pop_size
        if gen_num <= yellow_end:
            color = get_color(i, 0, yellow_count, 'YlOrBr')
        elif gen_num <= red_end:
            color = get_color(i, yellow_count, red_count, 'Reds')
        else:
            color = get_color(i, yellow_count + red_count, blue_count, 'Blues')
        ax.plot(time, pressure, color=color, alpha=0.8, linewidth=1)

    # Best fit and target
    ax.plot(final_time, final_result, label='Last params', color='black', linewidth=2)
    ax.plot(time_vector, target_function, label='Target Pressure', linestyle='--', color='orange', linewidth=2)

    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(objective_func_title, fontsize=14)
    ax.set_title('Objective Curve Fit Over Iterations', fontsize=16)
    ax.grid(True)

    # Legend
    legend_text = "Color legend:\nY: Gen 0–10\nR: Gen 11–50\nB: Gen 51+"
    ax.text(0.97, 0.97, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', alpha=0.6))

    # Best solution info
    textstr = "Best variables:\n" + "\n".join(
        f"{v['label'].split()[0]}: {val:.5f}" for v, val in zip(optimization_variables, best_x)
    ) + f"\nError = {best_f[0]:.2e}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10, bbox=props, verticalalignment='top')

def plot_error(ax, errors):
    ax.plot(range(1, len(errors) + 1), errors, marker='o', color='crimson', linewidth=1.5)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Total Error', fontsize=14)
    ax.set_ylim(0, 5)
    ax.set_title('Error Over Iterations', fontsize=16)
    ax.grid(True)

def plot_param_evolution(ax, values, title, ylabel, color='teal'):
    ax.plot(range(1, len(values) + 1), values, color=color, marker='o', markersize=3, linewidth=1)
    ax.set_xlabel('Iteration', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True)

def plot_all(
    iteration_curves_pressure,
    iteration_curves_time,
    iteration_errors,
    iteration_params,
    time_vector,
    target_pressure,
    final_sim_time,
    final_sim_pressure,
    best_x,
    best_f,
    pop_size,
    plot_dir,
    optimization_variables,
    objective_func_title,
    gen_range=None
):
    # Optionally filter errors, params, and curves based on gen_range
    if gen_range:
        start_gen, end_gen = gen_range
        start_idx = start_gen * pop_size
        end_idx = (end_gen + 1) * pop_size

        iteration_errors = iteration_errors[start_idx:end_idx]
        iteration_params = iteration_params[start_idx:end_idx]
        iteration_curves_pressure = iteration_curves_pressure[start_idx:end_idx]
        iteration_curves_time = iteration_curves_time[start_idx:end_idx]

    # Improved layout using matplotlib.gridspec
    import matplotlib.gridspec as gridspec
    num_vars = len(optimization_variables)
    fig = plt.figure(figsize=(18, 10), dpi=100, constrained_layout=True)
    gs = gridspec.GridSpec(2, max(4, num_vars), height_ratios=[3, 2], hspace=0.35, wspace=0.3)

    # Top row: pressure and error plots (make them span more columns to appear wider)
    ax1 = fig.add_subplot(gs[0, :2])
    ax2 = fig.add_subplot(gs[0, 2:4])

    # Bottom row: parameter evolution plots
    param_axes = []
    for i in range(num_vars):
        ax = fig.add_subplot(gs[1, i])
        param_axes.append(ax)

    # Pressure curves plot
    plot_objective_function(
        ax1, iteration_curves_pressure, iteration_curves_time,
        time_vector, target_pressure, final_sim_time,
        final_sim_pressure, best_x, best_f, pop_size, optimization_variables, objective_func_title
    )

    # Error plot
    plot_error(ax2, iteration_errors)

    # Parameter evolution (dynamic subplots)
    param_names = [var['label'] for var in optimization_variables]
    param_colors = [var.get('color', 'teal') for var in optimization_variables]
    for i, (name, color) in enumerate(zip(param_names, param_colors)):
        values = [p[i] for p in iteration_params]
        plot_param_evolution(param_axes[i], values, f"{name} Evolution", name, color=color)

    # Hide unused axes if num_vars < total_cols - 1 (because ax1, ax2 occupy [0,1] and [0,2])
    total_cols = max(3, num_vars)
    if num_vars < total_cols - 1:
        for j in range(num_vars, total_cols - 1):
            fig.add_subplot(gs[1, j]).set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_dir, dpi=300)
    print(f"✅ Plot saved to: {plot_dir}")
    plt.show()


def plot_curves(name_df, data_df, xvar_name, yvar_name, 
                tgt_xarr = None, tgt_yarr = None, scenario = 'best',
                output_folder = '.', savename = 'best_errors.png',
                xlabel = 'Time (s)', ylabel = 'Area ($m^2$)',
                show_plot = True):
    """ Plots data from dataframe based on the existing rows of another.
        Idea is that you did some filtering of a df (name_df)
        and will then only select the existing rows of that name_df in data_df.
        We'll also assume for now that if scenario is best, 'select_method'
        is used for labeling the curves.
    """
    assert scenario in ['all', 'best']

    if not os.path.isdir(output_folder): os.makedirs(output_folder)
    for idx, (_, row) in zip(name_df.index, name_df.iterrows()):
        label = None
        if scenario == 'best':
            label = row['select_method']
        plt.plot(data_df[xvar_name][idx], data_df[yvar_name][idx],
                marker = 'none', linestyle = '-', linewidth = 1.0, label = label)

    if tgt_xarr is not None and tgt_yarr is not None:
        plt.plot(tgt_xarr, tgt_yarr,
                    marker = 'None', color = 'k', linestyle = '--')
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(output_folder, savename))
    if show_plot:
        plt.show()
    plt.close()

def quick_two_dim_scatter_plot(x_values, y_values, color=None, size=30,
                       xlabel='X-axis', ylabel='Y-axis',
                       title='2D Scatter Plot', cmap='viridis',
                       output_folder = '.', savename = 'quick_scatter.png',
                       show_plot = True):
    """
    Creates a rapid 2D scatter plot with customization options.

    Parameters:
        x_values (array-like): X-axis values.
        y_values (array-like): Y-axis values.
        color (array-like, optional): Values to color the scatter points.
        size (float or int): Size of scatter points.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        title (str): Title of the scatter plot.
        cmap (str): Colormap for scatter points.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))

    if color is not None:
        scatter = plt.scatter(x_values, y_values, c=color,
                              s=size, cmap=cmap, alpha=0.75, edgecolor='k')
        plt.colorbar(scatter, label='Color Scale')
    else:
        plt.scatter(x_values, y_values, s=size, alpha=0.75, edgecolor='k')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, savename))
    if show_plot:
        plt.show()
    plt.close()

def plot_analytics_dict(dict_list: list[dict],
                        separate_by_param: bool = False,
                        file_prefix: str = None,
                        file_suffix: str = None,
                        output_folder: str = '.',
                        labels: list[str] = None,
                        show_plot = True): 
    """
    Plot analytics dictionaries (min, max, mean, median) and optionally save plots.

    Parameters:
    dict_list (list): List of analytics dictionaries.
    separate_by_param (bool): If True, creates a separate plot for each parameter.
    file_prefix (str): Optional prefix for saved plot filenames.
    file_suffix (str): Optional suffix for saved plot filenames.
    output_folder (str): Directory to save the plots. Default is current directory.
    labels (list): Optional list of labels for the x-axis when comparing multiple sets.
    """
    os.makedirs(output_folder, exist_ok=True)

    def save_plot(fig, title):
        filename = os.path.join(output_folder, (file_prefix or "") + title + (file_suffix or "") + ".png")
        fig.savefig(filename)
        print(f"Plot saved: {filename}")

    if separate_by_param:
        for param in dict_list[0].keys():
            fig, ax = plt.subplots(figsize=(8, 5))
            for idx, analytics_dict in enumerate(dict_list):
                if param in analytics_dict:
                    stats = analytics_dict[param]
                    ax.vlines(idx, stats['min'], stats['max'], colors='gray', lw=2)
                    ax.scatter(idx, stats['min'], color='red', marker='o', label='Min' if idx == 0 else "")
                    ax.scatter(idx, stats['max'], color='orange', marker='o', label='Max' if idx == 0 else "")
                    ax.scatter(idx, stats['mean'], color='blue', marker='x', label='Mean' if idx == 0 else "")
                    ax.scatter(idx, stats['median'], color='green', marker='d', label='Median' if idx == 0 else "")
            ax.set_title(f"Analytics - {param}")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xticks(range(len(dict_list)))
            # print("dict list:", dict_list[-1])
            if labels and len(labels) == len(dict_list):
                ax.set_xticklabels(labels, rotation=45, ha='right')
            else:
                ax.set_xticklabels([param], rotation=45, ha='right')
            plt.tight_layout()
            if show_plot:
                plt.show()
            save_plot(fig, f"Analytics_{param}")
            plt.close()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        positions = []
        labels_set = labels if labels else [f"Set {i+1}" for i in range(len(dict_list))]

        for idx, analytics_dict in enumerate(dict_list):
            for col_idx, (col, stats) in enumerate(analytics_dict.items()):
                pos = idx * len(analytics_dict) + col_idx
                positions.append(pos)
                ax.vlines(pos, stats['min'], stats['max'], colors='gray', lw=2)
                ax.scatter(pos, stats['min'], color='red', marker='o')
                ax.scatter(pos, stats['max'], color='orange', marker='o')
                ax.scatter(pos, stats['mean'], color='blue', marker='x')
                ax.scatter(pos, stats['median'], color='green', marker='d')

        ax.set_xticks(range(len(labels_set) * len(dict_list[0])))
        ax.set_xticklabels(labels_set * len(dict_list[0]), rotation=45, ha='right')
        ax.set_title("Analytics Single Boxplot")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        if show_plot:
                plt.show()
        save_plot(fig, "Analytics_Single_Boxplot")
        plt.close()

def plot_analytics(analytics_results, output_folder, original_analytics=None,
                   subfolder='.', param_labels=None):
    """
    Plot analytics comparison between the original data and multiple subsets
    filtered by branches in this case

    Parameters:
    - analytics_results (list): List of analytics results for each subset.
    - original_analytics (dict): Original data analytics
    - output_folder (str): Directory to save plots.
    - subfolder (str): Name of subfolder
    - param_labels (list): Optional labels for the different analytic dictionaries
    """
    plot_analytics_dict(
        analytics_results, 
        separate_by_param=True,  # <- Plot each parameter individually
        output_folder=os.path.join(output_folder, subfolder),
        labels=param_labels,     # <- Use the actual parameter labels
        show_plot=False
    )


def plot_curve_and_shape_star(df_select, df_data, time_arr, data_arr, outer_diam_grain,
                              output_folder, savename = 'tmp.png',
                              xvals = 'Time', yvals = 'Pressure'):
    """ Sequence of plotting the row entries of an input dataframe as well
        as the corresponding star shapes
    """
    plot_curves(df_select, df_data, xvals, yvals, 
                tgt_xarr = time_arr, tgt_yarr = data_arr, scenario = 'all',
                output_folder = output_folder, savename = savename.replace('.png','_curve.png'),
                show_plot= False)

    for index, row_vals in df_select.iterrows():
        plot_star_geometry(
        int(row_vals['number_branches']),  # Replace with actual column name
        row_vals['int_radius'],
        outer_diam_grain / 2,
        row_vals['web_burned'],
        row_vals['fill_radius'],
        row_vals['ang_coeff'],
        savefolder=output_folder,
        savename=savename.replace('.png', f"_shape_{index}.png"),
        show_plot=False
        )


def plot_subset_curves(subset_list, df_pressure, time_arr, BSA_array,
                       output_folder, subfolder='.', use_subset_nb_folder = True,
                       xvals= 'Time', yvals = 'Pressure'):
    """
    Plot time-pressure curves for each subset in the list.

    Parameters:
    - subset_list (list): List of data subsets to plot.
    - df_pressure (pd.DataFrame): Pressure dataframe for curve plotting.
    - time_arr (np.array): Time array for target values.
    - BSA_array (np.array): BSA array for target values.
    - output_folder (str): Directory to save plots.
    - subfolder (str): Name of subfolder
    - use_subset_nb_folder (bool): if True, creates a numbere subfolder for plots
    """
    subset_folder = os.path.join(output_folder, subfolder)
    os.makedirs(subset_folder, exist_ok=True)

    for ind, subset in enumerate(subset_list):
        subset_subfolder = subset_folder
        if use_subset_nb_folder:
            subset_subfolder = os.path.join(subset_folder, f"subset{ind + 1}")
            os.makedirs(subset_subfolder, exist_ok=True)
        
        plot_curves(
            subset, df_pressure, xvals, yvals, 
            tgt_xarr=time_arr, tgt_yarr=BSA_array, scenario='all',
            output_folder=subset_subfolder, savename='subset_curves.png',
            show_plot= False
        )

def rotate(points, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    return points @ R.T

def mirror_across_line(points, direction):
    direction = direction / np.linalg.norm(direction)
    projection = np.outer(np.dot(points, direction), direction)
    rejection = points - projection
    mirrored = projection - rejection
    return mirrored

def plot_star_geometry(N, Ri, Ro, w, f, e, savefolder = '.',
                       savename = None, show_plot = True):
    """ Function creates a sketch of the star shaped geometry based on the
        parametrization found in e.g. 
        Oh, Seok-Hwan, Tae-Seong Roh, and Hyoung Jin Lee.
        "A Study on the Optimal Design Method for Star-Shaped Solid
        Propellants through a Combination of Genetic Algorithm and
        Machine Learning." Aerospace 10.12 (2023): 979.

        Args:
            N (int): number of star branches
            Ri (float): internal radius
            Ro (float): extrenal radius
            w (float): web thickness
            f (float): fillet radius
            e (float): angle coefficient

        Returns:
            (None): plots / saves the resulting image
    """

    # if savename is not None:
    os.makedirs(savefolder, exist_ok=True)
    if not isinstance(N, int): 
        N = int(N) # just to make sure input is integer
        print(f"Converting input number of star branches to integer {N}")
    Rp = Ro - w - f
    angle1 = np.pi * e / N
    angle2 = np.pi / N
    theta = 2 * np.arctan((Rp * np.sin(angle1) * np.tan(angle1)) /
                          (Rp * np.sin(angle1) - Ri * np.tan(angle1)))
    half_theta = theta / 2

    Line_theta_dir = np.array([np.cos(half_theta), np.sin(half_theta)])
    Line2_dir = np.array([np.cos(angle1), np.sin(angle1)])
    Line4_dir = np.array([np.cos(angle2), np.sin(angle2)])

    Point0 = np.array([0, 0])
    P_Ri = np.array([Ri, 0])
    A = np.array([Line_theta_dir, -Line2_dir]).T
    b = Point0 - P_Ri
    t_values = np.linalg.solve(A, b)
    Point1 = P_Ri + t_values[0] * Line_theta_dir
    Point2 = Point1 + f * Line2_dir

    # Arc1
    angle_start_arc1 = np.arctan2(Point2[1], Point2[0])
    arc1_angles = np.linspace(angle_start_arc1, angle2, 100)
    arc1_pts = np.column_stack([
        (Rp + f) * np.cos(arc1_angles),
        (Rp + f) * np.sin(arc1_angles)
    ])

    # Arc2
    angle_start_arc2 = np.arctan2(Point2[1] - Point1[1], Point2[0] - Point1[0])
    arc2_angles = np.linspace(angle_start_arc2, angle_start_arc2 - 2 * np.pi, 10000)
    arc2_coords = []
    arc2_end = None
    for i in range(1, len(arc2_angles)):
        a1 = arc2_angles[i - 1]
        a2 = arc2_angles[i]
        p1 = Point1 + f * np.array([np.cos(a1), np.sin(a1)])
        p2 = Point1 + f * np.array([np.cos(a2), np.sin(a2)])
        tangent_vec = p2 - p1
        tangent_unit = tangent_vec / np.linalg.norm(tangent_vec)
        angle_diff = np.arccos(np.clip(np.dot(tangent_unit, Line_theta_dir), -1.0, 1.0))
        arc2_coords.append(p2)
        if np.isclose(angle_diff, np.deg2rad(180), atol=1e-2):
            arc2_end = p2
            break
    arc2_coords = np.array(arc2_coords)
    if arc2_end is None:
        arc2_end = arc2_coords[-1]

    # Line5
    t_line5 = -arc2_end[1] / Line_theta_dir[1]
    Point3 = arc2_end + t_line5 * Line_theta_dir
    line5 = np.array([arc2_end, Point3])

    # Arc3
    arc3_angles = np.linspace(0, angle2, 100)
    arc3_pts = np.column_stack([
        Ro * np.cos(arc3_angles),
        Ro * np.sin(arc3_angles)
    ])

    # Geometry to plot (Line1 and Line4 omitted)
    base_segments = {
        "Line5": line5,
        "Arc1": arc1_pts,
        "Arc2": arc2_coords,
        "Arc3": arc3_pts
    }

    mirrored_segments = {
        label: mirror_across_line(seg, Line4_dir)
        for label, seg in base_segments.items()
    }

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_title(f"Geometry with Mirror (N={N})")
    ax.grid(True, linestyle='--', alpha=0.4)

    for i in range(N):
        angle = i * 2 * np.pi / N
        for label, seg in base_segments.items():
            seg_rot = rotate(seg, angle)
            ax.plot(seg_rot[:, 0], seg_rot[:, 1], lw=1, label=label if i == 0 else "")
        for seg in mirrored_segments.values():
            seg_rot = rotate(seg, angle)
            ax.plot(seg_rot[:, 0], seg_rot[:, 1], lw=1)

    # ax.legend()
    if savename is not None:
        plt.savefig(os.path.join(savefolder, savename))
    if show_plot:
        plt.show()
    plt.close()
    # fig.savefig(f"WORK/geometry_circular.png", dpi=300, bbox_inches='tight')

def plot_star_petal(N, Ri, Ro, w, f, e):
    Rp = Ro - w - f  # Rp is now derived from Ro and w
    angle1 = np.pi * e / N
    angle2 = np.pi / N
    theta = 2 * np.arctan((Rp * np.sin(angle1) * np.tan(angle1)) /
                          (Rp * np.sin(angle1) - Ri * np.tan(angle1)))
    half_theta = theta / 2

    # Direction vectors
    Line_theta_dir = np.array([np.cos(half_theta), np.sin(half_theta)])
    Line2_dir = np.array([np.cos(angle1), np.sin(angle1)])
    Line4_dir = np.array([np.cos(angle2), np.sin(angle2)])

    # Base points
    Point0 = np.array([0, 0])
    P_Ri = np.array([Ri, 0])
    A = np.array([Line_theta_dir, -Line2_dir]).T
    b = Point0 - P_Ri
    t_values = np.linalg.solve(A, b)
    Point1 = P_Ri + t_values[0] * Line_theta_dir
    Point2 = Point1 + f * Line2_dir

    # Arc1 from center (Point0)
    angle_start_arc1 = np.arctan2(Point2[1], Point2[0])
    arc1_angles = np.linspace(angle_start_arc1, angle2, 100)
    arc1_radius = Rp + f
    arc1_x = arc1_radius * np.cos(arc1_angles)
    arc1_y = arc1_radius * np.sin(arc1_angles)

    # Arc2 from Point2 (centered at Point1), clockwise, stop when tangent to Line5
    angle_start_arc2 = np.arctan2(Point2[1] - Point1[1], Point2[0] - Point1[0])
    arc2_angles = np.linspace(angle_start_arc2, angle_start_arc2 - 2*np.pi, 1000)
    arc2_coords = []
    arc2_end = None

    for i in range(1, len(arc2_angles)):
        a1 = arc2_angles[i - 1]
        a2 = arc2_angles[i]
        p1 = Point1 + f * np.array([np.cos(a1), np.sin(a1)])
        p2 = Point1 + f * np.array([np.cos(a2), np.sin(a2)])
        tangent_vec = p2 - p1
        tangent_unit = tangent_vec / np.linalg.norm(tangent_vec)
        angle_diff = np.arccos(np.clip(np.dot(tangent_unit, Line_theta_dir), -1.0, 1.0))
        # print(f"Angle diff: {angle_diff:.4f} radians")  # Optional debug

        arc2_coords.append(p2)
        if np.isclose(angle_diff, np.deg2rad(180), atol=1e-2):
            arc2_end = p2
            break

    arc2_coords = np.array(arc2_coords)
    if arc2_end is None:
        arc2_end = arc2_coords[-1]

    # Line5: from arc2_end along Line_theta_dir until it hits y=0
    t_line5 = -arc2_end[1] / Line_theta_dir[1]
    Point3 = arc2_end + t_line5 * Line_theta_dir

    # Arc3 (center at Point0, radius Ro)
    arc3_angles = np.linspace(0, angle2, 100)
    arc3_x = Ro * np.cos(arc3_angles)
    arc3_y = Ro * np.sin(arc3_angles)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Lines
    ax.plot([0, Ro], [0, 0], 'k', label='Line1')
    ax.plot([0, 1.2*Ro*Line2_dir[0]], [0, 1.2*Ro*Line2_dir[1]], 'b--', label='Line2')
    ax.plot([Ri, Point1[0]], [0, Point1[1]], 'r', label='θ/2 Line')
    ax.plot([Point1[0], Point2[0]], [Point1[1], Point2[1]], 'g', label='Line3')
    ax.plot([0, 1.2*Ro*Line4_dir[0]], [0, 1.2*Ro*Line4_dir[1]], 'purple', label='Line4')
    ax.plot([arc2_end[0], Point3[0]], [arc2_end[1], Point3[1]], 'orange', label='Line5 (tangent)')

    # Arcs
    ax.plot(arc1_x, arc1_y, 'c', label='Arc1 (Rp + f)')
    ax.plot(arc2_coords[:, 0], arc2_coords[:, 1], 'm', label='Arc2 (tangent to Line5)')
    ax.plot(arc3_x, arc3_y, 'brown', label='Arc3 (Ro perimeter)')

    # Annotated points
    for pt, name in zip([Point0, Point1, Point2, arc2_end, Point3],
                        ['Point0', 'Point1', 'Point2', 'Arc2_End', 'Point3']):
        ax.plot(*pt, 'o')
        ax.annotate(name, pt + [0.02, 0.01])

    ax.set_title("Arc2 Ends When Tangent to Line5 (Using w)")
    ax.legend()
    plt.show()
    # fig.savefig(f"WORK/geometry_base.png", dpi=300, bbox_inches='tight')

def plot_truncated_star_geometry(N, r, f, w, Ro):
    TOL = 1e-4
    center = (0.0, 0.0)
    angle = math.pi / N
    l = Ro - w - f

    # Define base points
    point1 = (r, 0.0)
    point3 = (l * math.cos(angle), l * math.sin(angle))
    reverse_dir = (-point3[0], -point3[1])
    point4 = (point3[0] + f * math.cos(angle), point3[1] + f * math.sin(angle))
    point5 = (Ro * math.cos(angle), Ro * math.sin(angle))
    point6 = (
        point3[0] + f * math.cos(angle - math.pi / 2),
        point3[1] + f * math.sin(angle - math.pi / 2)
    )
    point7 = (Ro, 0.0)

    # Arc points for candidate intersection (Line6 with Line2)
    arc_points_candidate = [(r * math.cos(t), r * math.sin(t)) for t in [angle * i / 500 for i in range(501)]]

    def line6_param(t, direction):
        return (point6[0] + t * direction[0], point6[1] + t * direction[1])

    def point_to_segment_distance(px, py, ax, ay, bx, by):
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        ab_len_sq = abx**2 + aby**2
        if ab_len_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
        proj_x, proj_y = ax + t * abx, ay + t * aby
        return math.hypot(px - proj_x, py - proj_y)

    def find_intersection():
        for i in range(1000):
            t = i * 0.001
            p = line6_param(t, reverse_dir)
            for a, b in zip(arc_points_candidate[:-1], arc_points_candidate[1:]):
                if point_to_segment_distance(p[0], p[1], a[0], a[1], b[0], b[1]) < TOL:
                    return p
        return None

    point2 = find_intersection() or (
        point6[0] + f * reverse_dir[0] / math.hypot(*reverse_dir),
        point6[1] + f * reverse_dir[1] / math.hypot(*reverse_dir)
    )
    line6 = (point6, point2)

    # Arc constructors
    def arc(center, radius, start_angle, end_angle, steps=100):
        if end_angle < start_angle:
            end_angle += 2 * math.pi
        return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
                for a in [start_angle + i * (end_angle - start_angle) / steps for i in range(steps + 1)]]

    def arc_cw(center, radius, start_angle, end_angle, steps=100):
        if start_angle < end_angle:
            start_angle += 2 * math.pi
        return [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a))
                for a in [start_angle - i * (start_angle - end_angle) / steps for i in range(steps + 1)]]

    # Mirror and rotate
    def mirror_point(p, a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        d2 = dx*dx + dy*dy
        if d2 == 0: return p
        t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d2
        proj = (a[0] + t * dx, a[1] + t * dy)
        return (2 * proj[0] - p[0], 2 * proj[1] - p[1])

    def rotate_point(p, angle, origin=(0, 0)):
        x, y = p[0] - origin[0], p[1] - origin[1]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        return (cos_a * x - sin_a * y + origin[0], sin_a * x + cos_a * y + origin[1])

    # Define base arcs
    theta2 = math.atan2(point2[1], point2[0])
    if theta2 < 0: theta2 += 2 * math.pi

    arc2 = arc(center, r, 0, theta2)
    arc5 = arc_cw(point3, f, angle, angle - math.pi / 2)
    arc8 = arc_cw(center, Ro, math.atan2(point5[1], point5[0]), 0)

    # Generate pattern
    pattern = []
    for i in range(N):
        a = i * 2 * math.pi / N
        orig = {
            "line6": [rotate_point(p, a, center) for p in line6],
            "arc2": [rotate_point(p, a, center) for p in arc2],
            "arc5": [rotate_point(p, a, center) for p in arc5],
            "arc8": [rotate_point(p, a, center) for p in arc8],
        }
        mirr = {
            "line6": [rotate_point(mirror_point(p, center, point3), a, center) for p in line6],
            "arc2": [rotate_point(mirror_point(p, center, point3), a, center) for p in arc2],
            "arc5": [rotate_point(mirror_point(p, center, point3), a, center) for p in arc5],
            "arc8": [rotate_point(mirror_point(p, center, point3), a, center) for p in arc8],
        }
        pattern.append((orig, mirr))

    # Plotting
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Finocyl Geometry (Patterned)")
    ax.grid(True)

    for orig, mirr in pattern:
        ax.plot(*zip(*orig["arc2"]), color='red')
        ax.plot(*zip(*orig["arc5"]), color='brown')
        ax.plot(*zip(*orig["arc8"]), color='black')
        ax.plot(*zip(*orig["line6"]), color='orange')
        ax.plot(*zip(*mirr["arc2"]), color='red', linestyle='--')
        ax.plot(*zip(*mirr["arc5"]), color='brown', linestyle='--')
        ax.plot(*zip(*mirr["arc8"]), color='black', linestyle='--')
        ax.plot(*zip(*mirr["line6"]), color='orange', linestyle='--')

    plt.show()

def plot_truncated_star_petal(N, r, f, w, Ro):
    TOL = 1e-4  # Intersection distance threshold

    # -----------------------
    # Derived geometry
    # -----------------------
    l = Ro - w - f
    angle = math.pi / N
    center = (0.0, 0.0)
    point1 = (r, 0.0)
    point3 = (l * math.cos(angle), l * math.sin(angle))
    reverse_dir = (-point3[0], -point3[1])
    point4 = (point3[0] + f * math.cos(angle), point3[1] + f * math.sin(angle))
    point5 = (Ro * math.cos(angle), Ro * math.sin(angle))
    angle_line5_end = angle - math.pi / 2
    point6 = (
        point3[0] + f * math.cos(angle_line5_end),
        point3[1] + f * math.sin(angle_line5_end)
    )
    point7 = (Ro, 0.0)

    # Arc points for Line2 intersection search
    arc_points_candidate = [
        (r * math.cos(t), r * math.sin(t))
        for t in [0 + (angle - 0) * i / 500 for i in range(501)]
    ]

    def line6_param(t, direction):
        return (
            point6[0] + t * direction[0],
            point6[1] + t * direction[1]
        )

    def point_to_segment_distance(px, py, ax, ay, bx, by):
        abx = bx - ax
        aby = by - ay
        apx = px - ax
        apy = py - ay
        ab_len_sq = abx ** 2 + aby ** 2
        if ab_len_sq == 0:
            return math.hypot(px - ax, py - ay)
        t = max(0, min(1, (apx * abx + apy * aby) / ab_len_sq))
        proj_x = ax + t * abx
        proj_y = ay + t * aby
        return math.hypot(px - proj_x, py - proj_y)

    def find_intersection():
        for i in range(1000):
            t = i * 0.001
            p = line6_param(t, reverse_dir)
            for a, b in zip(arc_points_candidate[:-1], arc_points_candidate[1:]):
                d = point_to_segment_distance(p[0], p[1], a[0], a[1], b[0], b[1])
                if d < TOL:
                    return p
        return None

    point2 = find_intersection()
    if not point2:
        norm = math.hypot(*reverse_dir)
        unit_dir = (reverse_dir[0] / norm, reverse_dir[1] / norm)
        point2 = (
            point6[0] + f * unit_dir[0],
            point6[1] + f * unit_dir[1]
        )
    line6 = (point6, point2)

    # Arc Line2 (Point1 to Point2)
    theta1 = 0.0
    theta2 = math.atan2(point2[1], point2[0])
    if theta2 < 0:
        theta2 += 2 * math.pi
    arc_points = [
        (r * math.cos(t), r * math.sin(t))
        for t in [theta1 + (theta2 - theta1) * i / 500 for i in range(501)]
    ]

    # Arc Line8 (CW): Point5 → Point7
    angle5 = math.atan2(point5[1], point5[0])
    angle7 = 0.0
    if angle7 > angle5:
        angle5 += 2 * math.pi
    arc8_points = [
        (Ro * math.cos(t), Ro * math.sin(t))
        for t in [angle5 - (angle5 - angle7) * i / 500 for i in range(501)]
    ]

    # Arc Line5 (90° CW): centered at Point3
    arc5_points = [
        (point3[0] + f * math.cos(t), point3[1] + f * math.sin(t))
        for t in [angle - i * (math.pi / 2) / 100 for i in range(101)]
    ]

    # -----------------------
    # Plotting
    # -----------------------
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_title("Finocyl Geometry – Single Petal")
    ax.grid(True)

    def plot_line(p1, p2, color, label):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, label=label)

    # Lines
    plot_line(center, point1, 'blue', 'Line1: center → Point1')
    plot_line(center, point3, 'green', 'Line3: center → Point3')
    plot_line(point3, point4, 'purple', 'Line4: Point3 → Point4')
    plot_line(line6[0], line6[1], 'orange', 'Line6: Point6 → Point2')
    plot_line(center, point7, 'teal', 'Line7: center → Point7')
    plot_line(point4, point5, 'magenta', 'Line9: Point4 → Point5')

    # Arcs
    ax.plot(*zip(*arc_points), 'red', label='Line2: Arc (Point1 → Point2)')
    ax.plot(*zip(*arc8_points), 'cyan', label='Line8: Arc (Point5 → Point7, CW)')
    ax.plot(*zip(*arc5_points), 'brown', label='Line5: Arc (90° CW)')

    # Points
    points = {
        'Center': center,
        'Point1': point1,
        'Point2': point2,
        'Point3': point3,
        'Point4': point4,
        'Point5': point5,
        'Point6': point6,
        'Point7': point7
    }
    for label, (x, y) in points.items():
        ax.plot(x, y, 'ko')
        ax.text(x, y, f' {label}', fontsize=8, verticalalignment='bottom')

    ax.legend(loc='upper right')
    plt.show()