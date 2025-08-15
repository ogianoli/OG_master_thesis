# from master_thesis.my_functions.LD_Ratio.util.helper_functions import dataframe_analytics
# from master_thesis.my_functions.LD_Ratio.util.plot_functions import plot_analytics
# import pandas as pd
# import numpy as np


# output_folder = "master_thesis/my_functions/LD_Ratio/endburner/no_fail"

# # my_df = pd.DataFrame(
# #     {
# #         "column_1": [5, 19, 215, 400, 888],
# #         "column_2": [0, 123, 234, 456, 700],
# #         "column_3": [100, 109, 815, 870, 955]
# #     }
# # )

# # input_params = ["column_1", "column_2", "column_3"]


# # Load parameters from CSV file
# csv_path = "master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_endburner.csv"
# params_df = pd.read_csv(csv_path)

# # Compute additional columns
# params_df["LD_Ratio"] = params_df["length"] / params_df["diameter"]
# # params_df["WD_Ratio"] = ((params_df["diameter"] - params_df["coreDiameter"]) / 2) / (params_df["diameter"] / 2)

# # Define the input parameters to analyze
# input_params = params_df.columns.to_list()


# df_analytics = dataframe_analytics(params_df[input_params[:]])

# print([df_analytics][0].keys())

# plot_analytics([df_analytics], output_folder, 'subset_user', param_labels=input_params,)

from master_thesis.my_functions.LD_Ratio.util.helper_functions import dataframe_analytics
from master_thesis.my_functions.LD_Ratio.util.plot_functions import plot_analytics
import pandas as pd
import numpy as np

output_folder = "master_thesis/my_functions/LD_Ratio/endburner/no_fail2"

# Load parameters from CSV file
csv_path = "master_thesis/my_functions/LD_Ratio/util/no_fail_dataset/no_fail_dataset_endburner.csv"
params_df = pd.read_csv(csv_path)

# Simple filtering check
diameter_min, diameter_max = 0.0254, 0.0977975
length_min, length_max = 0.132, 0.381

params_df = params_df[
    (params_df["diameter"] > diameter_min) & (params_df["diameter"] < diameter_max) &
    (params_df["length"] > length_min) & (params_df["length"] < length_max)
]

# Compute additional columns
params_df["LD_Ratio"] = params_df["length"] / params_df["diameter"]
# params_df["WD_Ratio"] = ((params_df["diameter"] - params_df["coreDiameter"]) / 2) / (params_df["diameter"] / 2)

# Define the input parameters to analyze
input_params = params_df.columns.to_list()

df_analytics = dataframe_analytics(params_df[input_params[:]])

print(list(df_analytics.keys()))

plot_analytics([df_analytics], output_folder, 'subset_user', param_labels=input_params)