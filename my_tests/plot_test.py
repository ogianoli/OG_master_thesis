import matplotlib.pyplot as plt
import pandas as pd
import io

# Read the data from the CSV file into a Pandas DataFrame
df = pd.read_csv("01_test_bates/results/results.csv")

# Define the parameters to plot
parameters = {
    "Chamber Pressure(Pa)": "Chamber Pressure (Pa)",
    "Thrust(N)": "Thrust (N)",
    "Propellant Mass(G1;g)": "Propellant Mass (g)",
    "Regression Depth(G1;mm)": "Regression Depth (mm)"
}

# Create individual plots for each parameter
# for param, label in parameters.items():
#     plt.figure(figsize=(8, 5))
#     plt.plot(df["Time(s)"], df[param], marker="o", linestyle="-")
#     plt.xlabel("Time (s)")
#     plt.ylabel(label)
#     plt.title(f"{label} vs Time")
#     plt.grid()
    # plt.show()

    # Save the individual plot
    # plot_filename = f"test/data/my_test/01_test_bates/results/{param.replace(' ', '_').replace(';', '').replace('(', '').replace(')', '')}.png"
    # plt.savefig(plot_filename)

# Create a single image with all four plots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

for ax, (param, label) in zip(axs.flatten(), parameters.items()):
    ax.plot(df["Time(s)"], df[param], marker="o", linestyle="-")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs Time")
    ax.grid()

plt.tight_layout()
combined_plot_filename = "test/data/my_test/01_test_bates/results/combined_plots2.png"
plt.savefig(combined_plot_filename)

time_steps = df["Time(s)"].diff().fillna(0)
total_impulse = (df["Thrust(N)"] * time_steps).sum()

# Add a panel for the total impulse
fig.add_subplot(2, 2, 3, frame_on=False)
plt.text(0.5, 0.5, f"Total Impulse: {total_impulse:.2f} Ns", fontsize=12, ha='center', va='center', transform=fig.transFigure)
plt.axis('off')

plt.tight_layout()
combined_plot_filename = "test/data/my_test/01_test_bates/results/combined_plots_with_impulse.png"
plt.savefig(combined_plot_filename)
plt.show()
# Calculate the total impulse

