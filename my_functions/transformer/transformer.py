import pandas as pd
import numpy as np
from scipy.optimize import fsolve
from ruamel.yaml import YAML

class ThrustToPressureTransformer:
    def __init__(self, ric_file_path):
        # Load the motor file
        yaml = YAML()
        with open(ric_file_path, 'r') as f:
            motor_data = yaml.load(f)['data']

        prop_tab = motor_data['propellant']['tabs'][0]
        nozzle = motor_data['nozzle']
        config = motor_data['config']

        # Extract required parameters from .ric
        self.gamma = prop_tab['k']
        self.P_exit = config.get('ambPressure', 101325)
        self.P_ambient = config.get('ambPressure', 101325)
        self.D_throat = nozzle['throat']
        self.A_throat = np.pi * (self.D_throat / 2)**2
        self.A_exit = np.pi * (nozzle['exit'] / 2)**2
        self.eta_throat = 0.95
        self.eta_nozzle = 0.9
        self.eta_skin = 0.95
        self.theta_div = np.radians(nozzle.get('divAngle', 15))

        # Precompute constant
        self.K = ((1 + np.cos(self.theta_div)) / 2) * self.eta_throat * self.eta_nozzle * self.A_throat

    def C_F_ideal(self, P_chamber):
        γ = self.gamma
        if P_chamber <= self.P_exit:
            return 0  # Prevent sqrt of negative

        term1 = (2 * γ**2) / (γ - 1)
        term2 = (2 / (γ + 1))**((γ + 1) / (γ - 1))
        term3 = 1 - (self.P_exit / P_chamber)**((γ - 1) / γ)

        if term3 <= 0:
            return 0

        sqrt_part = np.sqrt(term1 * term2 * term3)
        pressure_term = ((self.P_exit - self.P_ambient) * self.A_exit) / (self.A_throat * P_chamber)
        return sqrt_part + pressure_term

    def solve_P_chamber(self, F):
        def equation(P):
            if P <= self.P_exit:
                return 1e6
            C_F = self.C_F_ideal(P)
            return self.K * (self.eta_skin * C_F + (1 - self.eta_skin)) * P - F

        P_guess = 5e5
        P_solution, = fsolve(equation, P_guess)
        return P_solution

    def transform(self, thrust_vector):
        return np.array([self.solve_P_chamber(F) for F in thrust_vector])

    def process_thrust_csv(self, csv_path):
        """
        Reads a CSV file with a thrust curve and adds a computed chamber pressure curve.
        The thrust column must be labeled 'Target Thrust Curve[N]'.
        The resulting DataFrame is saved back to the same CSV with a new column
        'Target Chamber Pressure[Pa]'.
        """
        # Load CSV file
        df = pd.read_csv(csv_path)

        # Check required column
        if "Target Thrust Curve[N]" not in df.columns:
            raise ValueError("CSV must contain a column named 'Target Thrust Curve[N]'")

        # Transform thrust to pressure
        thrust_vector = df["Target Thrust Curve[N]"].values
        pressure_vector = self.transform(thrust_vector)

        # Add new column and save
        df["Chamber Pressure(Pa)"] = pressure_vector
        df.to_csv(csv_path, index=False)



# import numpy as np
# from scipy.optimize import fsolve

# class ThrustToPressureTransformer:
#     def __init__(self,
#                  gamma=1.22,
#                  P_exit=101325,          # Pa
#                  P_ambient=101325,       # Pa
#                  D_throat=0.05,          # m
#                  A_exit=0.008,           # m²
#                  eta_throat=0.98,
#                  eta_nozzle=0.95,
#                  eta_skin=0.96,
#                  theta_div_deg=15):      # degrees

#         # Store all parameters
#         self.gamma = gamma
#         self.P_exit = P_exit
#         self.P_ambient = P_ambient
#         self.D_throat = D_throat
#         self.A_throat = np.pi * (D_throat / 2)**2
#         self.A_exit = A_exit
#         self.eta_throat = eta_throat
#         self.eta_nozzle = eta_nozzle
#         self.eta_skin = eta_skin
#         self.theta_div = np.radians(theta_div_deg)

#         # Precompute constant K
#         self.K = ((1 + np.cos(self.theta_div)) / 2) * self.eta_throat * self.eta_nozzle * self.A_throat

#     def C_F_ideal(self, P_chamber):
#         γ = self.gamma
#         term1 = (2 * γ**2) / (γ - 1)
#         term2 = (2 / (γ + 1))**((γ + 1) / (γ - 1))
#         term3 = 1 - (self.P_exit / P_chamber)**((γ - 1) / γ)
#         sqrt_part = np.sqrt(term1 * term2 * term3)

#         pressure_term = ((self.P_exit - self.P_ambient) * self.A_exit) / (self.A_throat * P_chamber)
#         return sqrt_part + pressure_term

#     def solve_P_chamber(self, F):
#         def equation(P):
#             if P <= 0:
#                 return 1e6  # Penalize non-physical solutions
#             C_F = self.C_F_ideal(P)
#             return self.K * (self.eta_skin * C_F + (1 - self.eta_skin)) * P - F

#         P_guess = 5e5  # Initial guess in Pa
#         P_solution, = fsolve(equation, P_guess)
#         return P_solution

#     def transform(self, thrust_vector):
#         """
#         Transforms a thrust-time vector to pressure-time vector.
#         Input: thrust_vector [N]
#         Output: pressure_vector [Pa]
#         """
#         return np.array([self.solve_P_chamber(F) for F in thrust_vector])