import importlib.util
import pandas as pd

# Load the prediction module dynamically
module_path = "master_thesis/my_tool/GUI_test/predict_class_from_file_GUI.py"
spec = importlib.util.spec_from_file_location("predictor", module_path)
predictor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor)

# Path to your input CSV file
csv_path = "master_thesis/my_tool/GUI_test/target/target_pressure_curve.csv"  # <-- Change this to your actual file

results = []

for model_num in range(1, 11):
    predictor.model_num = model_num
    pred_class, probs = predictor.predict_class_from_file(csv_path)
    row = {"Model": model_num, "Predicted Class": pred_class}
    for class_id, prob in probs:
        row[f"Prob_Class_{class_id}"] = round(prob, 4)
    results.append(row)

df = pd.DataFrame(results)
print(df.to_string(index=False))