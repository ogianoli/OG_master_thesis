import sys
import os
from master_thesis.my_tool.GUI_test.predict_class_from_file_GUI import predict_class_from_file

def run_pipeline(csv_path):
    # === RUN CLASSIFICATION ===
    predicted_class, predicted_prob = predict_class_from_file(csv_path)
    print(predicted_class)  # 0 = bates, 1 = star, 2 = endburner
    print(f"\nPredicted class: {predicted_class}")

    # === CLASS-TO-OPTIMIZATION MAPPING ===
    if predicted_class == 0:
        from the_main_bates_CLI import run_optimization as run_func
    elif predicted_class == 1:
        from the_main_star_CLI import run_optimization as run_func
    elif predicted_class == 2:
        from the_main_endburner_CLI import run_optimization as run_func
    else:
        print(f"❌ No optimization function defined for predicted class {predicted_class}")
        return None
    if predicted_class == 0:
        motor_file = "master_thesis/my_tool/GUI_test/motors/motor_bates.ric"
    elif predicted_class == 1:
        motor_file = "master_thesis/my_tool/GUI_test/motors/motor_star.ric"
    elif predicted_class == 2:
        motor_file = "master_thesis/my_tool/GUI_test/motors/motor_endburner.ric"
    output_dir = "master_thesis/my_tool/GUI_test/data/manual_run"
    pop_size = 20
    generations = 30
    print(f"Launching optimization for class {predicted_class} using {run_func.__module__}")
    result = run_func(csv_path, motor_file, output_dir, pop_size, generations)
    return result

if __name__ == "__main__":
    # CLI usage: python run_pipeline_GUI.py <csv_path>
    if len(sys.argv) > 1:
        input_curve_csv = sys.argv[1]
    else:
        input_curve_csv = "master_thesis/my_tool/GUI_test/target/target_pressure_curve2.csv"
    run_pipeline(input_curve_csv)