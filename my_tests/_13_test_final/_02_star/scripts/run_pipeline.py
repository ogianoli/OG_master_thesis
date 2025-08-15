import sys
import os
from master_thesis.my_tool.GUI_doublemotor.predict_class_from_file_GUI import predict_class_from_file
from master_thesis.my_tool.GUI_doublemotor.helper_functions_GUI import get_next_data_num, get_valid_data_dir
# from master_thesis.my_functions.transformer.transformer import ThrustToPressureTransformer

def run_pipeline(csv_path = ""):
    # === RUN CLASSIFICATION ===
    data_num = get_next_data_num(base_path="master_thesis/my_tests/_13_test_final/_02_star/data/")
    output_dir = get_valid_data_dir(data_num, base_path="master_thesis/my_tests/_13_test_final/_02_star/data/")
    csv_path = "master_thesis/my_tests/_13_test_final/_02_star/results/target.csv"
    motor_file = "master_thesis/my_tests/_13_test_final/_02_star/motor.ric"
    pop_size = 50
    generations = 50
    # transformer = ThrustToPressureTransformer(motor_file)
    # transformer.process_thrust_csv(csv_path)
    predicted_class = predict_class_from_file(csv_path)[0]

    print(predicted_class)  # 0 = bates, 1 = star, 2 = endburner
    print(f"\n🧠 Predicted class: {predicted_class}")

    # === CLASS-TO-OPTIMIZATION MAPPING ===
    if predicted_class == 0:
        from the_main_star_GUI import run_optimization as run_func
    elif predicted_class == 1:
        from the_main_star_GUI import run_optimization as run_func
    elif predicted_class == 2:
        from the_main_star_GUI import run_optimization as run_func
    else:
        print(f"❌ No optimization function defined for predicted class {predicted_class}")
        return None

    # Example values for motor_file, output_dir, pop_size, generations
    
    print(f"🚀 Launching optimization for class {predicted_class} using {run_func.__module__}")
    result = run_func(csv_path, motor_file, output_dir, pop_size, generations)
    return result

if __name__ == "__main__":
    # CLI usage: python run_pipeline_GUI.py <csv_path>
    if len(sys.argv) > 1:
        input_curve_csv = sys.argv[1]
        run_pipeline(input_curve_csv)
    else:
        run_pipeline()