from transformer import ThrustToPressureTransformer
motor_file_path = "master_thesis/my_tests/11_test_transformer/motor2.ric"
csv_path = "master_thesis/my_tests/11_test_transformer/targets2.csv"
transformer = ThrustToPressureTransformer(motor_file_path)
transformer.process_thrust_csv(csv_path)