# Subsystem: misc

## main.py
- Layer: utility
- Language: py
- Symbols:
  - `convert_ip_to_octets` (function, line 15) `def convert_ip_to_octets(ip)`
  - `preprocess_packet` (function, line 18) `def preprocess_packet(packet)`
  - `capture_packet` (function, line 23) `def capture_packet()`
  - `extract_features` (function, line 34) `def extract_features(packet)`
  - `create_database` (function, line 54) `def create_database()`
  - `store_positive_packet` (function, line 62) `def store_positive_packet(packet)`
  - `train_model` (function, line 73) `def train_model(X_train, y_train, X_val, y_val)`
  - `evaluate_model` (function, line 81) `def evaluate_model(model, X_val, y_val)`
  - `real_time_packet_capture` (function, line 85) `def real_time_packet_capture(model)`
