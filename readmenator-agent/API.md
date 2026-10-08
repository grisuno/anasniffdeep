# API

## main.py
- `convert_ip_to_octets` (function) `main.py:15` `def convert_ip_to_octets(ip)`
- `preprocess_packet` (function) `main.py:18` `def preprocess_packet(packet)`
- `capture_packet` (function) `main.py:23` `def capture_packet()`
- `extract_features` (function) `main.py:34` `def extract_features(packet)`
- `create_database` (function) `main.py:54` `def create_database()`
- `store_positive_packet` (function) `main.py:62` `def store_positive_packet(packet)`
- `train_model` (function) `main.py:73` `def train_model(X_train, y_train, X_val, y_val)`
- `evaluate_model` (function) `main.py:81` `def evaluate_model(model, X_val, y_val)`
- `real_time_packet_capture` (function) `main.py:85` `def real_time_packet_capture(model)`
