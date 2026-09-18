# API

## main.py

### convert_ip_to_octets (function) `def convert_ip_to_octets(ip)`
- Defined: `main.py:15`

### preprocess_packet (function) `def preprocess_packet(packet)`
- Defined: `main.py:18`

### capture_packet (function) `def capture_packet()`
- Defined: `main.py:23`

### extract_features (function) `def extract_features(packet)`
- Defined: `main.py:34`

### create_database (function) `def create_database()`
- Defined: `main.py:54`

### store_positive_packet (function) `def store_positive_packet(packet)`
- Defined: `main.py:62`

### train_model (function) `def train_model(X_train, y_train, X_val, y_val)`
- Defined: `main.py:73`

### evaluate_model (function) `def evaluate_model(model, X_val, y_val)`
- Defined: `main.py:81`

### real_time_packet_capture (function) `def real_time_packet_capture(model)`
- Defined: `main.py:85`
