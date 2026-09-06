# API

## main.py

### convert_ip_to_octets `def convert_ip_to_octets(ip)`
- Defined: `main.py:15`

### preprocess_packet `def preprocess_packet(packet)`
- Defined: `main.py:18`

### capture_packet `def capture_packet()`
- Defined: `main.py:23`

### extract_features `def extract_features(packet)`
- Defined: `main.py:34`

### create_database `def create_database()`
- Defined: `main.py:54`

### store_positive_packet `def store_positive_packet(packet)`
- Defined: `main.py:62`

### train_model `def train_model(X_train, y_train, X_val, y_val)`
- Defined: `main.py:73`

### evaluate_model `def evaluate_model(model, X_val, y_val)`
- Defined: `main.py:81`

### real_time_packet_capture `def real_time_packet_capture(model)`
- Defined: `main.py:85`
