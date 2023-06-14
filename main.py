import socket
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from scapy.all import sniff, RadioTap, Raw
from scapy.layers.dot11 import Dot11
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

def convert_ip_to_octets(ip):
    return list(map(int, socket.inet_aton(ip)))

def preprocess_packet(packet):
    dot11 = RadioTap() / Dot11() / Raw(packet)
    ethernet = Ether() / dot11
    return ethernet

def capture_packet():
    try:
        packet = sniff(count=1)
        if packet:
            return packet[0]
        else:
            return None
    except Exception as e:
        print(f'An error occurred while capturing packets: {e}')
        return None

def extract_features(packet):
    features = []
    packet.show()
    ip_layer = packet.getlayer(IP)
    if ip_layer:
        source_ip = ip_layer.src
        destination_ip = ip_layer.dst
        packet_length = len(packet)
        packet_protocol = ip_layer.proto

        source_ip_octets = convert_ip_to_octets(source_ip)
        destination_ip_octets = convert_ip_to_octets(destination_ip)

        features.extend(source_ip_octets)
        features.extend(destination_ip_octets)
        features.append(packet_length)
        features.append(packet_protocol)

    return features

def create_database():
    conn = sqlite3.connect('data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS positive_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        packet_data BLOB)''')
    conn.close()

def store_positive_packet(packet):
    try:
        conn = sqlite3.connect('data.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO positive_data (packet_data) VALUES (?)", (str(packet),))
        conn.commit()
        conn.close()
        print("Positive data saved in the database.")
    except Exception as e:
        print(f'An error occurred while storing positive packet: {e}')

def train_model(X_train, y_train, X_val, y_val):
    num_features = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=num_features))
    model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])
    history = model.fit(X_train.values, y_train, epochs=10, batch_size=32, validation_data=(X_val.values, y_val))
    return model, history

def evaluate_model(model, X_val, y_val):
    loss, accuracy = model.evaluate(X_val.values, y_val)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

def real_time_packet_capture(model):
    while True:
        captured_packet = capture_packet()

        if captured_packet:
            try:
                preprocessed_packet = preprocess_packet(captured_packet)
                packet_features = extract_features(preprocessed_packet)

                if packet_features:
                    packet_data = pd.DataFrame([packet_features], columns=X_encoded.columns)
                    predicted_label = model.predict(packet_data.values)
                    print(f'Predicted label: {predicted_label}')

                    if predicted_label[0][0] >= 0.5:
                        store_positive_packet(captured_packet)
                else:
                    print('Could not extract features from the packet.')
            except Exception as e:
                print(f'An error occurred while processing packets: {e}')
        else:
            print('No packet captured.')
            break

# Create the database if it doesn't exist
create_database()
print("Database created successfully.")

# Collect and prepare data
packets = sniff(filter="ip and tcp", count=100)
data = []

for packet in packets:
    if 'login' in packet.payload:
        label = 1
    else:
        label = 0

    if packet.haslayer(IP) and packet.haslayer(TCP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        src_port = packet[TCP].sport
        dst_port = packet[TCP].dport
        protocol = packet[IP].proto
    else:
        continue

    src_ip_octets = convert_ip_to_octets(src_ip)
    dst_ip_octets = convert_ip_to_octets(dst_ip)

    features = src_ip_octets + dst_ip_octets + [src_port, dst_port, protocol]
    data.append((features, label))

print("Data collected and prepared successfully.")

# Prepare data for learning
data = pd.DataFrame(data, columns=['features', 'label'])
X = data['features']
y = data['label']

X = pd.Series(X)
X_encoded = X.apply(lambda x: convert_ip_to_octets(x) if isinstance(x, str) else x)

print("Features encoded:", X_encoded)

mlb = MultiLabelBinarizer()
X_encoded = pd.DataFrame(mlb.fit_transform(X_encoded), columns=mlb.classes_, index=X_encoded.index)

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the model
model, history = train_model(X_train, y_train, X_val, y_val)
evaluate_model(model, X_val, y_val)

model.summary()
print("Model trained successfully.")

# Real-time packet capture and classification
real_time_packet_capture(model)
