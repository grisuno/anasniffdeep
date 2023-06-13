# anasniffdeep
deep learning 
# Network Packet Capture and Classification

This project captures network packets, extracts features from them, and classifies them as either positive or negative based on the presence of certain characteristics. It uses the Scapy library for packet capture and manipulation, SQLite for data storage, and TensorFlow-Keras for training and deploying the classification model.

## Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python 3.x
- Scapy library
- SQLite3
- pandas
- numpy
- scikit-learn
- TensorFlow-Keras

You can install the required Python libraries using pip:

pip install scapy sqlite3 pandas numpy scikit-learn tensorflow

bash
Copy code

## Usage

1. Clone the repository or download the source code files.

2. Open a terminal or command prompt and navigate to the project directory.

3. Run the following command to capture and classify network packets:

python packet_capture.py

markdown
Copy code

The program will start capturing network packets and classifying them in real-time. It will display the predicted label for each captured packet and store positive packets in the SQLite database.

4. Press `Ctrl+C` to stop the packet capture process.

5. You can check the stored positive packets in the `data.db` SQLite database.

## File Structure

The project contains the following files:

- `packet_capture.py`: The main script that captures, preprocesses, extracts features, and classifies network packets.
- `data.db`: SQLite database file to store positive packets.
- `README.md`: This README file.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

Feel free to use, modify, and distribute this code for educational and personal projects.

## Acknowledgments

- The project uses the Scapy library (https://scapy.net/) for network packet manipulation.
- It also utilizes the TensorFlow-Keras library (https://www.tensorflow.org/guide/keras) for training and deploying the classification model.

If you have any questions or feedback, please feel free to contact me.

Please note that the README assumes the name of the main script is packet_capture.py and the database file is data.db. If the actual filenames are different, you'll need to adjust the instructions accordingly.
