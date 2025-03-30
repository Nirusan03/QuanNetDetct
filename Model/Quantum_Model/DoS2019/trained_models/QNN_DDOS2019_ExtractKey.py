import os
import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# Prompt user for feature source
print("Choose feature mode for simulation:")
print("1 - Use default simulation with automated DDoS feature sampling")
print("2 - Use custom self-defined values for all flows")
feature_mode = input("Enter your choice (1 or 2): ").strip()

use_custom_features = feature_mode == "2"
custom_features = {}

if use_custom_features:
    print("\nEnter custom values to be used for all flows.")
    try:
        custom_features["Flow Duration"] = float(input("Flow Duration (e.g., 0.123): "))
        custom_features["Source Port"] = int(input("Source Port (e.g., 443): "))
        custom_features["Total Length of Fwd Packets"] = int(input("Total Length of Fwd Packets (e.g., 1234): "))
    except Exception as e:
        print(f"Invalid input. Aborting. Error: {e}")
        exit()

# Prompt for TLS version filter
print("\nSelect TLS version(s) to extract from PCAP:")
print("1 - TLSv1.2")
print("2 - TLSv1.3")
print("3 - Both TLSv1.2 and TLSv1.3")
tls_choice = input("Enter your choice (1/2/3): ").strip()

if tls_choice == "1":
    tls_filter = "tls.record.version == 0x0303" # TLSv1.2
elif tls_choice == "2":
    tls_filter = "tls.record.version == 0x0304" # TLSv1.3
elif tls_choice == "3":
    tls_filter = "(tls.record.version == 0x0303 or tls.record.version == 0x0304)"
else:
    print("Invalid input. Defaulting to both TLSv1.2 and TLSv1.3.")
    tls_filter = "(tls.record.version == 0x0303 or tls.record.version == 0x0304)"

# Load PCAP with TLS filter
pcap_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\real_traffic_test.pcap"
cap = pyshark.FileCapture(pcap_path, display_filter=tls_filter, only_summaries=False)
flows = defaultdict(list)

def extract_flow_key(pkt):
    try:
        ip_layer = pkt.ip
        proto = pkt.transport_layer
        if proto is None or proto.upper() != "TCP":
            return None
        src = ip_layer.src
        dst = ip_layer.dst
        sport = pkt[pkt.transport_layer].srcport
        dport = pkt[pkt.transport_layer].dstport
        return f"{src}-{dst}-{sport}-{dport}-{proto}"
    except:
        return None

for pkt in cap:
    key = extract_flow_key(pkt)
    if key:
        flows[key].append(pkt)

# Extract real flows from PCAP
real_flow_data = []
for key, packets in flows.items():
    try:
        pkt_count = len(packets)
        byte_count = sum(int(pkt.length) for pkt in packets if hasattr(pkt, 'length'))
        duration = float(packets[-1].sniff_timestamp) - float(packets[0].sniff_timestamp)
        duration = round(duration, 6)
        src_ip, dst_ip, sport, dport, proto = key.split('-')

        real_flow_data.append({
            'Flow Duration': duration,
            'Source Port': int(sport) if sport else 0,
            'Total Length of Fwd Packets': byte_count
        })
    except:
        continue

real_df = pd.DataFrame(real_flow_data)
print(f"\nCaptured {len(real_df)} TLS flows from PCAP.")

# Load attack features only if using automated mode
if not use_custom_features:
    full_attack_df = pd.read_csv("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\sample_all_attacks_test_data.csv")
    onehot_df = pd.read_csv("E:\\TLS_OneHotEncoded.csv")

    ddos_mask = (onehot_df['Label_0'] == 1.0) | (onehot_df['Label_1'] == 1.0) | (onehot_df['Label_2'] == 1.0)
    ddos_samples = onehot_df[ddos_mask].drop(columns=['Label_0','Label_1','Label_2','Label_3','Label_4','Timestamp'], errors='ignore').reset_index(drop=True)

# Generate simulation output
final_rows = []

for i in range(len(real_df)):
    if use_custom_features:
        # If custom, repeat the same user-defined row
        attack_row = pd.DataFrame([custom_features])
    else:
        # Sample attack features and embed real values
        attack_row = ddos_samples.sample(n=1, random_state=random.randint(0, 10000)).copy().reset_index(drop=True)
        attack_row.loc[0, 'Flow Duration'] = real_df.iloc[i]['Flow Duration']
        attack_row.loc[0, 'Source Port'] = real_df.iloc[i]['Source Port']
        attack_row.loc[0, 'Total Length of Fwd Packets'] = real_df.iloc[i]['Total Length of Fwd Packets']

    final_rows.append(attack_row)

# Save final output
if final_rows:
    final_df = pd.concat(final_rows, ignore_index=True)
    save_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\Model_Input_1.csv"
    final_df.to_csv(save_path, index=False)
    print(f"\nSimulation Completed: {len(final_df)} flows saved using mode = {'Custom Features' if use_custom_features else 'Automated DDoS Sampling'}")
    print(f"Saved as: {save_path}")
else:
    print("\nNo valid flows available to simulate. Please check your PCAP file or filter settings.")
