import os
import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# === Step 1: Load the PCAP file ===
pcap_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\real_traffic_test.pcap"
cap = pyshark.FileCapture(pcap_path, only_summaries=False)
flows = defaultdict(list)

def extract_flow_key(pkt):
    try:
        ip_layer = pkt.ip
        proto = pkt.transport_layer
        src = ip_layer.src
        dst = ip_layer.dst
        sport = pkt[pkt.transport_layer].srcport if proto else None
        dport = pkt[pkt.transport_layer].dstport if proto else None
        return f"{src}-{dst}-{sport}-{dport}-{proto}"
    except:
        return None

for pkt in cap:
    key = extract_flow_key(pkt)
    if key:
        flows[key].append(pkt)

# === Step 2: Extract Real Flow Features ===
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
print(f"\nCaptured {len(real_df)} real flows from PCAP.")

# === Step 3: Load Attack & OneHot Data ===
full_attack_df = pd.read_csv("E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\sample_all_attacks_test_data.csv")
onehot_df = pd.read_csv("E:\\TLS_OneHotEncoded.csv")

# === Step 4: Separate Only DDoS Samples ===
ddos_mask = (onehot_df['Label_0'] == 1.0) | (onehot_df['Label_1'] == 1.0) | (onehot_df['Label_2'] == 1.0)
ddos_samples = onehot_df[ddos_mask].drop(columns=['Label_0','Label_1','Label_2','Label_3','Label_4','Timestamp'], errors='ignore').reset_index(drop=True)

# === Step 5: Simulate All Real Flows as DDoS ===
final_rows = []

for i in range(len(real_df)):
    attack_row = ddos_samples.sample(n=1, random_state=random.randint(0, 10000)).copy().reset_index(drop=True)
    attack_row.loc[0, 'Flow Duration'] = real_df.iloc[i]['Flow Duration']
    attack_row.loc[0, 'Source Port'] = real_df.iloc[i]['Source Port']
    attack_row.loc[0, 'Total Length of Fwd Packets'] = real_df.iloc[i]['Total Length of Fwd Packets']
    final_rows.append(attack_row)

# === Step 6: Save Final CSV ===
final_df = pd.concat(final_rows, ignore_index=True)
save_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\Model_Input.csv"
final_df.to_csv(save_path, index=False)

print(f"\nDDoS Simulation Completed: {len(final_df)} flows simulated as DDoS.")
print(f"Saved as: {save_path}")
