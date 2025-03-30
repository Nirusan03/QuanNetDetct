import pyshark
import pandas as pd
import numpy as np
from collections import defaultdict
import random

# === STEP 1: Load the PCAP file ===
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

# === STEP 2: Extract Real Flow Features ===
real_flow_data = []

for key, packets in flows.items():
    try:
        pkt_count = len(packets)
        byte_count = sum(int(pkt.length) for pkt in packets if hasattr(pkt, 'length'))
        duration = float(packets[-1].sniff_timestamp) - float(packets[0].sniff_timestamp)
        duration = round(duration, 6)
        avg_pkt_size = byte_count / pkt_count if pkt_count > 0 else 0
        src_ip, dst_ip, sport, dport, proto = key.split('-')

        real_flow_data.append({
            'Flow Duration': duration,
            'Source Port': int(sport) if sport else 0,
            'Total Length of Fwd Packets': byte_count
        })
    except:
        continue

real_df = pd.DataFrame(real_flow_data)

# === STEP 3: Load One-Hot Encoded Attack Data for Sampling ===
onehot_file = "E:\\TLS_OneHotEncoded.csv"
onehot_df = pd.read_csv(onehot_file)

# Drop target and timestamp cols (if any)
onehot_df = onehot_df.drop(columns=[col for col in onehot_df.columns if 'Label_' in col or 'Timestamp' in col], errors='ignore')

# === STEP 4: Merge Real + Sampled Features ===
final_rows = []
for i in range(len(real_df)):
    sample_row = onehot_df.sample(n=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
    sample_row = sample_row.copy()

    sample_row.loc[0, 'Flow Duration'] = real_df.iloc[i]['Flow Duration']
    sample_row.loc[0, 'Source Port'] = real_df.iloc[i]['Source Port']
    sample_row.loc[0, 'Total Length of Fwd Packets'] = real_df.iloc[i]['Total Length of Fwd Packets']

    final_rows.append(sample_row)

final_df = pd.concat(final_rows, ignore_index=True)

# === STEP 5: Save Model-Ready CSV ===
output_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\model_ready_real_traffic.csv"
final_df.to_csv(output_path, index=False)

print("Model-compatible file saved to:")
print(f"     {output_path}")
