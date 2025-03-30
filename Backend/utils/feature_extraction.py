import asyncio
import pyshark
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import os

# Choose the correct Wireshark display filter based on user input
def get_tls_filter(version_choice):
    if version_choice == "1":
        return "tls.record.version == 0x0303"  # TLSv1.2
    elif version_choice == "2":
        return "tls.record.version == 0x0304"  # TLSv1.3
    return "(tls.record.version == 0x0303 or tls.record.version == 0x0304)"

# Main function to extract flows and simulate attack traffic
def process_pcap_and_simulate(pcap_path, save_csv_path, tls_version="3", mode="auto", custom_features=None, record_limit=None):
    # Fix for PyShark: create an event loop inside thread
    asyncio.set_event_loop(asyncio.new_event_loop())

    print(f"[+] Loading PCAP: {pcap_path} | Mode: {mode} | TLS: {tls_version}")
    display_filter = get_tls_filter(tls_version)

    # Read .pcap file and filter based on TLS version
    cap = pyshark.FileCapture(pcap_path, display_filter=display_filter, only_summaries=False)
    flows = defaultdict(list)

    # Extract flow key (5-tuple): src-dst-sport-dport-protocol
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

    # Group packets into flows
    for pkt in cap:
        key = extract_flow_key(pkt)
        if key:
            flows[key].append(pkt)

    print(f"[+] Extracted {len(flows)} TLS flows from PCAP.")

    # Extract features: duration, source port, total forward length
    real_flow_data = []
    for key, packets in flows.items():
        try:
            byte_count = sum(int(pkt.length) for pkt in packets if hasattr(pkt, 'length'))
            duration = float(packets[-1].sniff_timestamp) - float(packets[0].sniff_timestamp)
            src_ip, dst_ip, sport, dport, proto = key.split('-')
            real_flow_data.append({
                'Flow Duration': round(duration, 6),
                'Source Port': int(sport) if sport else 0,
                'Total Length of Fwd Packets': byte_count
            })
        except:
            continue

    # Optionally limit the number of flows to simulate
    if record_limit and record_limit < len(real_flow_data):
        real_flow_data = real_flow_data[:record_limit]

    real_df = pd.DataFrame(real_flow_data)

    # Use user-defined features if mode is 'custom'
    if mode == "custom":
        final_df = pd.DataFrame([custom_features] * len(real_df))
        print(f"[+] Using user-defined custom feature values.")
    else:
        # Load pre-encoded DDoS attack vectors
        base_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(base_dir, '..', 'outputs')
        onehot_path = os.path.join(outputs_dir, 'TLS_OneHotEncoded.csv')

        if not os.path.exists(onehot_path):
            raise FileNotFoundError(f"Missing file: {onehot_path}")

        onehot_df = pd.read_csv(onehot_path)

        # Select only attack samples
        ddos_mask = (onehot_df['Label_0'] == 1.0) | (onehot_df['Label_1'] == 1.0) | (onehot_df['Label_2'] == 1.0)
        ddos_samples = onehot_df[ddos_mask].drop(
            columns=['Label_0', 'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Timestamp'],
            errors='ignore'
        ).reset_index(drop=True)

        # For each flow, embed attack sample and keep flow-based fields
        final_rows = []
        for i in range(len(real_df)):
            attack_row = ddos_samples.sample(n=1, random_state=random.randint(0, 10000)).copy().reset_index(drop=True)
            attack_row.loc[0, 'Flow Duration'] = real_df.iloc[i]['Flow Duration']
            attack_row.loc[0, 'Source Port'] = real_df.iloc[i]['Source Port']
            attack_row.loc[0, 'Total Length of Fwd Packets'] = real_df.iloc[i]['Total Length of Fwd Packets']
            final_rows.append(attack_row)

        final_df = pd.concat(final_rows, ignore_index=True)
        print(f"[+] Simulated {len(final_df)} DDoS flows using automated features.")

    # Save the final simulated dataset
    final_df.to_csv(save_csv_path, index=False)
    print(f"[+] Saved simulation output to {save_csv_path}")