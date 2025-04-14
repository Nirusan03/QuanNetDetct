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
    return "(tls.record.version == 0x0303 or tls.record.version == 0x0304)" # Both TLS version

# Main function to extract flows and simulate attack traffic
def process_pcap_and_simulate(pcap_path, save_csv_path, tls_version="3", mode="auto", custom_features=None, record_limit=None):
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"[ERROR] PCAP file not found: {pcap_path}")
    if mode not in ["auto", "custom"]:
        raise ValueError(f"[ERROR] Invalid mode '{mode}'. Must be 'auto' or 'custom'.")
    if tls_version not in ["1", "2", "3"]:
        raise ValueError(f"[ERROR] Invalid TLS version '{tls_version}'. Must be '1', '2', or '3'.")

    asyncio.set_event_loop(asyncio.new_event_loop())
    display_filter = get_tls_filter(tls_version)
    print(f"[+] Loading PCAP: {pcap_path} | Mode: {mode} | TLS: {tls_version}")

    try:
        cap = pyshark.FileCapture(pcap_path, display_filter=display_filter, only_summaries=False)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to read PCAP with PyShark: {str(e)}")

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
        except Exception:
            return None

    try:
        for pkt in cap:
            key = extract_flow_key(pkt)
            if key:
                flows[key].append(pkt)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed during flow extraction: {str(e)}")

    if len(flows) == 0:
        raise ValueError("[ERROR] No valid TLS flows found in PCAP.")

    print(f"[+] Extracted {len(flows)} TLS flows from PCAP.")

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
        except Exception:
            continue

    if not real_flow_data:
        raise ValueError("[ERROR] No usable flow data extracted from PCAP.")

    if record_limit and isinstance(record_limit, int) and record_limit < len(real_flow_data):
        real_flow_data = real_flow_data[:record_limit]

    real_df = pd.DataFrame(real_flow_data)

    if mode == "custom":
        if not isinstance(custom_features, dict):
            raise TypeError("[ERROR] 'custom_features' must be a dictionary of key-value feature pairs.")
        try:
            final_df = pd.DataFrame([custom_features] * len(real_df))
            print(f"[+] Using user-defined custom feature values.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to apply custom features: {str(e)}")
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onehot_path = os.path.join(base_dir, '..', 'outputs', 'TLS_OneHotEncoded.csv')

        if not os.path.exists(onehot_path):
            raise FileNotFoundError(f"[ERROR] Required dataset not found: {onehot_path}")

        try:
            onehot_df = pd.read_csv(onehot_path)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load attack dataset: {str(e)}")

        ddos_mask = (onehot_df['Label_0'] == 1.0) | (onehot_df['Label_1'] == 1.0) | (onehot_df['Label_2'] == 1.0)
        ddos_samples = onehot_df[ddos_mask].drop(
            columns=['Label_0', 'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Timestamp'],
            errors='ignore'
        ).reset_index(drop=True)

        if ddos_samples.empty:
            raise ValueError("[ERROR] No DDoS attack samples found in TLS_OneHotEncoded.csv")

        final_rows = []
        for i in range(len(real_df)):
            attack_row = ddos_samples.sample(n=1, random_state=random.randint(0, 10000)).copy().reset_index(drop=True)
            attack_row.loc[0, 'Flow Duration'] = real_df.iloc[i]['Flow Duration']
            attack_row.loc[0, 'Source Port'] = real_df.iloc[i]['Source Port']
            attack_row.loc[0, 'Total Length of Fwd Packets'] = real_df.iloc[i]['Total Length of Fwd Packets']
            final_rows.append(attack_row)

        final_df = pd.concat(final_rows, ignore_index=True)
        print(f"[+] Simulated {len(final_df)} DDoS flows using automated features.")

    try:
        final_df.to_csv(save_csv_path, index=False)
        print(f"[+] Saved simulation output to {save_csv_path}")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to save CSV file: {str(e)}")
