import pyshark
import pandas as pd
from collections import defaultdict

# Load the PCAP file
pcap_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\real_traffic_test.pcap"
cap = pyshark.FileCapture(pcap_path, only_summaries=False)

# Data structure to hold flows
flows = defaultdict(list)

# Feature extractor
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

# Parse packets into flows
for pkt in cap:
    key = extract_flow_key(pkt)
    if key:
        flows[key].append(pkt)

# Extract simple features
flow_data = []

for key, packets in flows.items():
    pkt_count = len(packets)
    byte_count = sum(int(pkt.length) for pkt in packets if hasattr(pkt, 'length'))
    duration = float(packets[-1].sniff_timestamp) - float(packets[0].sniff_timestamp)
    duration = round(duration, 6)
    avg_pkt_size = byte_count / pkt_count if pkt_count > 0 else 0
    src_ip, dst_ip, sport, dport, proto = key.split('-')

    flow_data.append({
        'SrcIP': src_ip,
        'DstIP': dst_ip,
        'SrcPort': sport,
        'DstPort': dport,
        'Protocol': proto,
        'FlowDuration': duration,
        'TotPkts': pkt_count,
        'TotBytes': byte_count,
        'AvgPktSize': avg_pkt_size
    })

# Convert to DataFrame and save
df = pd.DataFrame(flow_data)
df.to_csv('E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\pyshark_extracted_flows.csv', index=False)
print("Flow features saved to pyshark_extracted_flows.csv")