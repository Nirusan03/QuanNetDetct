from scapy.all import *
import pandas as pd
import random
from datetime import datetime, timedelta

# === Load the model input CSV ===
csv_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\Model_Input.csv"
df = pd.read_csv(csv_path)

# === Prepare packet list ===
packets = []
start_time = datetime.now()

# === Generate synthetic TCP flows ===
for idx, row in df.iterrows():
    try:
        # Random synthetic IPs
        src_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
        dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"

        # Ports and flow details
        sport = int(row.get("Source Port", random.randint(1024, 65535)))
        dport = 443  # Assume TLS port

        duration = float(row.get("Flow Duration", 0.01))  # Avoid zero
        total_len = int(row.get("Total Length of Fwd Packets", 100))

        # Create packet with payload
        payload = Raw(load='X' * min(total_len, 1400))  # Max Ethernet MTU
        pkt = IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="S")/payload
        pkt.time = (start_time + timedelta(seconds=duration * idx)).timestamp()
        packets.append(pkt)
    except Exception as e:
        print(f"[Warning] Skipped packet {idx}: {e}")

# === Save to PCAP ===
pcap_output_path = "E:\\Studies\\IIT\\4 - Forth Year\\Final Year Project\\QuanNetDetct\\Model\\Quantum_Model\\DoS2019\\trained_models\\SimulatedDDoSOutput.pcap"
wrpcap(pcap_output_path, packets)
print(f"\nPCAP saved to: {pcap_output_path} with {len(packets)} packets.")

# === Validate the generated PCAP ===
print("\nValidating PCAP Contents...\n")
read_packets = rdpcap(pcap_output_path)

for i, p in enumerate(read_packets[:10]):  # Show first 10 for inspection
    if IP in p and TCP in p:
        print(f"Packet {i+1}: {p[IP].src}:{p[TCP].sport} -> {p[IP].dst}:{p[TCP].dport}, Size: {len(p)} bytes, Flags: {p[TCP].flags}")
    else:
        print(f"Packet {i+1}: Non-IP/TCP Packet")

print("\nPCAP validation complete.")
