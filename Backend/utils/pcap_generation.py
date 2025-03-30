# utils/pcap_generation.py

from scapy.all import IP, TCP, Raw, wrpcap, rdpcap
from datetime import datetime, timedelta
import pandas as pd
import random
import os

def generate_pcap_from_csv(csv_path, output_pcap_path):
    print(f"[+] Generating PCAP from: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[!] Failed to read CSV: {e}")
        return 0

    packets = []
    start_time = datetime.now()

    for idx, row in df.iterrows():
        try:
            # Random IPs for anonymisation
            src_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
            dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"

            sport = int(row.get("Source Port", random.randint(1024, 65535)))
            dport = 443  # Common TLS port

            duration = float(row.get("Flow Duration", 0.01))
            total_len = int(row.get("Total Length of Fwd Packets", 100))

            payload = Raw(load='X' * min(total_len, 1400))  # Respect Ethernet MTU
            pkt = IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="S")/payload
            pkt.time = (start_time + timedelta(seconds=duration * idx)).timestamp()
            packets.append(pkt)

        except Exception as e:
            print(f"[!] Skipping packet {idx}: {e}")
            continue

    wrpcap(output_pcap_path, packets)
    print(f"[+] Saved {len(packets)} packets to: {output_pcap_path}")
    return len(packets)


def validate_pcap(pcap_path, limit=100):
    print(f"[+] Validating PCAP: {pcap_path}")
    packets = []

    try:
        read_packets = rdpcap(pcap_path)
    except Exception as e:
        return [{"error": f"Failed to read PCAP: {e}"}]

    for i, pkt in enumerate(read_packets[:limit]):
        try:
            if IP in pkt and TCP in pkt:
                packets.append({
                    "index": i + 1,
                    "src_ip": pkt[IP].src,
                    "dst_ip": pkt[IP].dst,
                    "sport": pkt[TCP].sport,
                    "dport": pkt[TCP].dport,
                    "size": len(pkt),
                    "flags": str(pkt[TCP].flags)
                })
            else:
                packets.append({
                    "index": i + 1,
                    "info": "Non-IP/TCP Packet",
                    "size": len(pkt)
                })
        except Exception as e:
            packets.append({
                "index": i + 1,
                "info": f"Error reading packet: {e}"
            })

    return packets
