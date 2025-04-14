from scapy.all import IP, TCP, Raw, wrpcap, rdpcap
from datetime import datetime, timedelta
import pandas as pd
import random
import os

def generate_pcap_from_csv(csv_path, output_pcap_path):
    print(f"[+] Generating PCAP from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV file does not exist: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to read CSV file: {e}")

    if df.empty:
        raise ValueError("[ERROR] The CSV file is empty. Cannot generate PCAP.")

    packets = []
    start_time = datetime.now()

    for idx, row in df.iterrows():
        try:
            src_ip = f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}"
            dst_ip = f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}"

            sport = int(row.get("Source Port", random.randint(1024, 65535)))
            dport = 443  # Simulated target port for TLS (HTTPS)

            duration = float(row.get("Flow Duration", 0.01))
            total_len = int(row.get("Total Length of Fwd Packets", 100))

            payload = Raw(load='X' * min(total_len, 1400))  # Respect MTU limit
            pkt = IP(src=src_ip, dst=dst_ip)/TCP(sport=sport, dport=dport, flags="S")/payload
            pkt.time = (start_time + timedelta(seconds=duration * idx)).timestamp()
            packets.append(pkt)

        except Exception as e:
            print(f"[!] Skipping row {idx} due to error: {e}")
            continue

    if not packets:
        raise RuntimeError("[ERROR] No packets were created. Check your CSV content.")

    try:
        wrpcap(output_pcap_path, packets)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Failed to write PCAP file: {e}")

    print(f"[+] Saved {len(packets)} packets to: {output_pcap_path}")
    return len(packets)


def validate_pcap(pcap_path, limit=100):
    print(f"[+] Validating PCAP: {pcap_path}")

    if not os.path.exists(pcap_path):
        return [{"error": f"[ERROR] PCAP file not found: {pcap_path}"}]

    packets = []
    try:
        read_packets = rdpcap(pcap_path)
    except Exception as e:
        return [{"error": f"[ERROR] Failed to read PCAP: {e}"}]

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
                "info": f"[ERROR] Failed to parse packet {i + 1}: {e}"
            })

    return packets
