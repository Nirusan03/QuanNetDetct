import asyncio
import os
import pyshark
from collections import defaultdict

def summarize_pcap_for_visualization(pcap_path):
    if not os.path.exists(pcap_path):
        return {"error": f"[ERROR] PCAP file does not exist at: {pcap_path}"}

    try:
        asyncio.set_event_loop(asyncio.new_event_loop())
        cap = pyshark.FileCapture(pcap_path, only_summaries=True)
    except Exception as e:
        return {"error": f"[ERROR] Failed to load PCAP with PyShark: {e}"}

    flows = []
    protocol_counts = defaultdict(int)
    src_ports = defaultdict(int)
    dst_ports = defaultdict(int)

    try:
        for pkt in cap:
            try:
                flows.append({
                    'Time': pkt.time,
                    'Protocol': pkt.protocol,
                    'Source': pkt.source,
                    'Destination': pkt.destination,
                    'Length': pkt.length,
                    'Info': pkt.info
                })

                protocol_counts[pkt.protocol] += 1

                if "→" in pkt.info:
                    parts = pkt.info.split("→")
                    sport = parts[0].strip().split()[-1]
                    dport = parts[1].strip().split()[0]
                    src_ports[sport] += 1
                    dst_ports[dport] += 1

            except Exception as pkt_err:
                flows.append({
                    'Info': f"[WARNING] Skipped malformed summary line: {pkt_err}"
                })
                continue

        cap.close()  # Explicitly release resources

        if not flows:
            return {"error": "[ERROR] No summary packets found in PCAP."}

        summary = {
            "total_packets": len(flows),
            "protocol_distribution": dict(protocol_counts),
            "source_port_distribution": dict(src_ports),
            "destination_port_distribution": dict(dst_ports),
            "flows": flows
        }

        return summary

    except Exception as parse_error:
        return {"error": f"[ERROR] Failed to parse packets: {parse_error}"}
