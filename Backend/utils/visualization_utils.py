# utils/visualization_utils.py
import asyncio
import os
import pyshark
from collections import defaultdict

def summarize_pcap_for_visualization(pcap_path):
    """
    Parses .pcap file and returns structure for frontend visualisation:
    - Protocol usage
    - Port usage
    - All available flow summaries
    """
    asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        cap = pyshark.FileCapture(pcap_path, only_summaries=True)
        flows = []
        protocol_counts = defaultdict(int)
        src_ports = defaultdict(int)
        dst_ports = defaultdict(int)

        for pkt in cap:
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

        cap.close()  # Close pyshark handle explicitly

        summary = {
            "total_packets": len(flows),
            "protocol_distribution": dict(protocol_counts),
            "source_port_distribution": dict(src_ports),
            "destination_port_distribution": dict(dst_ports),
            "flows": flows
        }

        return summary

    except Exception as e:
        return {"error": f"Failed to process pcap: {str(e)}"}
