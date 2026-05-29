import os
import argparse
import json
from pathlib import Path
from services.data_simulator.generators.marketplace import MarketplaceSimulator

def main():
    parser = argparse.ArgumentParser(description="Nexus Synthetic Marketplace Event Generator")
    parser.add_argument("command", choices=["generate"])
    parser.add_argument("--n-events", type=int, default=100000)
    parser.add_argument("--n-users", type=int, default=1000)
    parser.add_argument("--n-items", type=int, default=500)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--to-parquet", action="store_true")

    args = parser.parse_args()

    if args.command == "generate":
        sim = MarketplaceSimulator(n_users=args.n_users, n_items=args.n_items)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        events = sim.generate_batch(args.n_events)
        
        if args.to_parquet:
            try:
                import pandas as pd
                df = pd.DataFrame(events)
                out_path = out_dir / "historical_interactions.parquet"
                df.to_parquet(out_path, index=False)
                print(f"[+] Output written to Parquet format: {out_path}")
            except ImportError:
                print("[!] Pandas or PyArrow missing! Falling back to JSON...")
                write_json_lines(events, out_dir)
        else:
            write_json_lines(events, out_dir)

def write_json_lines(events, directory):
    out_path = directory / "historical_interactions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    print(f"[+] Output written to JSON: {out_path}")

if __name__ == "__main__":
    main()
