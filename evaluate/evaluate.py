#!/usr/bin/env python3
import os, re, csv, math, argparse, subprocess, datetime

def parse_results(output):
    w = int(re.search(r"AZ Wins:\s*(\d+)", output).group(1))
    l = int(re.search(r"Minimax Wins:\s*(\d+)", output).group(1))
    d = int(re.search(r"Draws:\s*(\d+)", output).group(1))
    return w, l, d

def compute_elo(w, l, d, baseline=1500.0):
    n = w + l + d
    s = w + 0.5 * d

    if w == n:
        return math.inf
    elif l == n:
        return -math.inf

    rating_diff = 400.0 * math.log10(s / (n - s))
    return baseline + rating_diff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("eval_binary")
    parser.add_argument("--baseline", type=float, default=1500.0)
    parser.add_argument("-o", "--output", default="elo_results.csv")
    args = parser.parse_args()

    with open(args.output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["step", "elo", "created"])

        for fname in sorted(os.listdir(args.checkpoint_dir)):
            path = os.path.join(args.checkpoint_dir, fname)
            if not os.path.isfile(path): continue

            m = re.search(r"(\d+)", fname)
            step = m.group(1) if m else fname

            created = datetime.datetime.fromtimestamp(
                os.path.getctime(path)
            ).isoformat()

            proc = subprocess.run(
                [args.eval_binary, path],
                capture_output=True, text=True
            )
            if proc.returncode != 0:
                print(f"Error evaluating {fname}: {proc.stderr}")
                continue

            try:
                w, l, d = parse_results(proc.stdout)
                elo = compute_elo(w, l, d, args.baseline)

                elo_str = str(elo) if math.isinf(elo) else f"{elo:.2f}"
                writer.writerow([step, elo_str, created])

                if elo is not None:
                    print(f"Processed {fname}: Elo={elo:.1f}")
                else:
                    print(f"Processed {fname}: Elo=— (all wins or all losses)")
            except Exception as e:
                print(f"Failed to parse or compute Elo for {fname}: {e}")

if __name__ == "__main__":
    main()

