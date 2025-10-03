import sys, pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src)
sample = df.sample(n, random_state=42)
sample.to_csv(dst, index=False)
print(f"Wrote {len(sample)} rows to {dst}")
