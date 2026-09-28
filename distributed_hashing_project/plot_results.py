import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
df = pd.read_csv(path)
for workload in df.workload.unique():
    subset = df[(df.workload == workload) & (df.requests == df.requests.min())]
    for metric in ['imbalance', 'lookup_us', 'add_remap_pct']:
        pivot = subset.pivot(index='nodes', columns='method', values=metric)
        ax = pivot.plot(marker='o', title=f'{metric} ({workload})')
        ax.set_ylabel(metric)
        ax.figure.tight_layout()
        ax.figure.savefig(f'{metric}_{workload}.png', dpi=160)
        plt.close(ax.figure)
