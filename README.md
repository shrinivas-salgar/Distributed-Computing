# Distributed hashing comparison — M.Tech starter project

Implements Consistent Hashing (100 virtual nodes per server), Rendezvous Hashing and a simple load-aware *request routing* policy on top of Consistent Hashing. Runs repeatable synthetic uniform and skewed request experiments, measures routing time, throughput, per-node request-load imbalance and stable key remapping after adding/removing a server.

## Run

Python 3.9+ is sufficient for the simulator:

```bash
python main.py --nodes 5 10 20 50 100 --requests 10000 100000
```

For plots, install `pandas` and `matplotlib`, then run:

```bash
pip install pandas matplotlib
python plot_results.py results.csv
```

## Important research limitations

- The load-aware variant balances **request routing**, not persistent key ownership. The same key can route to different nodes at different times, so a real cache/storage implementation would need replication, forwarding or a separate ownership directory. Do not claim that its request routing achieves consistent key placement.
- Remapping percentages for load-aware rows measure its **underlying consistent hash ring**, not dynamic routed requests. Thus its remapping metric is not an apples-to-apples measure of dynamic routing churn.
- Lookup microseconds include Python simulation overhead, and load-aware time includes load-counter maintenance. Throughput is simulated routing decisions per second, not real network throughput.
- Uniform workload uses unique keys. Skewed workload sends 80% of requests to a hot set of 100 keys.
- Memory consumption, multi-run averages and actual network effects are not implemented yet. Extend the experiment before making claims on those metrics.
- The proposed policy is a student implementation concept, not an established research novelty. Compare with published bounded-load consistent hashing before claiming originality.
