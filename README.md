# Load Balance, Key Remapping, and Lookup Overhead in Distributed Hashing Under Dynamic Workloads

This simulator implements the paper-aligned comparison between:

- Consistent Hashing (CH)
- Rendezvous Hashing (RH)
- Bounded-candidate load-aware request routing over Consistent Hashing (LoadAwareCH)

## Research objective

The goal is to compare deterministic CH ownership, RH ownership, and a bounded load-aware routing policy that reassigns only the request destination when the CH primary owner is overloaded. The implementation separates:

- primary ownership: the deterministic CH owner of a key
- actual routing destination: the node that receives the request in the simulation
- membership remapping: primary-owner changes caused by node membership changes
- load reassignment: request redirection caused by the load-aware policy

## Algorithms implemented

### Consistent Hashing

- Logical ring over physical nodes
- Physical nodes mapped to a ring using virtual nodes
- Default virtual node count is 100 per physical node
- Deterministic hashing for ring placement
- Lookup uses binary search over sorted ring positions with approximately O(log V) complexity where V is the number of virtual-node positions
- The CH primary owner is always recorded and remains the deterministic ownership reference

### Rendezvous Hashing

- Each key receives a deterministic score for every active physical node
- The node with the highest score is selected
- Straightforward lookup complexity is O(N)
- Uses deterministic hashing and consistent key encoding

### Load-aware CH request routing

The load-aware method is evaluated as a request-routing policy over Consistent Hashing, not as a persistent key-ownership or data-migration mechanism.

The routing logic follows the paper specification:

1. Compute the CH primary owner.
2. Compute the current mean node load: L_bar = (sum of all node loads) / N.
3. A node is overloaded when L_i > alpha * L_bar.
4. If the CH primary owner is not overloaded, route to the CH primary.
5. If the CH primary owner is overloaded, evaluate only up to C alternative CH candidates.
6. Select the first candidate whose load satisfies L_i <= alpha * L_bar.
7. If none satisfy the threshold, select the least-loaded candidate among the bounded set.
8. Use deterministic tie-breaking.
9. Record whether actual_destination != CH_primary_owner.

This is load-induced request reassignment and is separate from membership-induced primary ownership remapping.

## Default experimental parameters

- alpha = 1.25
- candidate count C = 4
- virtual nodes per physical node = 100
- keys = 10,000
- requests = 10,000 and 100,000
- workloads = Uniform and Zipf(1.0)
- node counts = 5, 10, 20, 50, 100
- seeds = 42, 43, 44, 45, 46, 47, 48, 49, 50, 51

## Workloads

- Uniform
- Zipf(1.0)
- Sensitivity: Zipf(0.5), Zipf(1.0), Zipf(1.5)

The key population and request sequence are generated once per seed and reused for CH, RH, and LoadAwareCH so that the paired experimental design remains valid.

## Dynamic membership

The simulator evaluates both:

- add one node
- remove one node

For each experiment:

1. Record primary ownership before the membership change.
2. Add or remove exactly one node.
3. Recompute primary ownership.
4. Compare ownership before versus after.

Membership remapping is computed as:

R_membership = (number of keys whose PRIMARY OWNER changed / total keys) * 100

This metric is based only on the deterministic CH primary owner and does not count load-aware request redirection.

## Load-induced reassignment

For the load-aware policy, the simulator computes:

R_load = (number of requests where actual destination != CH primary owner / total requests) * 100

This metric is recorded separately from membership remapping and is not treated as persistent key migration.

## Metrics collected

The CSV contains the following output fields:

- mean_load
- max_load
- load_stddev
- peak_to_average
- jain_fairness
- p95_node_load
- p99_node_load
- routing_time_sec
- routing_ops_sec
- p95_routing_time_us
- p99_routing_time_us
- build_time_sec
- membership_rebuild_time_sec
- memory_peak_bytes
- membership_add_remap_pct
- membership_remove_remap_pct
- load_reassignment_pct

The simulator measures algorithmic routing performance and does not represent network, RPC, storage, or cluster throughput.

## Sensitivity experiments

The code supports separate sensitivity runs for:

- workload sensitivity: Zipf values 0.5, 1.0, 1.5
- threshold sensitivity: alpha values 1.0, 1.1, 1.25, 1.5
- candidate-count sensitivity: C values 2, 4, 8

These are separate experiment runs; the CSV identifies the sensitivity type and parameter values so the results are not mixed with the main experiment.

## Command-line usage

### Main experiment

```bash
python main.py --nodes 5 10 20 50 100 --requests 10000 100000 --alpha 1.25 --candidates 4 --seeds 42 43 44 45 46 47 48 49 50 51 --output results.csv
```

### Sensitivity: workload

```bash
python main.py --mode sensitivity --sensitivity-type workload --nodes 5 10 --requests 10000 --zipf-params 0.5 1.0 1.5 --output sensitivity_workload.csv
```

### Sensitivity: alpha

```bash
python main.py --mode sensitivity --sensitivity-type alpha --nodes 5 10 --requests 10000 --alpha-values 1.0 1.1 1.25 1.5 --output sensitivity_alpha.csv
```

### Sensitivity: candidate count

```bash
python main.py --mode sensitivity --sensitivity-type candidate --nodes 5 10 --requests 10000 --candidate-values 2 4 8 --output sensitivity_candidates.csv
```

### Plotting

```bash
python plot_results.py results.csv --output-dir plots
```

## Limitations

- This is a simulator, not a networked deployment test.
- Memory usage is tracked with Python tracing and is intended for comparative measurement in the simulator, not real cluster resource accounting.
- The load-aware method is intentionally a request-routing policy and should not be treated as a persistent ownership mechanism or a generalized hashing replacement.
- The simulator measures routing decisions and does not model RPC latency, storage I/O, or cluster throughput.
