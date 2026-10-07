import bisect
import hashlib
import random
import time
import csv
import argparse
from collections import Counter


def hash64(value):
    return int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], 'big')


class ConsistentHash:
    def __init__(self, nodes, replicas=100):
        self.replicas = replicas
        self.nodes = list(nodes)
        self.ring = []
        self.positions = []
        self._build()

    def _build(self):
        self.ring = sorted((hash64(f'{node}:replica:{i}'), node)
                           for node in self.nodes for i in range(self.replicas))
        self.positions = [position for position, _ in self.ring]

    def lookup(self, key):
        i = bisect.bisect_left(self.positions, hash64(key)) % len(self.ring)
        return self.ring[i][1]

    def add_node(self, node):
        self.nodes.append(node)
        self._build()

    def remove_node(self, node):
        self.nodes.remove(node)
        self._build()


class RendezvousHash:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def lookup(self, key):
        return max(self.nodes, key=lambda node: hash64(f'{key}|{node}'))

    def add_node(self, node):
        self.nodes.append(node)

    def remove_node(self, node):
        self.nodes.remove(node)


class LoadAwareHash(ConsistentHash):
    """Request routing policy; not stable ownership for persistent data."""
    def __init__(self, nodes, replicas=100, slack=1.10):
        super().__init__(nodes, replicas)
        self.slack = slack

    def route_requests(self, requests):
        loads = Counter({node: 0 for node in self.nodes})
        assignments = []
        target = len(requests) / len(self.nodes)
        capacity = max(1, int(target * self.slack + 0.999999))
        for key in requests:
            preferred = self.lookup(key)
            if loads[preferred] < capacity:
                chosen = preferred
            else:
                # Deterministic fallback order, subject to current load.
                alternatives = sorted(self.nodes, key=lambda n: hash64(f'{key}|{n}'), reverse=True)
                chosen = next((n for n in alternatives if loads[n] < capacity),
                              min(self.nodes, key=lambda n: (loads[n], n)))
            loads[chosen] += 1
            assignments.append(chosen)
        return assignments, loads


def make_requests(count, skewed=False, seed=42):
    rng = random.Random(seed)
    if skewed:
        return [f'key-{rng.randrange(100) if rng.random() < 0.8 else rng.randrange(count)}'
                for _ in range(count)]
    return [f'key-{i}' for i in range(count)]


def metrics(loads, seconds, total):
    values = list(loads.values())
    avg = sum(values) / len(values)
    std = (sum((x - avg) ** 2 for x in values) / len(values)) ** 0.5
    return dict(lookup_us=seconds * 1e6 / total,
                throughput=total / seconds if seconds else 0,
                max_load=max(values), load_stddev=std,
                imbalance=max(values) / avg)


def experiment(node_count, request_count, skewed=False):
    nodes = [f'node-{i}' for i in range(node_count)]
    requests = make_requests(request_count, skewed)
    unique_keys = sorted(set(requests))
    rows = []
    for name, cls in [('consistent', ConsistentHash),
                      ('rendezvous', RendezvousHash),
                      ('load_aware', LoadAwareHash)]:
        algo = cls(nodes)
        start = time.perf_counter()
        if name == 'load_aware':
            assigned, loads = algo.route_requests(requests)
        else:
            assigned = [algo.lookup(key) for key in requests]
            loads = Counter({node: 0 for node in nodes})
            loads.update(assigned)
        elapsed = time.perf_counter() - start
        # Key remapping is measured for the underlying stable placement.
        # For load-aware request routing, this is NOT a remapping measure
        # of the dynamic routed request assignments.
        before = {key: algo.lookup(key) for key in unique_keys}
        algo.add_node('new-node')
        after = {key: algo.lookup(key) for key in unique_keys}
        add_moved = sum(before[key] != after[key] for key in unique_keys)
        algo.remove_node('new-node')
        algo.remove_node(nodes[0])
        after_remove = {key: algo.lookup(key) for key in unique_keys}
        remove_moved = sum(before[key] != after_remove[key] for key in unique_keys)
        row = dict(method=name, nodes=node_count, requests=request_count,
                   workload='skewed' if skewed else 'uniform',
                   unique_keys=len(unique_keys),
                   add_remap_pct=100 * add_moved / len(unique_keys),
                   remove_remap_pct=100 * remove_moved / len(unique_keys),
                   **metrics(loads, elapsed, len(requests)))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--requests', type=int, nargs='+', default=[10000])
    parser.add_argument('--output', default='results.csv')
    args = parser.parse_args()
    rows = []
    for count in args.nodes:
        if count < 2:
            parser.error('nodes must be at least 2')
        for nreq in args.requests:
            if nreq < 1:
                parser.error('requests must be positive')
            for skewed in [False, True]:
                result = experiment(count, nreq, skewed)
                rows.extend(result)
                for row in result:
                    print(row)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
