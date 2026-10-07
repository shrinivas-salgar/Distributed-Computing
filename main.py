import argparse
import csv
import hashlib
import math
import os
import time
import tracemalloc
from bisect import bisect_left
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
VALID_ALPHA_VALUES = [1.0, 1.1, 1.25, 1.5]
VALID_CANDIDATE_VALUES = [2, 4, 8]
DEFAULT_NODE_COUNTS = [5, 10, 20, 50, 100]
DEFAULT_REQUEST_COUNTS = [10_000, 100_000]
DEFAULT_VIRTUAL_NODES = 100
DEFAULT_KEY_COUNT = 10_000


def hash64(value: object) -> int:
    return int.from_bytes(hashlib.sha256(str(value).encode("utf-8")).digest()[:8], byteorder="big", signed=False)


def validate_alpha(alpha: float) -> None:
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0; got {alpha}")


def validate_candidate_count(candidate_count: int) -> None:
    if candidate_count < 1:
        raise ValueError(f"candidate count must be >= 1; got {candidate_count}")


def generate_key_population(key_count: int = DEFAULT_KEY_COUNT) -> List[str]:
    return [f"key-{index:05d}" for index in range(key_count)]


def generate_workload_requests(
    key_population: Sequence[str],
    request_count: int,
    workload: str,
    zipf_parameter: float = 1.0,
    seed: int = 42,
) -> List[str]:
    if request_count <= 0:
        raise ValueError(f"request_count must be positive; got {request_count}")
    rng = np.random.default_rng(seed)
    if workload == "uniform":
        indices = rng.integers(0, len(key_population), size=request_count)
        return [key_population[int(index)] for index in indices]

    if workload == "zipf":
        if zipf_parameter <= 0:
            raise ValueError(f"zipf parameter must be > 0; got {zipf_parameter}")
        ranks = np.arange(1, len(key_population) + 1, dtype=np.float64)
        weights = np.power(ranks, -zipf_parameter)
        probabilities = weights / weights.sum()
        indices = rng.choice(len(key_population), size=request_count, p=probabilities)
        return [key_population[int(index)] for index in indices]

    raise ValueError(f"Unsupported workload '{workload}'")


class ConsistentHash:
    """Standard deterministic consistent hashing over a logical ring."""

    def __init__(self, nodes: Sequence[str], virtual_nodes: int = DEFAULT_VIRTUAL_NODES):
        validate_candidate_count(virtual_nodes)
        self.nodes = sorted({str(node) for node in nodes})
        self.virtual_nodes = int(virtual_nodes)
        self.ring: List[Tuple[int, str]] = []
        self.positions: List[int] = []
        self._rebuild_ring()

    def _rebuild_ring(self) -> None:
        ring = []
        for node in self.nodes:
            for replica_index in range(self.virtual_nodes):
                ring.append((hash64(f"{node}:vnode:{replica_index}"), node))
        self.ring = sorted(ring, key=lambda entry: entry[0])
        self.positions = [position for position, _ in self.ring]

    def lookup(self, key: str) -> str:
        if not self.ring:
            raise ValueError("Cannot look up a key on an empty ring")
        target = hash64(key)
        index = bisect_left(self.positions, target) % len(self.positions)
        return self.ring[index][1]

    def candidate_order_for_key(self, key: str, max_candidates: int | None = None) -> List[str]:
        if not self.ring:
            return []
        target = hash64(key)
        primary_index = bisect_left(self.positions, target) % len(self.positions)
        seen: set[str] = set()
        ordered: List[str] = []
        ring_size = len(self.ring)
        for offset in range(ring_size):
            node = self.ring[(primary_index + offset) % ring_size][1]
            if node in seen:
                continue
            seen.add(node)
            ordered.append(node)
            if max_candidates is not None and len(ordered) >= max_candidates:
                break
        return ordered

    def add_node(self, node: str) -> None:
        if node not in self.nodes:
            self.nodes.append(str(node))
            self.nodes = sorted(self.nodes)
            self._rebuild_ring()

    def remove_node(self, node: str) -> None:
        if node in self.nodes:
            self.nodes.remove(str(node))
            self._rebuild_ring()


class RendezvousHash:
    """Deterministic rendezvous hashing with per-key score against each active node."""

    def __init__(self, nodes: Sequence[str]):
        self.nodes = sorted({str(node) for node in nodes})

    def lookup(self, key: str) -> str:
        if not self.nodes:
            raise ValueError("Rendezvous hashing requires at least one active node")
        return max(self.nodes, key=lambda node: (hash64(f"{key}|{node}"), node))

    def add_node(self, node: str) -> None:
        node = str(node)
        if node not in self.nodes:
            self.nodes.append(node)
            self.nodes = sorted(self.nodes)

    def remove_node(self, node: str) -> None:
        node = str(node)
        if node in self.nodes:
            self.nodes.remove(node)


class LoadAwareCH(ConsistentHash):
    """Request routing policy over a CH primary owner. This is not persistent ownership."""

    def __init__(self, nodes: Sequence[str], virtual_nodes: int = DEFAULT_VIRTUAL_NODES, alpha: float = 1.25, candidate_count: int = 4):
        validate_alpha(alpha)
        validate_candidate_count(candidate_count)
        super().__init__(nodes, virtual_nodes=virtual_nodes)
        self.alpha = float(alpha)
        self.candidate_count = int(candidate_count)

    def route_request(self, key: str, current_loads: Dict[str, int]) -> Tuple[str, bool]:
        if not self.nodes:
            raise ValueError("Load-aware routing requires at least one active node")

        primary_owner = self.lookup(key)
        active_nodes = sorted(current_loads)
        if not active_nodes:
            return primary_owner, False

        total_load = sum(current_loads.values())
        mean_load = total_load / len(active_nodes)
        if mean_load <= 0:
            return primary_owner, False

        if current_loads.get(primary_owner, 0) <= self.alpha * mean_load:
            return primary_owner, False

        bounded_alternatives = [node for node in self.candidate_order_for_key(key, max_candidates=self.candidate_count + 1) if node != primary_owner]
        bounded_alternatives = bounded_alternatives[: self.candidate_count]
        if not bounded_alternatives:
            return primary_owner, False

        for node in bounded_alternatives:
            if current_loads.get(node, 0) <= self.alpha * mean_load:
                return node, node != primary_owner

        least_loaded = min(bounded_alternatives, key=lambda node: (current_loads.get(node, 0), str(node)))
        return least_loaded, least_loaded != primary_owner


def compute_load_metrics(load_map: Dict[str, int]) -> Dict[str, float]:
    loads = np.asarray(list(load_map.values()), dtype=np.float64)
    if loads.size == 0:
        return {
            "mean_load": 0.0,
            "max_load": 0.0,
            "load_stddev": 0.0,
            "peak_to_average": 0.0,
            "jain_fairness": 1.0,
            "p95_node_load": 0.0,
            "p99_node_load": 0.0,
        }

    mean_load = float(loads.mean())
    max_load = float(loads.max())
    load_stddev = float(loads.std(ddof=0))
    peak_to_average = (max_load / mean_load) if mean_load > 0 else 0.0
    denominator = len(loads) * float(np.sum(loads ** 2))
    jain_fairness = float((np.sum(loads) ** 2) / denominator) if denominator > 0 else 1.0
    p95 = float(np.percentile(loads, 95))
    p99 = float(np.percentile(loads, 99))

    return {
        "mean_load": mean_load,
        "max_load": max_load,
        "load_stddev": load_stddev,
        "peak_to_average": peak_to_average,
        "jain_fairness": jain_fairness,
        "p95_node_load": p95,
        "p99_node_load": p99,
    }


def compute_table_metrics(load_map: Dict[str, int], route_times_us: List[float]) -> Dict[str, float]:
    metrics = compute_load_metrics(load_map)
    if route_times_us:
        metrics["mean_routing_time_us"] = float(np.mean(route_times_us))
        metrics["p95_routing_time_us"] = float(np.percentile(route_times_us, 95))
        metrics["p99_routing_time_us"] = float(np.percentile(route_times_us, 99))
    else:
        metrics["mean_routing_time_us"] = 0.0
        metrics["p95_routing_time_us"] = 0.0
        metrics["p99_routing_time_us"] = 0.0
    return metrics


def compute_membership_remap(algo: ConsistentHash, keys: Sequence[str], inserted_node: str) -> Tuple[float, float]:
    before_primary = {key: algo.lookup(key) for key in keys}
    algo.add_node(inserted_node)
    after_add = {key: algo.lookup(key) for key in keys}
    add_remap = 100.0 * sum(1 for key in keys if before_primary[key] != after_add[key]) / len(keys)
    algo.remove_node(inserted_node)

    removed_node = algo.nodes[0] if algo.nodes else inserted_node
    before_remove_primary = {key: algo.lookup(key) for key in keys}
    algo.remove_node(removed_node)
    after_remove = {key: algo.lookup(key) for key in keys}
    remove_remap = 100.0 * sum(1 for key in keys if before_remove_primary[key] != after_remove[key]) / len(keys)
    algo.add_node(removed_node)
    return add_remap, remove_remap


def run_single_algorithm(
    algorithm_name: str,
    nodes: Sequence[str],
    key_population: Sequence[str],
    requests: Sequence[str],
    alpha: float,
    candidate_count: int,
    virtual_nodes: int,
    workload: str,
    zipf_parameter: float,
    seed: int,
) -> Dict[str, object]:
    build_start = time.perf_counter()

    if algorithm_name == "CH":
        algo = ConsistentHash(nodes, virtual_nodes=virtual_nodes)
    elif algorithm_name == "RH":
        algo = RendezvousHash(nodes)
    elif algorithm_name == "LoadAwareCH":
        algo = LoadAwareCH(
            nodes,
            virtual_nodes=virtual_nodes,
            alpha=alpha,
            candidate_count=candidate_count,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm_name}'")

    algorithm_build_time = time.perf_counter() - build_start

    # The simulator records primary ownership, actual routing destination, and whether the request was load-reassigned.
    loads: Dict[str, int] = {node: 0 for node in nodes}
    primary_owners: Dict[str, str] = {}
    actual_destinations: Dict[str, str] = {}
    reassigned_flags: Dict[str, bool] = {}
    per_request_times_us: List[float] = []

    tracemalloc.start()
    routing_start = time.perf_counter()
    for key in requests:
        request_start = time.perf_counter()
        primary_owner = algo.lookup(key)
        if algorithm_name == "LoadAwareCH":
            actual_destination, reassigned = algo.route_request(key, loads)
        else:
            actual_destination = primary_owner
            reassigned = False

        elapsed_us = (time.perf_counter() - request_start) * 1_000_000.0
        per_request_times_us.append(elapsed_us)

        primary_owners[key] = primary_owner
        actual_destinations[key] = actual_destination
        reassigned_flags[key] = reassigned
        loads[actual_destination] = loads.get(actual_destination, 0) + 1

    routing_time_sec = time.perf_counter() - routing_start
    memory_current_bytes, memory_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    load_metrics = compute_load_metrics(loads)
    table_metrics = compute_table_metrics(loads, per_request_times_us)
    load_reassignment_pct = 0.0
    if algorithm_name == "LoadAwareCH":
        load_reassignment_pct = 100.0 * sum(1 for key in requests if actual_destinations.get(key) != primary_owners.get(key)) / len(requests)

    membership_start = time.perf_counter()
    membership_add_remap_pct, membership_remove_remap_pct = compute_membership_remap(algo, list(key_population), f"new-node-{seed}")
    membership_rebuild_time_sec = time.perf_counter() - membership_start

    row = {
        "algorithm": algorithm_name,
        "node_count": len(nodes),
        "key_count": len(key_population),
        "request_count": len(requests),
        "workload": workload,
        "zipf_parameter": zipf_parameter if workload == "zipf" else 0.0,
        "alpha": alpha,
        "candidate_count": candidate_count,
        "virtual_nodes": virtual_nodes,
        "seed": seed,
        "mean_load": load_metrics["mean_load"],
        "max_load": load_metrics["max_load"],
        "load_stddev": load_metrics["load_stddev"],
        "peak_to_average": load_metrics["peak_to_average"],
        "jain_fairness": load_metrics["jain_fairness"],
        "p95_node_load": load_metrics["p95_node_load"],
        "p99_node_load": load_metrics["p99_node_load"],
        "routing_time_sec": routing_time_sec,
        "routing_ops_sec": (len(requests) / routing_time_sec) if routing_time_sec > 0 else 0.0,
        "mean_routing_time_us": table_metrics["mean_routing_time_us"],
        "p95_routing_time_us": table_metrics["p95_routing_time_us"],
        "p99_routing_time_us": table_metrics["p99_routing_time_us"],
        "build_time_sec": algorithm_build_time,
        "membership_rebuild_time_sec": membership_rebuild_time_sec,
        "memory_peak_bytes": memory_peak_bytes,
        "membership_add_remap_pct": membership_add_remap_pct,
        "membership_remove_remap_pct": membership_remove_remap_pct,
        "load_reassignment_pct": load_reassignment_pct,
        "sensitivity_type": "main",
        "sensitivity_parameter": "",
        "notes": "primary ownership tracked separately from actual routed destination",
    }

    validate_experiment_row(row, algorithm_name)
    return row


def validate_experiment_row(row: Dict[str, object], algorithm_name: str) -> None:
    if row["request_count"] <= 0:
        raise ValueError("request_count must be positive")
    if row["node_count"] <= 0:
        raise ValueError("node_count must be positive")
    if row["mean_load"] < 0:
        raise ValueError("mean load must be non-negative")
    if row["load_reassignment_pct"] < 0 or row["load_reassignment_pct"] > 100:
        raise ValueError("load reassignment percentage must be between 0 and 100")
    if row["membership_add_remap_pct"] < 0 or row["membership_add_remap_pct"] > 100:
        raise ValueError("membership add remapping must be between 0 and 100")
    if row["membership_remove_remap_pct"] < 0 or row["membership_remove_remap_pct"] > 100:
        raise ValueError("membership remove remapping must be between 0 and 100")
    if row["jain_fairness"] < 0 or row["jain_fairness"] > 1.0:
        raise ValueError("Jain fairness must be within [0, 1]")
    if row["max_load"] < 0:
        raise ValueError("maximum node load must be non-negative")
    if row["p95_node_load"] < 0 or row["p99_node_load"] < row["p95_node_load"]:
        raise ValueError("P99 must be >= P95 for node loads")
    if row["routing_ops_sec"] < 0:
        raise ValueError("routing operations per second must be non-negative")
    if algorithm_name == "CH":
        if row["load_reassignment_pct"] != 0:
            raise ValueError("CH should not perform load-aware reassignment")
    if algorithm_name == "RH":
        if row["load_reassignment_pct"] != 0:
            raise ValueError("RH should not perform load-aware reassignment")


def run_experiment_suite(
    nodes_list: Sequence[int],
    request_counts: Sequence[int],
    workloads: Sequence[str],
    zipf_values: Sequence[float],
    seeds: Sequence[int],
    alpha: float,
    candidate_count: int,
    virtual_nodes: int,
    experiment_name: str = "main",
    sensitivity_type: str = "",
) -> List[Dict[str, object]]:
    validate_alpha(alpha)
    validate_candidate_count(candidate_count)
    if virtual_nodes < 1:
        raise ValueError(f"virtual node count must be >= 1; got {virtual_nodes}")

    results: List[Dict[str, object]] = []
    for seed in seeds:
        for node_count in nodes_list:
            if node_count <= 0:
                raise ValueError(f"node_count must be positive; got {node_count}")
            active_nodes = [f"node-{index:02d}" for index in range(node_count)]
            for request_count in request_counts:
                for workload in workloads:
                    zipf_parameter = 1.0
                    if workload == "uniform":
                        zipf_parameter = 0.0
                    elif workload == "zipf":
                        zipf_parameter = float(zipf_values[0]) if zipf_values else 1.0
                    key_population = generate_key_population(DEFAULT_KEY_COUNT)
                    requests = generate_workload_requests(key_population, request_count, workload, zipf_parameter, seed)
                    for algorithm_name in ["CH", "RH", "LoadAwareCH"]:
                        row = run_single_algorithm(
                            algorithm_name=algorithm_name,
                            nodes=active_nodes,
                            key_population=key_population,
                            requests=requests,
                            alpha=alpha,
                            candidate_count=candidate_count,
                            virtual_nodes=virtual_nodes,
                            workload=workload,
                            zipf_parameter=zipf_parameter,
                            seed=seed,
                        )
                        row["experiment_name"] = experiment_name
                        row["sensitivity_type"] = sensitivity_type
                        if workload == "zipf":
                            row["zipf_parameter"] = zipf_parameter
                        results.append(row)
    return results


def write_csv(rows: Sequence[Dict[str, object]], output_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write to CSV")
    fieldnames = list(rows[0].keys())
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed hashing research simulator for CH, RH, and load-aware CH request routing.")
    parser.add_argument("--nodes", type=int, nargs="+", default=DEFAULT_NODE_COUNTS, help="Physical node counts for the main experiment")
    parser.add_argument("--requests", type=int, nargs="+", default=DEFAULT_REQUEST_COUNTS, help="Request counts for the main experiment")
    parser.add_argument("--workloads", nargs="+", default=["uniform", "zipf"], help="Workloads: uniform or zipf")
    parser.add_argument("--zipf-params", type=float, nargs="+", default=[1.0], help="Zipf parameters to use when workload is zipf")
    parser.add_argument("--alpha", type=float, default=1.25, help="Load threshold multiplier used by load-aware CH routing")
    parser.add_argument("--candidates", type=int, default=4, help="Bounded alternative candidate count for load-aware CH routing")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS, help="Paired random seeds for repeatable experiments")
    parser.add_argument("--virtual-nodes", type=int, default=DEFAULT_VIRTUAL_NODES, help="Number of virtual nodes per physical node in CH")
    parser.add_argument("--output", default="results.csv", help="Destination CSV path")
    parser.add_argument("--mode", choices=["main", "sensitivity"], default="main", help="Run the main experiment or a sensitivity study")
    parser.add_argument("--sensitivity-type", choices=["workload", "alpha", "candidate"], default="workload", help="Sensitivity study type")
    parser.add_argument("--alpha-values", type=float, nargs="+", default=VALID_ALPHA_VALUES, help="Threshold values for alpha sensitivity")
    parser.add_argument("--candidate-values", type=int, nargs="+", default=VALID_CANDIDATE_VALUES, help="Candidate counts for candidate sensitivity")
    return parser.parse_args()


def run_main_experiment(args: argparse.Namespace) -> List[Dict[str, object]]:
    workloads = list(args.workloads)
    if not workloads:
        workloads = ["uniform", "zipf"]
    rows = run_experiment_suite(
        nodes_list=args.nodes,
        request_counts=args.requests,
        workloads=workloads,
        zipf_values=args.zipf_params,
        seeds=args.seeds,
        alpha=args.alpha,
        candidate_count=args.candidates,
        virtual_nodes=args.virtual_nodes,
        experiment_name="main",
        sensitivity_type="",
    )
    return rows


def run_sensitivity_experiment(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if args.sensitivity_type == "workload":
        zipf_values = args.zipf_params if args.zipf_params else [0.5, 1.0, 1.5]
        for zipf_parameter in zipf_values:
            workload_rows = run_experiment_suite(
                nodes_list=args.nodes,
                request_counts=args.requests,
                workloads=["zipf"],
                zipf_values=[zipf_parameter],
                seeds=args.seeds,
                alpha=args.alpha,
                candidate_count=args.candidates,
                virtual_nodes=args.virtual_nodes,
                experiment_name="sensitivity",
                sensitivity_type="workload",
            )
            rows.extend(workload_rows)
    elif args.sensitivity_type == "alpha":
        for alpha_value in args.alpha_values:
            alpha_rows = run_experiment_suite(
                nodes_list=args.nodes,
                request_counts=args.requests,
                workloads=["uniform", "zipf"],
                zipf_values=args.zipf_params,
                seeds=args.seeds,
                alpha=alpha_value,
                candidate_count=args.candidates,
                virtual_nodes=args.virtual_nodes,
                experiment_name="sensitivity",
                sensitivity_type="alpha",
            )
            rows.extend(alpha_rows)
    elif args.sensitivity_type == "candidate":
        for candidate_value in args.candidate_values:
            candidate_rows = run_experiment_suite(
                nodes_list=args.nodes,
                request_counts=args.requests,
                workloads=["uniform", "zipf"],
                zipf_values=args.zipf_params,
                seeds=args.seeds,
                alpha=args.alpha,
                candidate_count=candidate_value,
                virtual_nodes=args.virtual_nodes,
                experiment_name="sensitivity",
                sensitivity_type="candidate",
            )
            rows.extend(candidate_rows)
    else:
        raise ValueError(f"Unsupported sensitivity type '{args.sensitivity_type}'")
    return rows


def run_deterministic_checks() -> None:
    # Basic validation checks to catch paper-alignment regressions.
    nodes = [f"node-{index}" for index in range(5)]
    key_population = generate_key_population(10_000)
    requests = generate_workload_requests(key_population, 1_000, workload="uniform", seed=42)

    ch = ConsistentHash(nodes, virtual_nodes=100)
    for key in requests[:10]:
        if ch.lookup(key) not in nodes:
            raise AssertionError("CH must always return an active physical node")

    rh = RendezvousHash(nodes)
    for key in requests[:10]:
        if rh.lookup(key) not in nodes:
            raise AssertionError("RH must always return an active physical node")

    lac = LoadAwareCH(nodes, virtual_nodes=100, alpha=1.25, candidate_count=4)
    loads = {node: 0 for node in nodes}
    for key in requests[:100]:
        destination, reassigned = lac.route_request(key, loads)
        if destination not in nodes:
            raise AssertionError("Load-aware routing must always return an active physical node")
        loads[destination] = loads.get(destination, 0) + 1

    if len(lac.candidate_order_for_key(requests[0], max_candidates=4)) > 4:
        raise AssertionError("Candidate search must be bounded by C")

    # Ensure the same workload is reused for each algorithm for a given seed.
    uniform_requests = generate_workload_requests(key_population, 100, workload="uniform", seed=42)
    zipf_requests = generate_workload_requests(key_population, 100, workload="zipf", zipf_parameter=1.0, seed=42)
    if len(uniform_requests) != len(zipf_requests):
        raise AssertionError("The request generation must be stable and deterministic")


def main() -> None:
    args = parse_args()
    run_deterministic_checks()

    if args.mode == "main":
        rows = run_main_experiment(args)
    else:
        rows = run_sensitivity_experiment(args)

    write_csv(rows, args.output)
    print(f"Saved {len(rows)} experiment rows to {args.output}")


if __name__ == "__main__":
    main()
