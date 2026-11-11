import argparse
import statistics
import time
from pathlib import Path

import requests


def benchmark(url: str, image_path: Path, runs: int, warmup: int):
    image_bytes = image_path.read_bytes()
    latencies = []

    def send_request():
        start = time.perf_counter()

        response = requests.post(
            url,
            files={"file": (image_path.name, image_bytes, "image/jpeg")},
            params={"top_k": 5},
            timeout=120,
        )
        response.raise_for_status()

        return (time.perf_counter() - start) * 1000

    print(f"Warming up: {warmup} requests")
    for _ in range(warmup):
        send_request()

    print(f"Benchmarking: {runs} requests")
    for i in range(runs):
        latency = send_request()
        latencies.append(latency)
        print(f"Request {i + 1}/{runs}: {latency:.2f} ms")

    sorted_latencies = sorted(latencies)

    def percentile(p):
        index = (len(sorted_latencies) - 1) * p / 100
        lower = int(index)
        upper = min(lower + 1, len(sorted_latencies) - 1)
        fraction = index - lower

        return (
            sorted_latencies[lower] * (1 - fraction)
            + sorted_latencies[upper] * fraction
        )

    total_seconds = sum(latencies) / 1000

    print("\n--- Benchmark Results ---")
    print(f"Runs: {runs}")
    print(f"Average: {statistics.mean(latencies):.2f} ms")
    print(f"P50: {percentile(50):.2f} ms")
    print(f"P95: {percentile(95):.2f} ms")
    print(f"Min: {min(latencies):.2f} ms")
    print(f"Max: {max(latencies):.2f} ms")
    print(f"Throughput: {runs / total_seconds:.2f} req/s (sequential)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/api/v1/analyze")
    parser.add_argument("--image", required=True)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("--runs must be >= 1 and --warmup must be >= 0")

    benchmark(args.url, Path(args.image), args.runs, args.warmup)