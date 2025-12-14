from __future__ import annotations

from typing import List, Dict, Any, Tuple
import json


def format_detection_report(
    series: List[float],
    indices: List[int],
    detector_name: str = "Detector",
    include_context: int = 2,
) -> str:
    """Format detection results as readable report."""
    lines = [f"\n=== {detector_name} Anomaly Report ===\n"]
    lines.append(f"Total points: {len(series)}")
    lines.append(f"Anomalies detected: {len(indices)}\n")

    if indices:
        lines.append("Anomalies:")
        for idx in sorted(indices):
            start = max(0, idx - include_context)
            end = min(len(series), idx + include_context + 1)
            context = series[start:end]
            lines.append(
                f"  Index {idx}: value={series[idx]:.2f} "
                f"(context: {[f'{v:.2f}' for v in context]})"
            )
    else:
        lines.append("No anomalies detected.")

    return "\n".join(lines)


def summary_stats(
    series: List[float],
    indices: List[int],
) -> Dict[str, Any]:
    """Compute summary statistics for series and anomalies."""
    if not series:
        return {}

    values = [series[i] for i in indices] if indices else []
    normal_values = [series[i] for i in range(len(series)) if i not in indices]

    return {
        "total_points": len(series),
        "anomaly_count": len(indices),
        "anomaly_rate": len(indices) / len(series) if series else 0.0,
        "series_mean": sum(series) / len(series),
        "series_std": (sum((x - sum(series) / len(series)) ** 2 for x in series) / max(1, len(series) - 1)) ** 0.5,
        "anomaly_mean": sum(values) / len(values) if values else None,
        "anomaly_min": min(values) if values else None,
        "anomaly_max": max(values) if values else None,
        "normal_mean": sum(normal_values) / len(normal_values) if normal_values else None,
    }


def compare_detectors(
    series: List[float],
    detector_results: Dict[str, List[int]],
) -> str:
    """Compare multiple detectors' results."""
    lines = ["\n=== Detector Comparison ===\n"]

    all_detected = set()
    for indices in detector_results.values():
        all_detected.update(indices)

    for det_name, indices in detector_results.items():
        stats = summary_stats(series, indices)
        lines.append(f"{det_name}:")
        lines.append(f"  Anomalies: {len(indices)} ({stats['anomaly_rate']:.1%})")
        if indices:
            lines.append(f"  Range: [{stats['anomaly_min']:.2f}, {stats['anomaly_max']:.2f}]")

    if all_detected:
        consensus = set(detector_results[list(detector_results.keys())[0]]) if detector_results else set()
        for indices in list(detector_results.values())[1:]:
            consensus &= set(indices)
        lines.append(f"\nConsensus anomalies (detected by all): {len(consensus)}")

    return "\n".join(lines)


def save_results_json(
    series: List[float],
    indices: List[int],
    output_path: str,
    metadata: Dict[str, Any] = None,
) -> None:
    """Save detection results to JSON."""
    result = {
        "series": series,
        "anomaly_indices": indices,
        "stats": summary_stats(series, indices),
        "metadata": metadata or {},
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
