from agents.anomaly.agent import AnomalyAgent, AnomalyAgentConfig
from agents.anomaly.detectors import ZScoreDetector, IQRDetector, StreamingThresholdDetector
from agents.anomaly.datasets import generate_spikes, generate_drift
from agents.anomaly.evaluation import precision_recall_f1
from agents.anomaly.visualization import format_detection_report, compare_detectors, save_results_json


def ensemble_detection():
    """Run multiple detectors and compare results."""
    print("\n=== Ensemble Anomaly Detection ===\n")

    # Generate test data
    series, labels = generate_spikes(n=200, spike_rate=0.05)

    # Create detectors
    detectors = {
        "ZScore": ZScoreDetector(threshold=2.5),
        "IQR": IQRDetector(k=1.5),
        "Streaming": StreamingThresholdDetector(window=10, threshold=2.5),
    }

    # Run detections
    results = {}
    for name, detector in detectors.items():
        res = detector.detect(series)
        results[name] = res.indices
        metrics = precision_recall_f1(res.indices, labels)
        print(f"{name}: F1={metrics['f1']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}")

    # Compare
    print(compare_detectors(series, results))

    # Save ensemble results
    save_results_json(series, list(set().union(*results.values())), "outputs/anomaly_ensemble.json", {"detectors": list(detectors.keys())})


def drift_detection_demo():
    """Detect anomalies in drifting time-series."""
    print("\n=== Drift Detection Demo ===\n")

    series, labels = generate_drift(n=250, drift=0.01)

    agent = AnomalyAgent(
        detector=ZScoreDetector(threshold=2.0),
        config=AnomalyAgentConfig(window=15),
    )
    res = agent.run(series)

    print(format_detection_report(series, res["indices"], "ZScore Detector"))
    metrics = precision_recall_f1(res["indices"], labels)
    print(f"\nMetrics: F1={metrics['f1']:.3f}")

    save_results_json(series, res["indices"], "outputs/anomaly_drift.json", {"method": "zscore", "window": 15})


def detector_comparison():
    """Show sensitivity comparison across detectors."""
    print("\n=== Detector Sensitivity Analysis ===\n")

    series, _ = generate_spikes(n=300, spike_rate=0.03)

    configs = [
        ("ZScore (τ=2.0)", ZScoreDetector(threshold=2.0)),
        ("ZScore (τ=2.5)", ZScoreDetector(threshold=2.5)),
        ("ZScore (τ=3.0)", ZScoreDetector(threshold=3.0)),
        ("IQR (k=1.0)", IQRDetector(k=1.0)),
        ("IQR (k=1.5)", IQRDetector(k=1.5)),
    ]

    print(f"{'Detector':<20} {'Anomalies':>12} {'Rate':>8}")
    print("-" * 42)
    for name, det in configs:
        res = det.detect(series)
        rate = len(res.indices) / len(series) * 100
        print(f"{name:<20} {len(res.indices):>12} {rate:>7.1f}%")


def main():
    ensemble_detection()
    drift_detection_demo()
    detector_comparison()


if __name__ == "__main__":
    main()
