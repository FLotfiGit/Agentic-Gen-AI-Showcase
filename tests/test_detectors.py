from agents.anomaly.detectors import ZScoreDetector, IQRDetector, StreamingThresholdDetector


def test_zscore_detector_basic():
    det = ZScoreDetector(threshold=2.0)
    res = det.detect([0, 0, 0, 10, 0])
    assert 3 in res.indices
    assert len(res.indices) >= 1


def test_iqr_detector_basic():
    det = IQRDetector(k=1.5)
    res = det.detect([0, 0, 0, 10, 0])
    assert 3 in res.indices


def test_streaming_detector_window():
    det = StreamingThresholdDetector(window=5, threshold=2.0)
    res = det.detect([0, 0, 0, 10, 0, 0])
    assert 3 in res.indices
