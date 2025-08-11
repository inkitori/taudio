from typing import Dict


class Metrics:
    def __init__(self):
        self.metrics = {}
        self.scale_factor = 1

    def add(self, key: str):
        self.metrics[key] = 0

    def update(self, key: str, value: float):
        """
        Accumulate a value into a metric. If the metric key does not
        yet exist, create it and set it to the provided value.
        """
        if key not in self.metrics:
            self.metrics[key] = float(value)
            return
        self.metrics[key] += float(value)

    def set_scale_factor(self, factor: int):
        self.scale_factor = factor

    def get(self, key: str):
        return self.metrics[key]

    def get_scaled(self, key: str):
        return self.metrics[key] / self.scale_factor

    def reset(self):
        self.metrics.clear()


class AverageMetrics:
    """
    Maintains running averages for arbitrary numeric metrics.

    Usage:
    - update(key, value): adds a new observation for the metric keyed by `key`.
    - get(key): returns the running average for the key.
    - get_count(key): returns the number of observations seen for the key.
    - get_sum(key): returns the sum of observations for the key.
    - reset(): clears all tracked metrics.
    """

    def __init__(self):
        self._sum = {}
        self._count = {}

    def update(self, key: str, value: float):
        """Add a new sample value for the given metric key."""
        if key not in self._sum:
            self._sum[key] = float(value)
            self._count[key] = 1
            return
        self._sum[key] += float(value)
        self._count[key] += 1

    def update_dict(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            self.update(key, value)

    def get(self, key: str) -> float:
        """Return the running average for the given key."""
        count = self._count.get(key, 0)
        if count == 0:
            return 0.0
        return self._sum[key] / count

    def get_count(self, key: str) -> int:
        return int(self._count.get(key, 0))

    def get_sum(self, key: str) -> float:
        return float(self._sum.get(key, 0.0))

    def reset(self):
        self._sum.clear()
        self._count.clear()

    def to_dict(self):
        return {key: self.get(key) for key in self._sum.keys()}
