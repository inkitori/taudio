class Metrics:
	def __init__(self):
		self.metrics = {}
		self.scale_factor = 1

	def add(self, key: str):
		self.metrics[key] = 0

	def update(self, key: str, value: float):
		self.metrics[key] += value

	def set_scale_factor(self, factor: int):
		self.scale_factor = factor
	
	def get(self, key: str):
		return self.metrics[key]

	def get_scaled(self, key: str):
		return self.metrics[key] / self.scale_factor

	def reset(self):
		for key in self.metrics.keys():
			self.metrics[key] = 0