from typing import Dict, Union, Callable, Optional
import numpy as np
from sklearn.preprocessing import LabelEncoder
import shoji


# Based on https://github.com/mahmoud/lithoxyl/blob/master/lithoxyl/moment.py
# but adapted to use numpy arrays (element-wise) instead of single values
# and to support labeled groups of array-values
class Accumulator:
	def __init__(self):
		self._sum = None
		self._count = None
		self._nnz = None
		self._min = None
		self._max = None
		self._mean = None
		self._m2 = None
		self._m3 = None
		self._m4 = None
		self._first = None

	def add(self, x):
		if self._count is None:
			self._sum = x.astype("float64")
			self._count = np.ones_like(self._sum, dtype="uint64")
			self._nnz = np.zeros_like(self._sum, dtype="uint64")
			self._nnz[x > 0] += 1
			self._min = x.astype("float64")
			self._max = x.astype("float64")
			self._mean = x.astype("float64")
			self._m2 = np.zeros_like(self._mean)
			self._m3 = np.zeros_like(self._mean)
			self._m4 = np.zeros_like(self._mean)
			self._first = x
		else:
			self._nnz[x > 0] = np.add(self._nnz[x > 0], 1, casting="unsafe")  # This convoluted scheme is needed to avoid error when adding 1 to a scalar array
			self._sum += x
			self._min = np.minimum(self._min, x)
			self._max = np.maximum(self._min, x)
			np.add(self._count, 1, out=self._count, casting="unsafe")  # Same issue as above, but at least here we can do it in-place
			n = self._count
			delta = x - self._mean
			delta_n = delta / n
			delta_n2 = delta_n ** 2
			self._mean += delta_n
			term = delta * delta_n * (n - 1)
			self._m4 = (self._m4 + term * delta_n2 * (n ** 2 - 3 * n + 3) + 6 * delta_n2 * self._m2 - 4 * delta_n * self._m3)
			self._m3 = (self._m3 + term * delta_n * (n - 2) - 3 * delta_n * self._m2)
			self._m2 = self._m2 + term

	@property
	def first(self):
		return self._first

	@property
	def count(self):
		return self._count

	@property
	def sum(self):
		return self._sum

	@property
	def nnz(self):
		return self._nnz

	@property
	def mean(self):
		return self._mean

	@property
	def variance(self):
		return self._m2 / (self._count - 1)

	@property
	def skewness(self):
		return ((self._count ** 0.5) * self._m3) / (self._m2 ** 1.5)

	@property
	def kurtosis(self):
		# TODO: subtract 3? (for normal curve = 0)
		return (self._count * self._m4) / (self._m2 ** 2)

	@property
	def sd(self):
		return self.variance ** 0.5


class GroupAccumulator:
	def __init__(self) -> None:
		self.groups: Dict[int, Accumulator] = {}

	def add(self, label, x) -> None:
		self.groups.setdefault(label, Accumulator()).add(x)

	def sum(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.sum for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].sum

	def nnz(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.nnz for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].nnz

	def count(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.count for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].count

	def first(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.first for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].first

	def mean(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.mean for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].mean

	def variance(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.variance for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].variance

	def skewness(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.skewness for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].skewness

	def kurtosis(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.kurtosis for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].kurtosis

	def sd(self, label = None) -> np.ndarray:
		if label is None:
			x = {label: x.sd for label, x in self.groups.items()}
			return np.array(list(x.keys())), np.array(list(x.values()))
		return self.groups[label].sd


class GroupViewBy:
	def __init__(self, view: "shoji.view.View", labels: Optional[Union[str, np.ndarray]], projection: Callable = None) -> None:
		self.view = view
		self.labels = labels
		if isinstance(self.labels, str):
			tensor = view.wsm._get_tensor(self.labels)
			if tensor.rank != 1:
				raise ValueError(f"Cannot groupby('{self.labels}'); a rank-1 tensor is required")
		self.projection = projection
		self.acc: Optional[GroupAccumulator] = None

	def stats(self, of_tensor: str) -> GroupAccumulator:
		if self.acc is not None:
			return self.acc
		tensor = self.view.wsm._get_tensor(of_tensor)
		n_rows = self.view.get_length(tensor.dims[0])
		if self.labels is None:
			label_values = np.zeros(n_rows)
		elif isinstance(self.labels, np.ndarray):
			label_values = self.labels
		else:
			label_values = self.view[self.labels]
		if self.projection is not None:
			label_values = [self.projection(x) for x in  label_values]
		le = LabelEncoder()
		labels = le.fit_transform(label_values)  # Encode string labels and non-contiguous integers into integers 0, 1, 2, ...
		acc = GroupAccumulator()

		n_rows_per_batch = 1000
		for ix in range(0, n_rows, n_rows_per_batch):
			chunk = self.view._read_chunk(tensor, ix, ix + n_rows_per_batch)
			chunk_labels = labels[ix: ix + n_rows_per_batch]
			for i, label in enumerate(chunk_labels):
				acc.add(le.classes_[label], chunk[i])
		return acc

	def sum(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).sum()

	def count(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).count()

	def first(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).first()

	def nnz(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).nnz()

	def mean(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).mean()

	def variance(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).variance()

	def skewness(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).skewness()

	def kurtosis(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).kurtosis()

	def sd(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).sd()


class GroupDimensionBy:
	def __init__(self, dim: "shoji.dimension.Dimension", labels: Optional[Union[str, np.ndarray]], projection: Callable = None, chunk_size: int = 1000) -> None:
		self.dim = dim
		assert dim.wsm is not None, "Cannot group by unbound dimension"
		self.labels = labels
		if isinstance(self.labels, str):
			tensor = dim.wsm._get_tensor(self.labels)
			if tensor.rank != 1:
				raise ValueError(f"Cannot groupby('{self.labels}'); a rank-1 tensor is required")
		self.chunk_size = chunk_size
		self.projection = projection
		self.acc: Optional[GroupAccumulator] = None

	def stats(self, of_tensor: str) -> GroupAccumulator:
		if self.acc is not None:
			return self.acc
		assert self.dim.wsm is not None
		le = LabelEncoder()
		if self.labels is None:
			label_values = np.zeros(self.dim.length)
		elif isinstance(self.labels, np.ndarray):
			label_values = self.labels
		else:
			label_values = self.dim.wsm[self.labels][:]
		if self.projection is not None:
			label_values = [self.projection(x) for x in  label_values]
		labels = le.fit_transform(label_values)  # Encode string labels and non-contiguous integers into integers 0, 1, 2, ...
		acc = GroupAccumulator()
		n_rows = self.dim.length
		n_rows_per_batch = 1000
		for ix in range(0, n_rows, n_rows_per_batch):
			batch = self.dim.wsm._get_tensor(of_tensor)[ix: ix + n_rows_per_batch]
			batch_labels = labels[ix: ix + n_rows_per_batch]
			for i, label in enumerate(batch_labels):
				acc.add(le.classes_[label], batch[i])
		return acc


	def sum(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).sum()

	def count(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).count()

	def nnz(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).nnz()

	def mean(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).mean()

	def variance(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).variance()

	def skewness(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).skewness()

	def kurtosis(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).kurtosis()

	def sd(self, of_tensor: str) -> np.ndarray:
		return self.stats(of_tensor).sd()
