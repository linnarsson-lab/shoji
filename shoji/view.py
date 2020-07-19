"""
Views of tensors, defined by filters.
"""
from typing import Tuple
import shoji
import numpy as np


class View:
	def __init__(self, wsm: shoji.WorkspaceManager, filters: Tuple[shoji.Filter, ...]) -> None:
		super().__setattr__("filters", {f.dim: f for f in filters})
		super().__setattr__("wsm", wsm)
	
	def __getattr__(self, name: str) -> np.ndarray:
		# Get the tensor
		tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"
		# if tensor.rank == 0:
		# 	codec = shoji.Codec(tensor.dtype)
		# 	# Read the value directly, since you can't filter a scalar
		# 	key = self.wsm._subspace.pack(("tensor_values", name, 0, 0))
		# 	val = self.wsm._db.transaction[key]
		# 	return codec.decode(val)

		indices = None
		if tensor.rank > 0 and tensor.dims[0] in self.filters:
			indices = np.sort(self.filters[tensor.dims[0]].get_rows(self.wsm))
		# Read the tensor (all or selected rows)
		result = shoji.io.read_tensor_values(self.wsm._db.transaction, self.wsm, name, tensor, indices)
		# Filter the remaining dimensions
		for i, dim in enumerate(tensor.dims):
			if i == 0:
				continue
			if isinstance(dim, str) and dim in self.filters:
				# Filter this dimension
				indices = self.filters[dim].get_rows(self.wsm)
				result = result.take(indices, axis=i)
		return result

	def __getitem__(self, name: str) -> np.ndarray:
		return self.__getattr__(name)

	def __setattr__(self, name: str, vals: np.ndarray) -> None:
		tensor: shoji.Tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"
		indices = []
		for dim in tensor.dims:
			if dim in self.filters:
				indices.append(np.sort(self.filters[tensor.dims[0]].get_rows(self.wsm)))
			else:
				indices.append(slice(None))
		tensor.inits = vals
		shoji.io.write_tensor_values(self.wsm._db.transaction, self.wsm, name, tensor, indices)

	def __setitem__(self, name: str, vals: np.ndarray) -> None:
		return self.__setattr__(name, vals)
