# from typing import Dict, Union, Callable, Any, List
# import numpy as np
# import shoji


# class GroupBy:
# 	def __init__(self, view: "shoji.View", tensor_name: str, projection: Callable = None) -> None:
# 		self.view = view
# 		self.groupby_name = tensor_name
# 		tensor = view.wsm._get_tensor(tensor_name)
# 		if tensor.rank != 1:
# 			raise ValueError(f"Cannot groupby() tensor '{tensor_name}' of rank {tensor.rank}; a rank-1 tensor is required")


# 	def mapreduce(self, f_map: Callable, f_reduce: Callable, tensor_name: str) -> np.ndarray:
# 		groupby_tensor = self.view.wsm._get_tensor(self.groupby_name)
# 		applyto_tensor = self.view.wsm._get_tensor(tensor_name)
# 		indices = self.view.get
# 		i = 0

# 		vals = self.view[self.groupby_name]
# 		self.groups: Dict[Any, List] = {}  # Dict of projected values to row indices
# 		self.length = len(vals)
# 		for i in range(vals.shape[0]):
# 			if projection is not None:
# 				self.groups.setdefault(projection(vals[i]), []).append(i)
# 			else:
# 				self.groups.setdefault(vals[i], []).append(i)

# 		vals = self.view[tensor_name]
# 		if len(vals) != self.length:
# 			raise ValueError("Tensors are not the same length in groupby() operation")
# 		result = np.empty(len(self.groups), dtype=vals.dtype)
# 		for i, k in enumerate(self.groups.keys()):
# 			result[i] = np.mean(vals[self.groups[k]])
# 		return result

# 	def map_reduce(self, tensor_name: str, n_rows_per_batch: int = 100):
# 		i = 0
# 		while True:
# 			vals = self.view[tensor]