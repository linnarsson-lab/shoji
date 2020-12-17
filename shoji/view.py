"""
Views let you work with a selected subset of a workspace. Reading from the view automatically
returns values from the selected subset of the database. Values written to the view
are automatically written to the corresponding subset of the database.

Views are *selections of elements* along one or more dimensions. 

View are created by filter expressions (see `shoji.filter`) on workspaces or dimensions. Once you 
have obtained a view, you can read from it just like you would from the workspace itself:

```python
view = ws.scRNA.cells[:1000]  # A view of the first 1000 rows along the cells dimension
ages = view.Age				  # The first 1000 values of the Age tensor, as np.ndarray
```

You can also write to the underlying workspace by assigning values to a view:

```python
new_ages = np.array(...)	  # A numpy array of values
view = ws.scRNA.cells[::2]    # A view of every other row along the cells dimension
view.Age = new_ages           # The corresponding rows in the underlying tensor are updated
```

Assigning values in this way is an atomic operation (it will either succeed or fail
completely), and is subject to the [size and time limits](file:///Users/stelin/shoji/html/shoji/index.html#limitations) of shoji transactions.

You can create a view that selects rows along more than one dimension, by providing two 
or more filter expressions separated by comma:

```python
ws = db.scRNA
ws.cells = shoji.Dimension(shape=None)
ws.genes = shoji.Dimension(shape=31768)
ws.Age = shoji.Tensor("string", ("cells",))
ws.Chromosome = shoji.Tensor("string", ("genes",))
# Slice both dimensions:
view = ws.scRNA[ws.Age > 10, ws.Chromosome == "chr1"]
```


"""
from typing import Tuple, Callable, Union
import shoji
import shoji.io
import numpy as np


class View:
	def __init__(self, wsm: shoji.WorkspaceManager, filters: Tuple[shoji.Filter, ...]) -> None:
		super().__setattr__("filters", {f.dim: f for f in filters})
		super().__setattr__("wsm", wsm)

	def groupby(self, labels: Union[str, np.ndarray], projection: Callable = None) -> shoji.GroupViewBy:
		return shoji.GroupViewBy(self, labels, projection)

	def get_length(self, dim: str) -> int:
		if dim in self.filters:
			return len(self.filters[dim].get_rows(self.wsm))
		else:
			return self.wsm._get_dimension(dim).length

	def get_shape(self, tensor: shoji.Tensor) -> Tuple[int, ...]:
		shape = []
		for i, dim in enumerate(tensor.dims):
			if isinstance(dim, str):
				shape.append(self.get_length(dim))
			else:
				shape.append(tensor.shape[i])
		return tuple(shape)

	def __getattr__(self, name: str) -> np.ndarray:
		tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"

		indices = []
		for i, dim in enumerate(tensor.dims):
			if dim in self.filters:
				indices.append(np.sort(self.filters[dim].get_rows(self.wsm)))
			else:
				indices.append(np.arange(tensor.shape[i]))
		# Read the tensor (all or selected rows)
		return shoji.io.read_at_indices(self.wsm, name, indices, tensor.chunks, tensor.compressed, False)

	def __getitem__(self, expr: Union[str, slice]) -> np.ndarray:
		# Is it a slice? Return a slice of the view
		if isinstance(expr, slice):
			# if expr.start is None and expr.stop is None:
			# 	return self
			# elif len(self.filters) == 1:
			# 	dim = next(self.filters.keys())
			# 	return View(self.wsm, self.filters + (shoji.DimensionSliceFilter(dim, expr),))
			# else:
			raise KeyError("Cannot slice a view (not implemented)")
		return self.__getattr__(expr)

	def __setattr__(self, name: str, vals: np.ndarray) -> None:
		tensor: shoji.Tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"
		assert isinstance(vals, (np.ndarray, list, tuple)), f"Value assigned to '{name}' is not a numpy array or a list or tuple of numpy arrays"

		indices = []
		for i, dim in enumerate(tensor.dims):
			if dim in self.filters:
				indices.append(np.sort(self.filters[dim].get_rows(self.wsm)))
			else:
				indices.append(np.arange(tensor.shape[i]))
		tv = shoji.TensorValue(vals)
		shoji.io.write_at_indices(self.wsm._db.transaction, self.wsm, ("tensors", name), indices, tensor.chunks, tv, tensor.compressed)

	def __setitem__(self, name: str, vals: np.ndarray) -> None:
		return self.__setattr__(name, vals)
