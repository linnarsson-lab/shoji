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
from typing import Tuple, Callable, Union, List
import shoji
import shoji.io
from shoji.io import Compartment
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

	def _read_batch(self, tensor: shoji.Tensor, start: int, end: int) -> np.ndarray:
		if tensor.jagged:
			raise ValueError(f"Cannot read batches of jagged tensor '{tensor.name}'")
		indices = []
		for i, dim in enumerate(tensor.dims):
			if dim in self.filters:
				indices = self.filters[dim].get_rows(self.wsm)
			else:
				indices = np.arange(tensor.shape[i])
			if i == 0:
				indices[-1] = indices[-1][start:end]

		# Read the tensor (selected rows)
		result = shoji.io.read_at_indices(self.wsm, tensor.name, indices, tensor.chunks, tensor.compressed, False)
		return result

	def __getattr__(self, name: str) -> np.ndarray:
		return shoji.io.read_filtered(self.wsm, name, self.filters.values())

	def __getitem__(self, expr: Union[str, slice]) -> np.ndarray:
		if isinstance(expr, slice):
			raise KeyError("Cannot slice a view (not implemented)")
		return self.__getattr__(expr)

	def __setattr__(self, name: str, vals: Union[List[np.ndarray], np.ndarray]) -> None:
		return shoji.io.write_filtered(self.wsm._db, self.wsm, name, vals, self.filters.values())

	def __setitem__(self, name: str, vals: np.ndarray) -> None:
		return self.__setattr__(name, vals)
