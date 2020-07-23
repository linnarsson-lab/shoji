"""
Views let you work with a selected subset of a workspace. Reading from the view automatically
returns values from the selected subset of the database. Values written to the view
are automatically written to the corresponding subset of the database.

Views are *selections of elements* along one or more dimensions. 

View are created by filter expressions (see `shoji.filters`) on workspaces or dimensions. Once you 
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
