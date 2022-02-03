"""
Dimensions represent named, shared tensor axes. When two tensors share an axis, they are
constrained to have the same number of elements along that axis. 

..image:: assets/bitmap/tensor_dims@2x.png

## Overview

Dimensions must be defined in the `shoji.workspace.Workspace` before they can be used:

```python
db = shoji.connect()
db.scRNA = shoji.Workspace()
db.scRNA.cells = shoji.Dimension(shape=None)
db.scRNA.genes = shoji.Dimension(shape=5000)
```

Once the dimensions have been declared, tensors can use those dimensions:

```python
db.scRNA.Expression = shoji.Tensor("int16", ("cells", "genes"))
db.scRNA.CellType = shoji.Tensor("string", ("cells",))
db.scRNA.Length = shoji.Tensor("uint16", ("genes",))
db.scRNA.Chromosome = shoji.Tensor("string", ("genes",))
```

## Adding data along a dimension

In order to ensure that the dimension constraints are always fulfilled, data must be added
in parallel to all tensors that share a dimension, using the `append()` method on the 
dimension. For example, to add cells to a project with a `cells` dimension:

```python
db.scRNA.cells.append({
	"CellType": np.array(["Neuron", "Astrocyte", "Neuron", "Miroglia"], dtype=object),
	"Expression": np.random.randint(0, 10, size=(4, 5000), dtype="uint16")
})
```

Note that if you leave out one of the tensors (of the ones that have `"cells"` as one of their
dimensions), or supply data of inconsistent shape, the append method will raise an exception.

Appending data using this method is guaranteed to never fail with a partial row written (but may not
complete all the rows successfully), and will never leave the database in an inconsistent state 
(e.g. with data appended to only one of the tensors). If you need a stronger guarantee of success/failure,
wrap the `append()` in a `shoji.transaction.Transaction`.

## Grouping

Calling `.groupby(labels)` on a dimension creates a `shoji.groupby.GroupDimensionBy` object. 
The `labels` should be the name of a tensor which is used to group the dimension.
Calling an aggregation function such as `mean()` then returns a grouped tensor.

For example, suppose you have an Expression tensor with dimensions ("genes", "cells") and
a ClusterID tensor with dimension ("cells",). Suppose there are 10 distinct values of
ClusterID. The following code will return an np.ndarray of shape ("genes", 10) containing 
mean values for each ClusterID:

```python
grouped = ws.cells.groupby("ClusterID")
(labels, values) = grouped.mean("Expression")
# labels is a list of distinct ClusterID values
# values is a np.ndarray where the "cells" dimension is replaced by the distinct cluster IDs
```

"""
from typing import Optional, Dict, Union, List, Callable
import numpy as np
import shoji
import fdb


class Dimension:
	"""
	Class representing a named dimension, which can be shared by multiple `shoji.tensor.Tensor`s
	"""
	def __init__(self, shape: Optional[int], length: int = 0) -> None:
		"""
		Create a new Dimension

		Args:
			shape		An integer, or None to create a variable-length dimension
			length		(do not use when creating a new Dimension)
		"""
		if shape == -1:
			shape = None
		if shape is not None and shape < 0:
			raise ValueError("Length must be non-negative")
		self.shape = shape  # None means variable length (i.e. can append) or jagged

		self.length = length  # Actual length, will be set when dimension is read from db
		self.name = ""  # Will be set if the Dimension is read from the db
		self.wsm: Optional[shoji.Workspace] = None  # Will be set if the Dimension is read from the db

	# Support pickling
	def __getstate__(self):
		"""Return state values to be pickled."""
		return (self.shape, self.length)

	def __setstate__(self, state):
		"""Restore state from the unpickled state values."""
		self.shape, self.length = state

	def __getitem__(self, key) -> "shoji.View":
		if self.wsm is None:
			raise ValueError("Cannot filter unbound dimension")
		if isinstance(key, slice):
			return shoji.View(self.wsm, (shoji.DimensionSliceFilter(self, key),))
		if isinstance(key, (list, tuple, int)):
			key = np.array(key)
		if isinstance(key, np.ndarray):
			if np.issubdtype(key.dtype, np.bool_):
				return shoji.View(self.wsm, (shoji.DimensionBoolFilter(self, key),))
			elif np.issubdtype(key.dtype, np.int_):
				return shoji.View(self.wsm, (shoji.DimensionIndicesFilter(self, key),))
		raise IndexError(f"Invalid fancy index along dimension '{self.name}' (only slice, bool array or int array are allowed)")

	def __repr__(self) -> str:
		if self.shape is None:
			return "<Dimension of variable shape>"
		else:
			return f"<Dimension of shape {self.shape}>"

	def __len__(self) -> int:
		return self.length

	def groupby(self, labels: Union[str, np.ndarray], projection: Callable = None) -> "shoji.GroupDimensionBy":
		return shoji.GroupDimensionBy(self, labels, projection)

	def append(self, vals: Dict[str, Union[List[np.ndarray], np.ndarray]]) -> None:
		"""
		Append values to all tensors that have this as one of their dimensions

		Args:
			vals: Dict mapping tensor names (`str`) to tensor values (`np.ndarray`)

		Remarks:
			The method is transactional, i.e. it's guaranteed to either succeed or
			fail without leaving the database in an inconsistent state. If it fails, 
			a smaller than expected number of rows may have been appended, but all 
			tensors will have the same length along the dimension.
		"""
		assert self.wsm is not None, "Cannot append to unsaved dimension"
		assert self.shape is None, "Cannot append to fixed-size dimension"

		# Figure out the relevant axes
		axes: List[int] = []
		n_rows = -1
		for name, values in vals.items():
			assert isinstance(values, np.ndarray), f"Input values must be numpy ndarrays, but '{name}' was {type(values)}"
			assert values.ndim >= 1, f"Input values must be at least 1-dimensional, but '{name}' was scalar"
			tensor = self.wsm._get_tensor(name)
			assert self.name in tensor.dims, f"Input values were provided for '{name}', but '{self.name}' is not one of its dimensions"
			axis = tensor.dims.index(self.name)
			if n_rows == -1:
				n_rows = values.shape[axis]
			elif values.shape[axis] != n_rows:
				raise ValueError(f"Length (along first dimension) of tensors must be the same when appending, but '{name}' was length {len(values)} while other arrays were {n_rows} long")
			axes.append(axis)

		names = list(vals.keys())
		values = [shoji.TensorValue(x) for x in vals.values()]
		shoji.io.append_values_multibatch(self.wsm, names, values, tuple(axes))

	def extend(self, length: int) -> int:
		"""
		Extend the dimension by the indicated length

		Args:

			length: 	Number of elements to extend the dimension by
		
		Returns:

			index:		The index to the newly extended part of the dimension
		
		Remarks:
			This method extends the dimension transactionally, i.e. it is safe to call from multiple
			threads and processes concurrently. However, the newly extended segment may not be
			placed at the current end of the dimension (since another thread might be extending
			concurrently). The caller should use the returned index to know where the extension
			ended up.
		"""
		return self.ws.io.extend_dimension(self.name, length)

