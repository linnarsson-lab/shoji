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

Note that if you leave out one of the tensors (of the ones that have `"samples"` as their first
dimension), or supply data of inconsistent shape, the append method will raise an exception.

Appending data using this method is guaranteed to never fail with a partial row written (but may not
complete all the rows successfully), and will never leave the database in an inconsistent state 
(e.g. with data appended to only one of the tensors). If you need a stronger guarantee of success/failure,
wrap the `append()` in a `shoji.transaction.Transaction`.
"""
from typing import Optional, Dict, Union, List
import numpy as np
import shoji
import fdb


class Dimension:
	"""
	Class representing a named dimension, which can be shared by multiple `shoji.tensor.Tensor`s
	"""
	def __init__(self, *, shape: Optional[int], length: int = 0) -> None:
		if shape == -1:
			shape = None
		if shape is not None and shape < 0:
			raise ValueError("Length must be non-negative")
		self.shape = shape  # None means variable length (i.e. can append) or jagged

		self.length = length  # Actual length, will be set when dimension is read from db
		self.name = ""  # Will be set if the Dimension is read from the db
		self.wsm: Optional[shoji.WorkspaceManager] = None  # Will be set if the Dimension is read from the db

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

	def append(self, vals: Dict[str, Union[List[np.ndarray], np.ndarray]]) -> None:
		"""
		Append values to all tensors that have this as their first dimension

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

		# Check that all the values have same length
		n_rows = -1
		for name, values in vals.items():
			assert isinstance(values, np.ndarray), f"Input values must be numpy ndarrays, but '{name}' was {type(values)}"
			assert values.ndim >= 1, f"Input values must be at least 1-dimensional, but '{name}' was scalar"
			if n_rows == -1:
				n_rows = len(values)
			elif len(values) != n_rows:
				raise ValueError(f"Length (along first dimension) of tensors must be the same when appending, but '{name}' was length {len(values)} while other arrays were {n_rows} long")

		n_bytes = 0
		for name, val in vals.items():
			tv = shoji.TensorValue(val)
			n_bytes += tv.size_in_bytes()

		n_bytes_per_transaction = 1_000_000  # Starting point, but we'll adapt it below
		ix: int = 0
		while ix < n_rows:
			n_rows_per_transaction = int(max(1, n_rows / (n_bytes / n_bytes_per_transaction)))
			batch = {k: v[ix: ix + n_rows_per_transaction] for k, v in vals.items()}
			try:
				n_bytes_written = shoji.io.append_tensors(self.wsm._db.transaction, self.wsm, self.name, batch)
			except fdb.impl.FDBError as e:
				if e.code in (1004, 1007, 1031, 2101) and n_rows_per_transaction > 1:  # Too many bytes or too long time, so try again with less
					n_bytes_per_transaction = max(1, n_bytes_per_transaction // 2)
					continue
				else:
					raise e
			if n_bytes_written < n_bytes_per_transaction // 2:  # Not enough bytes, so increase for next round
				n_bytes_per_transaction *= 2
			ix += n_rows_per_transaction
