"""
Dimensions represent named, shared tensor axes. When two tensors share an axis, they are
constrained to have the same number of elements along that axis. 

## Adding data along a dimension

In order to ensure that the dimension constraints are always fulfilled, data must be added
in parallel to all tensors that share a dimension, using the `append()` method. For example, 
to add samples to a project with a `samples` dimension:

```python
ws = db.cancer_project
ws.samples.append({
	"SampleID": np.array([1, 2, 3, 4]),
	"SampleName": np.array(["Sample 1", "Sample 2", "Sample 3", "Sample 4"]),
	"Age": np.array([10, 23, 21, 24]),
	"Description": np.array(["", "", "", ""])
})
```

Note that if you leave out one of the tensors (of the ones that have `"samples"` as their first
dimension), or supply data of inconsistent shape, the append method will raise an exception.

Appending data using this method is guaranteed to never fail with a partial row written (but may not
complete all the rows successfully), and will never leave the database in an inconsistent state 
(e.g. with data appended to only one of the tensors). If you need a stronger guarantee of success/failure,
wrap the append() in a `shoji.transaction.Transaction`.
"""
from typing import Optional, Dict
import numpy as np
import shoji


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

	def append(self, vals: Dict[str, np.ndarray]) -> None:
		"""
		Append values to all tensors that have this as their first dimension

		Args:
			vals: Dict mapping tensor names (`str`) to tensor values (`np.ndarray`)

		Returns:
			n_rows: the number of rows actually appended

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
		for values in vals.values():
			if n_rows == -1:
				n_rows = len(values)
			elif len(values) != n_rows:
				raise ValueError(f"Length (along first dimension) of tensors must be the same when appending")

		# Total size of the transaction (for jagged arrays, sum of row sizes)
		n_bytes = sum([(n.size * n.itemsize if isinstance(n, np.ndarray) else (sum(i.size * i.itemsize) for i in n)) for n in vals.values()])
		n_batches = max(1, n_bytes // 5_000_000) # Should be plenty, given that we'll also be compressing the rows when writing
		n_rows_per_transaction = max(1, n_rows // n_batches)
		ix: int = 0
		while ix < n_rows:
			with shoji.Transaction(self.wsm):
				batch = {k: v[ix: ix + n_rows_per_transaction] for k, v in vals.items()}
				shoji.io.append_tensors(self.wsm._db.transaction, self.wsm, self.name, batch)
			ix += n_rows_per_transaction
