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
import numpy as np
import xarray as xr


class View:
	def __init__(self, wsm: shoji.WorkspaceManager, filters: Tuple[shoji.Filter, ...]) -> None:
		super().__setattr__("filters", {f.dim: f for f in filters})
		super().__setattr__("wsm", wsm)
	
	def xarray(self, *, tensors: Tuple[str, ...] = None) -> xr.Dataset:
		"""
		Return the whole view as an in-memory `xarray.Dataset` object with named dimensions

		Args:
			tensors:	The tensors to include in the xarray, or None to include all
		
		Remarks:
			All tensors in the view are loaded into the xarray Dataset, with the exception
			of jagged tensors, which are not supported by xarray.
		"""
		variables = {}
		for t in self.wsm._tensors():
			if tensors is not None and t not in tensors:
				continue
			tensor = self.wsm._get_tensor(t)
			if tensor.jagged:
				continue
			variables[t] = (tensor.dims, self[t])
		return xr.Dataset(variables)

	def groupby(self, labels: Union[str, np.ndarray], projection: Callable = None) -> shoji.GroupViewBy:
		return shoji.GroupViewBy(self, labels, projection)

	def get_length(self, dim: str) -> int:
		if dim in self.filters:
			return len(self.filters[dim].get_rows(self.wsm))
		else:
			return self.wsm._get_dimension(dim).length

	def get_shape(self, tensor: shoji.Tensor) -> int:
		shape = []
		for i, dim in enumerate(tensor.dims):
			if isinstance(dim, str):
				shape.append(self.get_length(dim))
			else:
				shape.append(tensor.shape[i])
		return shape

	def _read_chunk(self, tensor: shoji.Tensor, start: int, end: int) -> np.ndarray:
		if tensor.rank > 0 and tensor.dims[0] in self.filters:
			indices = self.filters[tensor.dims[0]].get_rows(self.wsm)[start: end]
		else:
			indices = np.arange(start, end)

		# Read the tensor (selected rows)
		result = shoji.io.read_tensor_values(self.wsm._db.transaction, self.wsm, tensor.name, tensor, indices)
		# Filter the remaining dimensions
		for i, dim in enumerate(tensor.dims):
			if i == 0:
				continue
			if isinstance(dim, str) and dim in self.filters:
				# Filter this dimension
				indices = self.filters[dim].get_rows(self.wsm)
				result = result.take(indices, axis=i)
		return result

	def __getattr__(self, name: str) -> np.ndarray:
		# Get the tensor
		tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"

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

	def __getitem__(self, expr: Union[str, slice]) -> np.ndarray:
		# Is it a slice? Return a slice of the view
		if isinstance(expr, slice):
			if expr.start is None and expr.stop is None:
				return self
			elif len(self.filters) == 1:
				dim = next(self.filters.keys())
				return View(self.wsm, self.filters + (shoji.DimensionSliceFilter(dim, expr),))
			else:
				raise KeyError("Cannot slice a view unless it's filtered on exactly one dimension")
		elif expr == ...:
			return NonTransactionalView(self)
		return self.__getattr__(expr)

	def __setattr__(self, name: str, vals: np.ndarray) -> None:
		tensor: shoji.Tensor = self.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"
		assert isinstance(vals, (np.ndarray, list, tuple)), f"Value assigned to '{name}' is not a numpy array or a list or tuple of numpy arrays"

		for i, dim in enumerate(tensor.dims):
			if i == 0:
				continue
			if dim in self.filters:
				raise IndexError("Cannot write to view filtered non non-first tensor dimension")
		if tensor.rank == 0:
			indices = np.array([0], dtype="int32")
		elif tensor.dims[0] in self.filters:
			indices = self.filters[tensor.dims[0]].get_rows(self.wsm)
		else:
			indices = np.arange(tensor.shape[0])
		tv = shoji.TensorValue(vals)
		shoji.io.write_tensor_values(self.wsm._db.transaction, self.wsm, name, tv, indices)

	def __setitem__(self, name: str, vals: np.ndarray) -> None:
		return self.__setattr__(name, vals)


class NonTransactionalView:
	def __init__(self, view: View) -> None:
		super().__setattr__("view", view)
	
	def xarray(self, *, tensors: Tuple[str, ...] = None) -> xr.Dataset:
		"""
		Return the whole view as an in-memory `xarray.Dataset` object with named dimensions

		Args:
			tensors:	The tensors to include in the xarray, or None to include all
		
		Remarks:
			All tensors in the view are loaded into the xarray Dataset, with the exception
			of jagged tensors, which are not supported by xarray.
		"""
		variables = {}
		for t in self.view.wsm._tensors():
			if tensors is not None and t not in tensors:
				continue
			tensor = self.view.wsm._get_tensor(t)
			if tensor.jagged:
				continue
			variables[t] = (tensor.dims, self[t])
		return xr.Dataset(variables)

	def get_length(self, dim: str) -> int:
		if dim in self.filters:
			return len(self.filters[dim].get_rows(self.wsm))
		else:
			return self.wsm._get_dimension(dim).length

	def __getattr__(self, name: str) -> np.ndarray:
		# Get the tensor
		tensor = self.view.wsm[name]
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a Tensor"
		
		if tensor.rank == 0:
			return self.view[name]

		if tensor.dims[0] in self.view.filters:
			indices = np.sort(self.view.filters[tensor.dims[0]].get_rows(self.view.wsm))
		else:
			indices = np.arange(tensor.shape[0])

		# Read the tensor (all or selected rows)
		if tensor.dtype == "string" or tensor.rank == 0:
			result = shoji.io.read_tensor_values(self.view.wsm._db.transaction, self.view.wsm, name, tensor, indices)
		else:
			shape = self.view.get_shape(tensor)  # Shape of expected result from this view after filtering
			result = np.zeros(shape, dtype=tensor.dtype)
			BYTES_PER_BATCH = 10_000_000
			n_rows_per_batch = max(1, BYTES_PER_BATCH // int(tensor.bytewidth * np.prod(tensor.shape[1:])))
			for ix in range(0, shape[0], n_rows_per_batch):
				result[ix:ix + n_rows_per_batch] = shoji.io.read_tensor_values(self.view.wsm._db.transaction, self.view.wsm, name, tensor, indices[ix: ix + n_rows_per_batch])

		# Filter the remaining dimensions
		for i, dim in enumerate(tensor.dims):
			if i == 0:
				continue
			if isinstance(dim, str) and dim in self.view.filters:
				# Filter this dimension
				indices = self.view.filters[dim].get_rows(self.view.wsm)
				result = result.take(indices, axis=i)
		return result

	def __getitem__(self, expr: Union[str, slice]) -> np.ndarray:
		# Is it a slice? Return a slice of the view
		if isinstance(expr, slice):
			if expr.start is None and expr.stop is None:
				return self
			elif len(self.view.filters) == 1:
				dim = next(self.view.filters.keys())
				return NonTransactionalView(View(self.view.wsm, self.view.filters + (shoji.DimensionSliceFilter(dim, expr),)))
			else:
				raise KeyError("Cannot slice a view unless it's filtered on exactly one dimension")
		return self.__getattr__(expr)
