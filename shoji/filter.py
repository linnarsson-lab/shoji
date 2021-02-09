"""

## Using filters

Filters are expressions used to select tensor rows, columns, etc., for reading or writing to the database.
Filters can be applied to a workspace, dimension or tensor using the slicing expression `[]`.

### Applying filters to tensors

Filtering on a tensor selects a subtensor by filtering along each dimension, returning a `numpy.ndarray`:

```python
vals = ws.scRNA.Age[ws.scRNA.Tissue == "Cortex"]
# Returns those rows of Age where Tissue equals "Cortex"
```

Filters applied to a tensor must match tensor dimensions. For example, if a tensor is defined
on ("cells", "genes"), then the first filter expression must run along the "cells" dimension
and the second filter expression along the "genes" dimension. If a filter expression is not
given for a dimension, then all indices along that dimension are included in the result. If you 
want to omit dimensions in the middle, use ellipsis:

```python
vals = ws.images.ImageStack[10:20, ..., 90:100]
# Returns indices 10 to 20 along dimension 1, all indices along dimension 2, and 90:100 along dimension 3.
```

### Writing through a filter (view)

Assigning values (which must be a `numpy.ndarray` of the right shape and `dtype`) to a 
filtered tensor causes the corresponding tensor elements in the database to be updated:

```python
vals = np.array(...)  # A numpy array of the right shape and dtype
ws.scRNA.Age[ws.scRNA.Tissue == "Cortex"] = vals
# Those rows of Age where Tissue equals "Cortex" are updated
```

Assigning values in this way is an atomic operation (it will either succeed or fail
completely), and is subject to the [size and time limits](file:///Users/stelin/shoji/html/shoji/index.html#limitations) of shoji transactions.

### Applying filters to dimensions

Filtering on a dimension selects rows of that dimension and returns a `shoji.view.View`.
You can then read tensors from the view, or assign values to tensors through the view:

```python
view = ws.scRNA.cells[:10]
# A view that includes the first ten rows along the 'cells' dimension
t = view.Tissue # Returns a numpy.ndarray of the first ten values of the Tissue tensor 
a = view.Age  # Returns a numpy.ndarray of the first ten values of the Age tensor
view.Age = vals  # assign a np.ndarray of suitable shape and dtype
```

Assigning values in this way is an atomic operation (it will either succeed or fail
completely), and is subject to the [size and time limits](file:///Users/stelin/shoji/html/shoji/index.html#limitations) of shoji transactions.


### Applying filters to workspaces

Recall that workspaces may contain multiple dimensions. When filtering on a workspace,
the dimension that the filter applies to is inferred from the expression. For example,
if the filter expression is `ws.Age > 10` and `Age` is a tensor with `dims=("cells",)`,
then the filter expression applies along the `cells` dimension.

Filtering on a workspace selects rows of the inferred dimension and returns a `shoji.view.View`.
You can then read tensors from the view, or assign values to tensors through the view as above.

However, when filtering on workspaces, you can also simultaneously filter on multiple dimensions,
by providing two or more filter expressions separated by comma:

```python
ws = db.scRNA
ws.cells = shoji.Dimension(shape=None)
ws.genes = shoji.Dimension(shape=31768)
ws.Age = shoji.Tensor("string", ("cells",))
ws.Chromosome = shoji.Tensor("string", ("genes",))
# Slice both dimensions:
view = ws.scRNA[ws.Age > 10, ws.Chromosome == "chr1"]
```

Creating views on workspaces like this is a powerful way to focus on a defined subset of the dataset.

To work with views, see the `shoji.view` API reference.


## Kinds of filters

### Comparisons

You can compare a tensor to a constant:

```python
ws = db.cancer_project
view = ws[ws.Age > 12]  # Create a view of the workspace including only samples where Age > 12
```

Or compare a two tensors:

```python
ws = db.cancer_project
view = ws[ws.Age == ws.OriginalAge]  # Create a view of the workspace including only samples where Age == OriginalAge
```

Comparison operators `==`, `!=`, `>`, `>=`, `<`. `<=` are supported.


### Slices

You can use Python slices on dimensions and tensors (but not on workspaces):

```python
ws = db.cancer_project
view = ws.samples[3:10]  # Create a view along the 'samples' dimension including only rows 3 - 9 (zero-based)
```


### Index arrays

You can use lists, tuples or np.ndarrays of integers to select rows on dimensions and tensors (but not on workspaces):

```python
ws = db.cancer_project
view = ws.samples[(0, 1, 2, 3, 10, 20, 21)]
```


### Boolean arrays

You can use lists, tuples or np.ndarrays of bools to select rows on dimensions and tensors (but not on workspaces):

```python
ws = db.cancer_project
view = ws.samples[(True, False, False, True, False)]
```


### Compound filters

You can combine filters using `&` (and), `|` (or), `~` (not), `^` (xor), and `-` (set difference). 

**Note:** The individual filter expressions must be surrounded by parentheses:

```python
ws = db.cancer_project
view = ws[(ws.Age > 12) & (ws.SampleID < 10)]
```

The set difference operator `-` returns all rows selected by the left-hand expression except
those selected by the right-hand expression.

"""
from typing import Union, Optional
import numpy as np
import shoji


class Filter:
	def __init__(self) -> None:
		self.dim: Union[str, int, None]

	def _combine(self, operator: str, this: Union["Filter", "shoji.View"], other: Union["Filter", "shoji.View"]) -> "Filter":
		def fixup(arg):
			if isinstance(arg, Filter):
				return arg
			elif isinstance(this, shoji.View):
				if len(this.filters) == 1:
					for f in this.filters.values():
						return f
				else:
					raise ValueError("Cannot use logical expression on compound view")

		a = fixup(this)
		b = fixup(other)
		return shoji.CompoundFilter(operator, a, b)

	def __and__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("&", self, other)

	def __rand__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("&", other, self)

	def __or__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("|", self, other)

	def __ror__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("|", other, self)

	def __sub__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("-", self, other)

	def __rsub__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("-", other, self)

	def __xor__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("^", self, other)

	def __rxor__(self, other: Union["Filter", "shoji.View"]) -> "Filter":
		return self._combine("^", other, self)

	def __invert__(self) -> "Filter":
		return shoji.CompoundFilter("~", self, None)

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		pass

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		pass


class CompoundFilter(Filter):
	"""Filter that compares two filters"""
	def __init__(self, operator: str, left_operand: Filter, right_operand: Optional[Filter]) -> None:
		self.operator = operator
		if operator not in ("~", "&", "|", "-", "^"):
			raise SyntaxError(f"Invalid operator {operator}")
		self.left_operand = left_operand
		self.right_operand = right_operand
		if left_operand.dim is not None:
			self.dim = left_operand.dim
			if (right_operand is not None) and (right_operand.dim is not None) and left_operand.dim != right_operand.dim:
				raise SyntaxError("All tensors in an expression must have same first dimensions")
		else:
			self.dim = right_operand.dim if right_operand is not None else None

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.left_operand.get_all_rows(wsm))

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		if self.operator == "&":
			assert isinstance(self.left_operand, Filter)
			assert isinstance(self.right_operand, Filter)
			return np.intersect1d(self.left_operand.get_rows(wsm), self.right_operand.get_rows(wsm))

		if self.operator == "|":
			assert isinstance(self.left_operand, Filter)
			assert isinstance(self.right_operand, Filter)
			return np.union1d(self.left_operand.get_rows(wsm), self.right_operand.get_rows(wsm))

		if self.operator == "-":
			assert isinstance(self.left_operand, Filter)
			assert isinstance(self.right_operand, Filter)
			return np.setdiff1d(self.left_operand.get_rows(wsm), self.right_operand.get_rows(wsm))

		if self.operator == "^":
			assert isinstance(self.left_operand, Filter)
			assert isinstance(self.right_operand, Filter)
			return np.setxor1d(self.left_operand.get_rows(wsm), self.right_operand.get_rows(wsm))

		if self.operator == "~":
			assert isinstance(self.left_operand, Filter)
			return np.setdiff1d(self.left_operand.get_all_rows(wsm), self.left_operand.get_rows(wsm))

	def __repr__(self) -> str:
		if self.operator == "~":
			return f"~{self.left_operand}"
		else:
			return f"({self.left_operand} {self.operator} {self.right_operand})"


class TensorFilter(Filter):
	"""Filter that compares two tensors"""
	def __init__(self, operator: str, left_operand: shoji.Tensor, right_operand: shoji.Tensor) -> None:
		self.operator = operator
		if operator not in (">", "<", ">=", "<=", "==", "!="):
			raise SyntaxError(f"Invalid operator {operator}")
		self.left_operand = left_operand
		if left_operand.rank != 1:
			raise SyntaxError(f"Only rank-1 tensors can be used in filters")
		self.right_operand = right_operand
		if right_operand.rank != 1:
			raise SyntaxError(f"Only rank-1 tensors can be used in filters")
		if isinstance(left_operand.dims[0], str):
			self.dim = left_operand.dims[0]
			if isinstance(right_operand.dims[0], str) and left_operand.dims[0] != right_operand.dims[0]:
				raise SyntaxError("All tensors in an expression must have same first dimensions")
		elif isinstance(right_operand.dims[0], str):
			self.dim = right_operand.dims[0]
		else:
			self.dim = None
		if left_operand.shape[0] != right_operand.shape[0]:
			raise SyntaxError(f"Tensor first dimensions mismatch")

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.left_operand.shape[0])  # TODO: maybe read this from db instead, to avoid stale state

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		if self.operator == ">":
			# Do the range lookups on the tensor indexes
			raise NotImplementedError("Tensor-tensor comparisons not yet supported")
		# etc

	def __repr__(self) -> str:
		return f"({self.left_operand} {self.operator} {self.right_operand})"

class ConstFilter(Filter): 
	"""Filter that compares a tensor to a constant"""
	def __init__(self, operator: str, left_operand: shoji.Tensor, right_operand: Union[str, int, float, bool]) -> None:
		self.operator = operator
		if operator not in (">", "<", ">=", "<=", "==", "!="):
			raise SyntaxError(f"Invalid operator {operator}")
		self.left_operand = left_operand
		if left_operand.rank != 1:
			raise SyntaxError(f"Only rank-1 tensors can be used in filters")
		self.dim = left_operand.dims[0]
		if not isinstance(self.dim, str):
			# TODO: relax this limitation by reading the whole tensor and filtering on the values (maybe?)
			raise SyntaxError(f"Only tensors with named first dimension can be used in filters")
		self.right_operand = right_operand
		if type(right_operand) not in (str, int, float, bool):
			raise SyntaxError(f"Only str, int, float and bool can be used as constants in filters")

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.left_operand.shape[0])

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return shoji.io.const_compare(wsm._db.transaction, wsm, self.left_operand.name, self.operator, self.right_operand)

	def __repr__(self) -> str:
		return f"({self.left_operand} {self.operator} {self.right_operand})"


class DimensionSliceFilter(Filter):
	def __init__(self, dim: shoji.Dimension, slice_: slice) -> None:
		self.dim = dim.name
		self.dimension = dim
		self.slice_ = slice_

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.dimension.length)

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		s = self.slice_.indices(self.dimension.length)
		return np.arange(s[0], s[1], s[2])

	def __repr__(self) -> str:
		s = self.slice_.indices(self.dimension.length)
		return f"({self.dim}[{s[0]}:{s[1]}:{s[2]}])"


class DimensionIndicesFilter(Filter):
	def __init__(self, dim: shoji.Dimension, indices: np.ndarray) -> None:
		self.dim = dim.name
		self.dimension = dim
		self.indices = indices

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.dimension.length)

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		self.indices[self.indices < 0] = self.indices[self.indices < 0] + self.dimension.length
		if not np.all(self.indices < self.dimension.length):
			raise IndexError("Index out of range")
		return self.indices[self.indices < self.dimension.length]

	def __repr__(self) -> str:
		return f"({self.dim}[{self.indices}])"


class DimensionBoolFilter(Filter):
	def __init__(self, dim: shoji.Dimension, selected: np.ndarray) -> None:
		self.dim = dim.name
		self.dimension = dim
		self.selected = selected

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.dimension.length)

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		if self.selected.shape[0] != self.dimension.length:
			raise IndexError(f"Boolean array used for fancy indexing along '{self.dim}' has {self.selected.shape[0]} elements but dimension length is {self.dimension.length}")
		return np.where(self.selected)[0]

	def __repr__(self) -> str:
		return f"({self.dim}[{self.selected}])"


class TensorSliceFilter(Filter):
	def __init__(self, tensor: shoji.Tensor, slice_: slice, axis: int) -> None:
		self.dim = tensor.dims[axis] if tensor.rank > 0 else None
		self.tensor = tensor
		self.slice_ = slice_
		self.axis = axis

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		raise NotImplementedError()

	def get_rows(self, wsm: shoji.WorkspaceManager, n_rows: int = None) -> np.ndarray:
		if n_rows is None:
			n_rows = self.tensor.shape[self.axis]
		s = self.slice_.indices(n_rows)
		return np.arange(s[0], s[1], s[2])

	def __repr__(self) -> str:
		s = self.slice_.indices(self.tensor.shape[self.axis])
		return f"({self.tensor.name}[{s[0]}:{s[1]}:{s[2]}])"


class TensorIndicesFilter(Filter):
	def __init__(self, tensor: shoji.Tensor, indices: np.ndarray, axis: int) -> None:
		self.axis = axis
		self.dim = tensor.dims[axis] if tensor.rank > 0 else None
		self.tensor = tensor
		self.indices = indices

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		raise NotImplementedError()

	def get_rows(self, wsm: shoji.WorkspaceManager, n_rows: int = None) -> np.ndarray:
		if n_rows is None:
			n_rows = self.tensor.shape[self.axis]
		self.indices[self.indices < 0] = self.indices[self.indices < 0] + self.tensor.shape[self.axis]
		if not np.all(self.indices < n_rows):
			raise IndexError("Index out of range")
		return self.indices[self.indices < n_rows]

	def __repr__(self) -> str:
		return f"({self.tensor.name}[{self.indices}])"


class TensorBoolFilter(Filter):
	def __init__(self, tensor: shoji.Tensor, selected: np.ndarray, axis: int) -> None:
		self.axis = axis
		self.dim = tensor.dims[axis] if tensor.rank > 0 else None
		self.tensor = tensor
		self.selected = selected

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		raise NotImplementedError()

	def get_rows(self, wsm: shoji.WorkspaceManager, n_rows: int = None) -> np.ndarray:
		if n_rows is None:
			n_rows = self.tensor.shape[self.axis]

		if self.selected.shape[0] != n_rows:
			raise IndexError(f"Boolean array used for fancy indexing along axis {self.axis} of '{self.tensor.name}' has {self.selected.shape[0]} elements but tensor length is {n_rows}")
		return np.where(self.selected)[0]

	def __repr__(self) -> str:
		return f"({self.tensor.name}[{self.selected}])"
