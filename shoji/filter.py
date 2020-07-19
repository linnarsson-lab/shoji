"""
Filters are expressions used to read and write selected rows from the database. They are typically
used to create `shoji.view.View`s of the database, by applying filters to the workspace using the
slicing expression `[]`.

## Constant comparisons

You can compare a tensor to a constant:

```python
ws = db.cancer_project
view = ws[ws.Age > 12]  # Create a view of the workspace including only samples where Age > 12
```

Comparison operators `==`, `!=`, `>`, `>=`, `<`. `<=` are supported.

## Compound filters

You can combine filters using `&` (and), `|` (or) and `~` (not). Make sure to use parentheses
around the individual filter expressions:

```python
ws = db.cancer_project
view = ws[(ws.Age > 12) & (ws.SampleID < 10)]
```



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
		return other._combine("-", other, self)

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
	def __init__(self, tensor: shoji.Tensor, slice_: slice) -> None:
		self.dim = tensor.dims[0] if tensor.rank > 0 else None
		self.tensor = tensor
		self.slice_ = slice_

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.tensor.shape[0])

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		s = self.slice_.indices(self.tensor.shape[0])
		return np.arange(s[0], s[1], s[2])

	def __repr__(self) -> str:
		s = self.slice_.indices(self.tensor.shape[0])
		return f"({self.tensor.name}[{s[0]}:{s[1]}:{s[2]}])"


class TensorIndicesFilter(Filter):
	def __init__(self, tensor: shoji.Tensor, indices: np.ndarray) -> None:
		self.dim = tensor.dims[0] if tensor.rank > 0 else None
		self.tensor = tensor
		self.indices = indices

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.tensor.shape[0])

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		self.indices[self.indices < 0] = self.indices[self.indices < 0] + self.tensor.shape[0]
		if not np.all(self.indices < len(self.tensor)):
			raise IndexError("Index out of range")
		return self.indices[self.indices < self.tensor.shape[0]]

	def __repr__(self) -> str:
		return f"({self.tensor.name}[{self.indices}])"


class TensorBoolFilter(Filter):
	def __init__(self, tensor: shoji.Tensor, selected: np.ndarray) -> None:
		self.dim = tensor.dims[0] if tensor.rank > 0 else None
		self.tensor = tensor
		self.selected = selected

	def get_all_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		return np.arange(self.tensor.shape[0])

	def get_rows(self, wsm: shoji.WorkspaceManager) -> np.ndarray:
		if self.selected.shape[0] != self.tensor.shape[0]:
			raise IndexError(f"Boolean array used for fancy indexing along first dimension of '{self.tensor.name}' has {self.selected.shape[0]} elements but tensor length is {self.tensor.shape[0]}")
		return np.where(self.selected)[0]

	def __repr__(self) -> str:
		return f"({self.tensor.name}[{self.selected}])"
