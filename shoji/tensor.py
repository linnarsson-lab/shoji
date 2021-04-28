"""
All data in Shoji is stored as N-dimensional tensors. A tensor is a
generalisation of scalars, vectors and matrices to N dimensions. 

Tensors are defined by their *rank*, *datatype*, *dimensions* and *shape*.
In addition, tensors can be *jagged* (i.e. some dimensions have non-uniform
sizes).

Tensors can be extended along any of their dimensions (unless the 
dimension is declared fixed-length) by appending values. 

## Overview

Tensors are created like this:

```python
import shoji
tissues = ...        # Assume we have an np.ndarray of tissue names
db = shoji.connect() # Connect to the database
ws = db.scRNA        # scRNA is a Workspace in the database, previously created

ws.Tissue = shoji.Tensor("string", ("cells",), inits=tissues)
```

The tensor is declared with a datatype `"string"`, a tuple of dimensions `("cells",)` and an optional `np.ndarray` of initial values.

### Rank

The *rank* of a tensor is the number of dimensions of the tensor. A scalar 
value has rank 0, a vector has rank 1, and a matrix has rank 2. Higher ranks
are possible; for example, a vector of 2D images would have rank 3, and a
timelapse recording in several color channels would have rank 4 (timepoint, x, y,
color).

### Datatype

Tensors support the following datatypes: 

```python
"bool"
"uint8", "uint16", "uint32", "uint64"
"int8", "int16", "int32", "int64"
"float16", "float32", "float64"
"string"
```

The datatype of a tensor must always be declared; there is no default type.

When a tensor is created, any initial values provided (via the `inits` argument)
must have the matching numpy datatype. The bool and numeric datatypes match 1:1 with numpy dtypes. 

However, the Shoji `string` datatype is a Unicode string of variable length,
which corresponds to a numpy array of string objects. That is, the corresponding
numpy datatype is *not* `str` or `"unicode"`. Instead, Shoji string tensors correspond
to numpy `object` arrays whose elements are Python `str` objects. You can cast a numpy
`str` array to an `object` array as follows:

```python
import numpy as np
s = np.array(["dog", "cat", "apple", "orange"])  # s.dtype.kind == 'U'
t = s.astype(object)  # t.dtype.kind == 'O'
# Or directly, using dtype
s = np.array(["dog", "cat", "apple", "orange"], dtype="object")
```

The reason for this discrepancy is that numpy `str` arrays store only
fixed-length strings, whereas Shoji `string` tensors store strings of variable length.

### Dimensions

When creating a tensor, its dimensions must be declared using a tuple. 
Scalars have rank zero, and are declared with the empty tuple `()`. 
Vectors have rank one, and are declared with a single-element tuple, e.g. 
`(20,)` (note the comma, which is necessary). Matrices have rank 2, and are
declared with a two-element tuple, e.g. `(20, 40)`. Higher-rank tensors are 
declared with correspondingly longer tuples. 

Dimensions can be fixed or variable-length. A fixed-length dimension is 
declared with an integer specifying the number of elements of the dimension.
A variable-length dimension is declared as `None`. For example, `(10, None)` 
is a matrix with ten rows and a variable number of columns. 

The meaning of a variable-length dimension is slightly different for regular
and jagged tensors. For a regular tensor, if a dimension is variable-length then 
the tensor can be extended along that dimension by appending data. Thus a tensor declared 
with `dims=(None, 10)` at any point in time has a fixed number of
rows and columns, but rows can be appended (see `shoji.dimension` and 
`shoji.dimension.Dimension.append`).

If the tensor is jagged, then a variable-length dimension can contain 
individual rows (columns, etc) of different lengths.

Each dimension of a tensor can be named, and named dimensions (within a 
`shoji.workspace.Workspace`) are constrained to have the same number of elements. 
For example, if two tensors are declared with dimensions `("cells",)` 
and `("cells", "genes")`, then the first dimensions are guaranteed to have 
the same number of elements, which are assumed to be in the same order.

Named dimensions must be declared before they are used; see `shoji.dimension`.


### Shape

The `shape` of a `shoji.tensor.Tensor` is a tuple of integers that gives the current
shape of the tensor as stored in the database. For example, a tensor with `dims=(None, 10, 20)`
might have `shape=(10, 10, 20)`, indicating that currently the tensor has ten rows. Since
the first dimension is variable-length (in this case), rows might be appended later, and the shape
would change to reflect the new number of rows.


## Chunks

Data in Shoji is stored and retrieved as N-dimensional chunks. When you
read or write from a tensor, your operations are converted to operations
on chunks. For example, if you access a single element of a matrix, under 
the hood the whole chunk containing the element is retrieved. 

When you create a tensor, you can optionally specify the chunk size along
each dimension. Chunking is **very** important for performance. Small chunks 
such as (10,100) or even (1, 100) can be an order of magnitude 
faster for random access, but an order of magnitude slower for contiguous
access, as compared to large chunks like (100, 1000) or (1000, 1000). If
you know that you will only read in large contiguous blocks, use large chunks 
along those dimensions. If you know you will be reading many randomly 
placed single or few indices, use small chunks along those dimensions.


## Reading from tensors

The universal method for reading data in shoji is to create a `shoji.view.View`
of the workspace. However, sometimes you just want to read from one tensor
and don't care about creating a view. Shoji supports indexing tensors similar 
to numpy "fancy indexing" (and similar to how views are created):

```python
x = ws.Expression[:]  # Read the whole tensor
y = ws.Expression[10:20]  # Read a slice
z = ws.Expression[(1, 2, 5, 9)]  # Read specific rows
w = ws.Expression[(True, False, True)]  # Read rows given by bool mask array
```

The above expressions are just shorthands for creating the corresponding view
and immediately reading from the tensor. There is no difference in performance.
For example, these two expressions below are equivalent:

```python
x = ws.Expression[:]
x = ws[:].Expression
```

## Jagged tensors

If a tensor is declared *jagged*, the size along variable-length dimensions 
can be different for different rows (columns, etc.). For example:

```python
ws.cells = shoji.Dimension(shape=None)
ws.Image = shoji.Tensor("uint16", ("cells", None, None), jagged=True)
```

In this example, we declare a 3D jagged tensor `Image`, where dimensions 2 and 3 
are variable-length. This could be used to store 2D images of cells, each of which 
has a different width and height. The first dimension represents the objects 
(individual cells) and the 2nd and 3rd dimensions represent the images. Accessing 
a single row of this tensor would return a single 2D image matrix. Accessing a set 
of rows would return a list of 2D images.

In a similar way, we could store multichannel timelapse images of cells:

```python
ws.cells = shoji.Dimension(shape=None)
ws.channels = shoji.Dimension(shape=3)
ws.timepoints = shoji.Dimension(shape=1200)  # 1200 timepoints
ws.Image = shoji.Tensor("uint16", ("cells", "channels", "timepoints", None, None), jagged=True)
```

In this examples, `Image` is a 5-dimensional tensor, where the last two dimensions 
have variable length.
"""
from typing import Tuple, Union, List, Optional, Callable
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import shoji
import sys
import logging


FancyIndexElement = Union["shoji.Filter", slice, int, np.ndarray]
FancyIndex = Union[FancyIndexElement, Tuple[FancyIndexElement, ...]]

class TensorValue:
	def __init__(self, values: Union[Tuple[np.ndarray], List[np.ndarray], np.ndarray]) -> None:
		self.values = values
		if isinstance(values, (list, tuple)):
			self.jagged = True
			self.rank = values[0].ndim + 1
			self.dtype = values[0].dtype.name
			if self.dtype == "object":
				self.dtype = "string"

			shape = np.array(values[0].shape)
			for i, array in enumerate(values):
				if not isinstance(array, np.ndarray):
					raise ValueError("Rows of jagged tensor must be numpy ndarrays")
				if self.rank != array.ndim + 1:
					raise ValueError("Rows of jagged tensor cannot be mixed rank")
				if self.dtype != array.dtype:
					raise ValueError("Rows of jagged tensor cannot be mixed dtype")
				if self.dtype == "string":
					if not all([isinstance(x, str) for x in array.flat]):
						raise TypeError("string tensors (numpy dtype='object') must contain only string elements")
				if array.ndim != len(shape):
					raise ValueError(f"Rank mismatch: shape {array.shape} of subarray at row {i} is not the same rank as shape {shape} at row 0")
				shape = np.maximum(shape, array.shape)
			self.shape = tuple([len(values)] + list(shape))
		else:
			self.jagged = False
			self.rank = values.ndim
			self.dtype = values.dtype.name
			if self.dtype == "object":
				self.dtype = "string"
				if not all([isinstance(x, str) for x in values.flat]):
					raise TypeError("string tensors (numpy dtype='object') must contain only string elements")
			self.shape = values.shape

		if self.dtype not in Tensor.valid_types:
			raise TypeError(f"Invalid dtype '{self.dtype}' for tensor value")

	@property
	def size(self) -> int:
		return np.prod(self.shape)

	def __len__(self) -> int:
		if self.rank > 0:
			return self.shape[0]
		return 1
	
	def __iter__(self):
		for row in self.values:
			yield row

	def __getitem__(self, slice_) -> "TensorValue":
		if self.jagged:
			if isinstance(slice_, slice):
				slice_ = (slice_)
			slice_ = slice_ + (slice(None),) * (self.rank - len(slice_))
			sliced = [vals[slice_[1:]] for vals in self.values[slice_[0]]]
			return TensorValue(sliced)
		return TensorValue(self.values[slice_])

	def size_in_bytes(self) -> int:
		n_bytes = 0
		if not self.jagged:
			if self.dtype == "string":
				n_bytes += sum([len(s) for s in self.values]) * 2
			else:
				n_bytes += self.values.size * self.values.itemsize  # type: ignore
		else:
			for row in self.values:
				if self.dtype == "string":
					n_bytes += sum([len(s) for s in row]) * 2
				else:
					n_bytes += row.size * row.itemsize
		return n_bytes

class Tensor:
	valid_types = ("bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "string")

	def __init__(self, dtype: str, dims: Union[Tuple[Union[None, int, str], ...]], *, chunks: Tuple[int, ...] = None, jagged: bool = False, inits: Union[List[np.ndarray], np.ndarray] = None) -> None:
		"""
		Args:
			dtype:	string giving the datatype of the tensor elements
			dims:	A tuple of None, int, string (empty tuple designates a scalar)
			chunks: Tuple defining the chunk size along each dimension, or "auto" to use automatic chunking
			jagged: If true, this is a jagged tensor (and inits must be a list of ndarrays)
			inits:	Optional values to initialize the tensor with

		Remarks:
			Dimensions are specified as:

				None:		resizable/jagged anonymous dimension
				int:		fixed-shape anonymous dimension
				string:		named dimension

			Chunking is VERY important for performance. Small chunks such as (10,100) or even (1, 100) can be an order of magnitude 
			faster for random access, but an order of magnitude slower for contiguous access, as compared to large chunks 
			like (100, 1000) or (1000, 1000). If you know that you will only read in large contiguous blocks, use large chunks 
			along those dimensions. If you know you will be reading many randomly placed single or few indices, use small chunks 
			along those dimensions.

			For rank-0 tensors, use chunks=(1,)
		"""
		self.dtype = dtype
		
		# Check that the type is valid
		if dtype not in Tensor.valid_types:
			raise TypeError(f"Invalid Tensor type {dtype}")

		self.dims = dims
		self.jagged = jagged

		self.name = ""  # Will be set if the Tensor is read from the db
		self.wsm: Optional[shoji.WorkspaceManager] = None  # Will be set if the Tensor is read from the db

		if inits is None:
			self.inits: Optional[TensorValue] = None
			self.shape = (0,) * len(dims)
		else:
			# If scalar, convert to an ndarray scalar which will have shape ()
			if np.isscalar(inits):
				self.inits = TensorValue(np.array(inits, dtype=self.numpy_dtype()))
			else:
				self.inits = TensorValue(inits)
			if self.inits.jagged and not self.jagged:
				raise ValueError(f"Jagged inits cannot be used to create non-jagged tensor")
			self.shape = self.inits.shape

			if len(self.dims) != len(self.shape):
				raise ValueError(f"Rank mismatch: shape {self.dims} declared is not the same rank as shape {self.shape} of values")

			if self.dtype != self.inits.dtype:
				raise TypeError(f"Tensor dtype '{self.dtype}' does not match dtype of inits '{self.inits.dtype}'")

		for ix, dim in enumerate(self.dims):
			if dim is not None and not isinstance(dim, int) and not isinstance(dim, str):
				raise ValueError(f"Dimension {ix} '{dim}' is invalid (must be None, int or str)")

			if isinstance(dim, int) and self.inits is not None:
				if self.shape[ix] != dim:  # type: ignore
					raise IndexError(f"Mismatch between the declared shape {dim} of dimension {ix} and the inferred shape {self.shape} of values")

		self.chunks: Tuple[int, ...] = ()
		if chunks is None:
			if dtype in ("bool", "uint8", "int8"):
				byte_size = 1
			if dtype in ("uint16", "int16", "float16"):
				byte_size = 2
			elif dtype in ("uint32", "int32", "float32"):
				byte_size = 4
			elif dtype in ("uint64", "int64", "float64"):
				byte_size = 8
			elif dtype == "string":
				byte_size = 32  # This will fail for very long strings
			if self.rank == 0:
				self.chunks = ()
			elif self.rank == 1:
				self.chunks = (500 // byte_size,)
			else:
				desired_sizes = (300 // byte_size, 100) + (1,) * (self.rank - 2)
				max_sizes = (dim if isinstance(dim, int) else sys.maxsize for dim in self.dims)
				self.chunks = tuple(min(a,b) for a,b in zip(max_sizes, desired_sizes))
		else:
			if len(chunks) != self.rank:
				raise ValueError(f"chunks={chunks} is wrong number of dimensions for rank-{self.rank} tensor" + (" (use () for rank-0 tensor)" if self.rank == 0 else ""))
			self.chunks = chunks
		self.initializing = False

	# Support pickling
	def __getstate__(self):
		"""Return state values to be pickled."""
		return (self.dtype, self.jagged, self.dims, self.shape, self.chunks, self.initializing, 0)  # The extra zero is for future use as a version flag

	def __setstate__(self, state):
		"""Restore state from the unpickled state values."""
		self.dtype, self.jagged, self.dims, self.shape, self.chunks, self.initializing, _ = state

	def __len__(self) -> int:
		if self.rank > 0:
			return self.shape[0]
		return 0

	@property
	def rank(self) -> int:
		return len(self.dims)

	@property
	def bytewidth(self) -> int:
		if self.dtype in ("bool", "uint8", "int8"):
			return 1
		elif self.dtype in ("uint16", "int16", "float16"):
			return 2
		elif self.dtype in ("uint32", "int32", "float32"):
			return 3
		elif self.dtype in ("uint64", "int64", "float64"):
			return 4
		return -1

	def _fancy_indexing(self, expr: FancyIndex) -> Tuple["shoji.Filter", ...]:
		if isinstance(expr, tuple):
			fancyindex: Tuple[FancyIndexElement, ...] = expr
		else:
			fancyindex = (expr,)
	
		# Fill in missing axes with : just like numpy does
		if any(isinstance(x, type(...)) for x in fancyindex):  # We can't use a simple "if ... in fancyindex" because fancyindex may contain numpy arrays which complain about ==
			ix = fancyindex.index(...)
			fancyindex = fancyindex[:ix] + (slice(None),) * (self.rank - len(fancyindex) - 1) + fancyindex[ix + 1:]

		if len(fancyindex) < self.rank:
			fancyindex += (slice(None),) * (self.rank - len(fancyindex))

		filters: List[shoji.Filter] = []
		for axis, (dim, fi) in enumerate(zip(self.dims, fancyindex)):
			# Maybe it's a Filter?
			if isinstance(fi, shoji.Filter):
				if isinstance(dim, str) and isinstance(fi.dim, str) and fi.dim != dim:
					raise IndexError(f"Tensor dimension '{dim}' cannot be indexed uing filter expression '{fi}' with dimension '{fi.dim}'")
				filters.append(fi)
			elif isinstance(fi, slice):
				filters.append(shoji.TensorSliceFilter(self, fi, axis))
			elif isinstance(fi, (int, np.int64, np.int32)):
				filters.append(shoji.TensorIndicesFilter(self, np.array(fi), axis))
			elif isinstance(fi, np.ndarray):
				if np.issubdtype(fi.dtype, np.bool_):
					filters.append(shoji.TensorBoolFilter(self, fi, axis))
				elif np.issubdtype(fi.dtype, np.int_):
					filters.append(shoji.TensorIndicesFilter(self, fi, axis))
			else:
				raise KeyError()
		return tuple(filters)

	def __getitem__(self, expr: FancyIndex) -> np.ndarray:
		assert self.wsm is not None, "Tensor is not bound to a database"
		return shoji.View(self.wsm, self._fancy_indexing(expr))[self.name]

	def __setitem__(self, expr: FancyIndex, vals: np.ndarray) -> None:
		assert self.wsm is not None, "Tensor is not bound to a database"
		shoji.View(self.wsm, self._fancy_indexing(expr))[self.name] = vals

	def numpy_dtype(self) -> str:
		if self.dtype == "string":
			return "object"
		return self.dtype

	def python_dtype(self) -> Callable:
		if self.dtype == "string":
			return str
		if self.dtype == "bool":
			return bool
		if self.dtype in ("float16", "float32", "float64"):
			return float
		return int

	def _compare(self, operator, other) -> "shoji.Filter":
		if isinstance(other, Tensor):
			return shoji.TensorFilter(operator, self, other)
		elif isinstance(other, (str, int, float, bool)):
			return shoji.ConstFilter(operator, self, other)
		elif isinstance(other, np.integer):
			return shoji.ConstFilter(operator, self, int(other))
		elif isinstance(other, np.float):
			return shoji.ConstFilter(operator, self, float(other))
		elif isinstance(other, np.object):
			return shoji.ConstFilter(operator, self, str(other))
		elif isinstance(other, np.bool):
			return shoji.ConstFilter(operator, self, bool(other))
		else:
			raise TypeError("Invalid operands for expression")

	def __eq__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare("==", other)
		
	def __ne__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare("!=", other)

	def __gt__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare(">", other)

	def __lt__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare("<", other)

	def __ge__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare(">=", other)

	def __le__(self, other) -> "shoji.Filter":  # type: ignore
		return self._compare("<=", other)

	def append(self, vals: Union[List[np.ndarray], np.ndarray], axis: int = 0) -> None:
		assert self.wsm is not None, "Cannot append to unbound tensor"
		
		if self.rank == 0:
			raise ValueError("Cannot append to a scalar")

		tv = TensorValue(vals)
		shoji.io.append_values_multibatch(self.wsm, [self.name], [tv], (axis,))

	def _quick_look(self) -> str:
		if self.rank == 0:
			if self.dtype == "string":
				s = f'"{self[:]}"'
			else:
				s = str(self[:])
			if len(s) > 60:
				return s[:56] + " ..."
			return s

		def look(vals) -> str:
			s = "["
			if not isinstance(vals, list) and vals.ndim == 1:
				if self.dtype == "string":
					s += ", ".join([f'"{x}"' for x in vals[:5]])
				else:
					s += ", ".join([str(x) for x in vals[:5]])
			else:
				elms = []
				for val in vals[:5]:
					elms.append(look(val))
				s += ", ".join(elms)
			if len(vals) > 5:
				s += ", ...]"
			else:
				s += "]"
			return s

		s = look(self[:10])
		if len(s) > 60:
			return s[:56] + " ···"
		return s



	def __repr__(self) -> str:
		return f"<Tensor {self.name} dtype='{self.dtype}' dims={self.dims}, shape={self.shape}, chunks={self.chunks}>"
