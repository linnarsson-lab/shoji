"""
Tensors, representing multidimensional arrays of numbers or strings.
"""
from typing import Tuple, Union, List, Optional, Callable
import numpy as np
import shoji


class Tensor:
	valid_types = ("bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "string")

	def __init__(self, dtype: str, dims: Union[Tuple[Union[None, int, str], ...]], inits: Union[List[np.ndarray], np.ndarray] = None, shape: Tuple[int, ...] = None) -> None:
		"""
		Args:
			dtype:	string giving the datatype of the tensor elements
			dims:	A tuple of None, int, string (empty tuple designates a scalar)
			inits:	Optional values to initialize the tensor with

		Remarks:
			Dimensions are specified as:

				None:		resizable/jagged anonymous dimension
				int:		fixed-shape anonymous dimension
				string:		named dimension
			
			The first dimension can be fixed-shape or resizable; all other dimensions can be fixed-shape or jagged. 
		"""
		self.dtype = dtype
		# Check that the type is valid
		if dtype not in Tensor.valid_types:
			raise TypeError(f"Invalid Tensor type {dtype}")

		self.dims = dims
	
		self.name = ""  # Will be set if the Tensor is read from the db
		self.wsm: Optional[shoji.WorkspaceManager] = None  # Will be set if the Tensor is read from the db

		self.inits = inits
		if self.inits is None:
			self.jagged = False
			self.shape = shape if shape is not None else (0,) * len(dims)
		else:
			# If scalar, convert to an ndarray scalar which will have shape ()
			if np.isscalar(self.inits):
				self.inits = np.array(self.inits, dtype=self.numpy_dtype())
			# If not jagged, then it's a regular ndarray
			if isinstance(self.inits, np.ndarray):
				self.shape = self.inits.shape
				self.jagged = False
				self._check_type(self.inits)
			else: # It's jagged: a list of ndarrays, possibly of different shapes
				assert isinstance(self.inits, list)
				self.jagged = True
				shp: List[int] = list(self.inits[0].shape)
				for i, array in enumerate(self.inits):
					assert isinstance(array, np.ndarray), "All rows of jagged array must be numpy ndarrays"
					self._check_type(array)
					if len(array.shape) != len(shp):
						raise ValueError(f"Rank mismatch: shape {array.shape} of subarray at row {i} is not the same rank as shape {shp} at row 0")
					for ix in range(len(shp)):
						if shp[ix] != array.shape[ix]:
							shp[ix] = 0
				self.shape = tuple([len(self.inits)] + shp)

			if len(self.dims) != len(self.shape):
				raise ValueError(f"Rank mismatch: shape {self.dims} declared is not the same rank as shape {self.shape} of values")

		for ix, dim in enumerate(self.dims):
			if dim is not None and not isinstance(dim, int) and not isinstance(dim, str):
				raise ValueError(f"Dimension {ix} '{dim}' is invalid (must be None, int or str)")

			if isinstance(dim, int) and self.inits is not None:
				if self.shape[ix] != dim:  # type: ignore
					raise IndexError(f"Mismatch between the declared shape {dim} of dimension {ix} and the inferred shape {self.shape} of values")
		self.rank = len(self.dims)

	def __len__(self) -> int:
		if self.rank > 0:
			return self.shape[0]
		return 0

	def __getitem__(self, expr: Union["shoji.Filter", tuple, slice, int]) -> np.ndarray:
		assert self.wsm is not None, "Tensor is not bound to a database"
		# Maybe it's a Filter, or a tuple of Filters?
		if isinstance(expr, shoji.Filter):
			return shoji.View(self.wsm, (expr,))[self.name]
		elif isinstance(expr, tuple) and isinstance(expr[0], shoji.Filter):
			return shoji.View(self.wsm, expr)[self.name]
		# Or a slice?
		if isinstance(expr, slice):
			if expr.start is None and expr.stop is None:
				return shoji.View(self.wsm, ())[self.name]
			return shoji.View(self.wsm, (shoji.TensorSliceFilter(self, expr),))[self.name]
		if isinstance(expr, (list, tuple, int)):
			expr = np.array(expr)
		if isinstance(expr, np.ndarray):
			if np.issubdtype(expr.dtype, np.bool_):
				return shoji.View(self.wsm, (shoji.TensorBoolFilter(self, expr),))[self.name]
			elif np.issubdtype(expr.dtype, np.int_):
				return shoji.View(self.wsm, (shoji.TensorIndicesFilter(self, expr),))[self.name]
		raise KeyError(expr)

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
		elif isinstance(other, str) or isinstance(other, int) or isinstance(other, float) or isinstance(other, bool):
			return shoji.ConstFilter(operator, self, other)
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

	def _check_type(self, inits) -> None:
		# Check that the type is exactly right
		if self.dtype != inits.dtype.name:
			if self.dtype != "string":
				raise TypeError("Tensor dtype and inits dtype do not match")
			if (self.dtype == "string" and inits.dtype.name != "object") or (self.dtype == "string" and inits.dtype.name == "object" and not isinstance(inits.flat[0], str)):
				raise TypeError("string tensors must be initialized with ndarray dtype 'object', where objects are strings")

	def _quick_look(self) -> str:
		if self.rank == 0:
			return self[:]
		elif self.rank == 1:
			s = "[" + ", ".join([str(x) for x in self[:5]])
			if len(self) > 5:
				s += ", ...]"
			else:
				s += "]"
			return s
		elif self.rank == 2:
			result = "["
			elms = []
			for x in self[:5]:
				s = "["
				s += ", ".join([str(x) for x in x[:5]])
				if len(x) > 5:
					s += ", ...]"
				else:
					s += "]"
				elms.append(s)
			result += ", ".join(elms)
			if len(self) > 5:
				result += ", ...]"
			else:
				result += "]"
			return result
		else:
			return f"(rank={self.rank})"

	def __repr__(self) -> str:
		return f"<Tensor dtype='{self.dtype}' dims={self.dims}, shape={self.shape}>"
