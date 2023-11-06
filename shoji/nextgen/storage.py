import re
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, List, Iterator
import numpy as np
from numpy.typing import NDArray
from pytypes import typechecked


FancyIndices = Tuple[Union[NDArray[np.integer], NDArray[np.bool_], slice, int], ...]


@typechecked
class Path:
	"""
	Represents an absolute path attached to a Storage, but where the path may or may not actually correspond to an entity in the Storage
	"""
	def __init__(self, storage: "Storage", path: Union[str, "Path"]) -> None:
		self.storage = storage
		self._pathstring: str = ""
		if isinstance(path, Path):
			self._pathstring = path._pathstring
		else:
			self._pathstring = path  # Now this must be a string
		self._ensure_legal_absolute_path(self._pathstring)

	def _ensure_legal_absolute_path(self, path) -> None:
		if "//" in path:
			raise ValueError("Empty path segments (e.g. '/foo//bar') are not allowed")
		if not path.startswith("/"):
			raise ValueError("Only absolute path is allowed, which must begin with a leading slash ('/')")
		for name in self.segments_():
			self._ensure_legal_entity_name(name)

	def _ensure_legal_entity_name(self, name) -> None:
		if not re.match("[_a-zA-Z][_a-zA-Z0-9]*", name):
			raise ValueError(f"Invalid name '{name}' (only alphanumeric and underscore allowed, and name cannot start with a number)")

	def __truediv__(self, other: str) -> "Path":
		self._ensure_legal_entity_name(other)
		return Path(self.storage, self._pathstring + "/" + other)

	@property
	def name_(self) -> str:
		return self.segments_()[-1]

	def segments_(self) -> List[str]:
		return list(filter(None, self._pathstring.split("/")))

	def parent_(self) -> "Path":
		segments = self.segments_()
		if len(segments) > 0:
			return Path(self.storage, "/".join(segments))
		raise IOError("Cannot take parent of root workspace")

	def exists_(self) -> bool:
		return self.storage._get_entity_type(self) is not None

	def is_workspace_(self) -> bool:
		return isinstance(self.storage._get_entity_type(self), Path)

	def is_dimension_(self) -> bool:
		return isinstance(self.storage._get_entity_type(self), Dimension)

	def is_tensor_(self) -> bool:
		return isinstance(self.storage._get_entity_type(self), Tensor)

	def __dir__(self) -> List[str]:
		workspace_names = [ws.name_ for ws in self.storage._list_workspaces(self)]
		dimension_names = [d.name for d in self.storage._list_dimensions(self)]
		tensor_names = [t.name for t in self.storage._list_tensors(self)]
		self_fields = list(object.__dir__(self))
		return workspace_names + dimension_names + tensor_names + self_fields

	def __iter__(self):
		for w in self.storage._list_workspaces(self):
			yield w
		for d in self.storage._list_dimensions(self):
			yield d
		for t in self.storage._list_tensors(self):
			yield t

	def __contains__(self, name: str) -> bool:
		return (self / name).exists_()

	def __str__(self) -> str:
		return self._pathstring

class Tensor:
	def __init__(self, storage: "Storage", path: str, name: str, dtype: str, dims: Tuple[Union[int, str], ...], *, chunks: Optional[Tuple[int, ...]] = None, allow_index: bool = True) -> None:
		self.storage = storage
		self.path = path
		self.name = name
		self.dtype = "bool"
		self.dims: Tuple[Union[int, str], ...] = ()
		self.chunks: Optional[Tuple[int, ...]] = None
		self.allow_index = True


class Dimension:
	def __init__(self, storage: "Storage", path: str, name: str, length: int) -> None:
		self.storage = storage
		self.path = path
		self.name = name
		self.length = length

@typechecked
class Storage(ABC):
	def __init__(self) -> None:
		pass

	# When called, all parents exist (they are first created in order if not)
	@abstractmethod
	def _create_workspace(self, path: Path) -> "Path":
		pass

	def create_workspace(self, path: Union[str, Path]) -> "Path":
		path = Path(self, path)

		partial = Path(self, "/")
		for name in path.segments_():
			partial /= name
			tp = self._get_entity_type(partial)
			if tp is None:
				self._create_workspace(partial)
			elif not isinstance(tp, Path):
				raise ValueError(f"Cannot create new workspace '{path}' because '{partial}' is not a workspace")
		return path

	@abstractmethod
	def _list_workspaces(self, path: Path) -> Iterator["Path"]:
		pass

	def list_workspaces(self, path: Union[str, Path]) -> Iterator["Path"]:
		path = Path(self, path)
		for item in self._list_workspaces(path):
			yield item

	# Must override in concrete implementation, to perform the actual work of deleting a workspace
	# When called, path is valid but workspace or its parents may not exist
	# The workspace and all its contents, including any child workspaces, should be deleted
	@abstractmethod
	def _delete_workspace(self, path: Path) -> None:
		pass

	def delete_workspace(self, path: Union[str, Path]) -> None:
		path = Path(self, path)
		self._delete_workspace(path)


	# Must return a type (Path, Dimension or Tensor), or None if the path does not exist
	@abstractmethod
	def _get_entity_type(self, path: Path) -> Optional[Union[Path, Dimension, Tensor]]:
		pass

	def get_entity_type(self, path: Union[str, Path]) ->  Optional[Union[Path, Dimension, Tensor]]:
		path = Path(self, path)
		return self._get_entity_type(path)

	# Must override in concrete implementation, to perform the actual work of creating a dimension
	# When called, path and name are valid but workspace or its parents may not exist
	# If dimension already exists, an exception should be raised
	@abstractmethod
	def _create_dimension(self, path: Path, length: int) -> "Dimension":
		pass

	def create_dimension(self, path: Union[str, Path], length: int) -> "Dimension":
		path = Path(self, path)
		if length < 0:
			raise ValueError(f"Shape {length} of dimension '{path.name_}' cannot be negative")
		return self._create_dimension(path, length)

	# Must override in concrete implementation, to perform the actual work of getting a dimension
	# When called, path and name are valid but workspace or its parents may not exist
	@abstractmethod
	def _get_dimension(self, path: Path) -> "Dimension":
		pass

	def get_dimension(self, path: Union[str, Path]) -> "Dimension":
		path = Path(self, path)
		return self._get_dimension(path)

	# Must override in concrete implementation, to perform the actual work of getting a dimension
	# When called, path and name are valid but workspace or its parents may not exist
	@abstractmethod
	def _list_dimensions(self, path: Path) -> Iterator["Dimension"]:
		pass

	def list_dimensions(self, path: Union[str, Path]) -> Iterator["Dimension"]:
		path = Path(self, path)
		for item in self._list_dimensions(path):
			yield item


	# Must override in concrete implementation, to perform the actual work of updating a dimension
	# When called, path and name are valid but workspace or its parents may not exist
	# If dimension doesn't exist, an exception should be raised
	# After resizing the dimension, all tensors using the dimension should henceforth conform to the new shape
	@abstractmethod
	def _resize_dimension(self, path: Path, length: int) -> "Dimension":
		pass

	def resize_dimension(self, path: Union[str, Path], length: int) -> "Dimension":
		path = Path(self, path)
		if length is not None and length < 0:
			raise ValueError(f"Shape {length} of dimension '{path.name_}' cannot be negative")
		return self._resize_dimension(path, length)


	# Must override in concrete implementation, to perform the actual work of deleting a dimension
	# When called, path and name are valid but workspace or its parents may not exist
	# If dimension doesn't exist, an exception should NOT be raised (just return silently)
	# If any tensors use the dimension, an exception should be thrown (the tensors must be deleted first)
	@abstractmethod
	def _delete_dimension(self, path: Path) -> None:
		pass

	def delete_dimension(self, path: Union[str, Path]) -> None:
		path = Path(self, path)
		self._delete_dimension(path)

	# Must override in concrete implementation, to perform the actual work of creating a tensor
	# When called, path, dims and chunks are valid but dimensions, workspace or its parents may not exist
	# If the tensor already exists, throw an exception
	# The tensor should be initially filled with the default value of dtype (but this need not necessarily be physically written to the backing store)
	@abstractmethod
	def _create_tensor(self, path: Path, dtype: str, dims: Tuple[Union[int, str], ...], *, chunks: Tuple[int, ...] = None, allow_index: bool = True) -> "Tensor":
		pass

	def create_tensor(self, path: Union[str, Path], dtype: str, dims: Tuple[Union[int, str], ...], *, chunks: Tuple[int, ...] = None, allow_index: bool = True) -> "Tensor":
		path = Path(self, path)
		for dim in dims:
			assert dim is not None, "Dimension cannot be None"
			if isinstance(dim, str):
				_ = Path(self, path.parent_() / dim)  # This is just to ensure the dimension makes a valid path
			elif isinstance(dim, int):
				assert dim >= 0, "Dimension shape cannot be negative"
		if chunks is not None:
			assert len(chunks) == len(dims), "Chunks tuple must match number of dims"
			for chunk in chunks:
				assert isinstance(chunk, int), "Chunks must be integers"
				assert chunk > 0, "Chunks must be greater than zero"
		return self._create_tensor(path, dtype, dims, chunks=chunks, allow_index=allow_index)

	# Must override in concrete implementation, to perform the actual work of getting a tensor
	# When called, path and name are valid but workspace or its parents may not exist
	@abstractmethod
	def _get_tensor(self, path: Path) -> "Tensor":
		pass

	def get_tensor(self, path: Union[str, Path]) -> "Tensor":
		path = Path(self, path)
		return self._get_tensor(path)

	# Must override in concrete implementation, to perform the actual work of listing tensors
	# When called, path and name are valid but workspace or its parents may not exist
	@abstractmethod
	def _list_tensors(self, path: Path) -> List["Tensor"]:
		pass

	def list_tensors(self, path: Union[str, Path]) -> Iterator["Tensor"]:
		path = Path(self, path)
		for item in self._list_tensors(path):
			yield item

	# Must override in concrete implementation, to perform the actual work of resizing a tensor
	# When called, path, name, and dims are valid but dimensions, workspace or its parents may not exist
	# If the tensor does not exist, throw an exception
	# Resizing should always succeed; if the new shape is larger along some dimension, the new regions should be filled by the default value of the dtype
	# The space used by shrinking a dimension may or may not be reclaimed
	def _resize_tensor(self, path: Path, dims: Tuple[Union[int, str], ...]) -> None:
		pass

	def resize_tensor(self, path: Union[str, Path], dims: Tuple[Union[int, str], ...]) -> None:
		path = Path(self, path)
		for dim in dims:
			assert dim is not None, "Dimension cannot be None"
			if isinstance(dim, str):
				_ = Path(self, path.parent_() / dim)  # This is just to ensure the dimension makes a valid path
			elif isinstance(dim, int):
				assert dim >= 0, "Dimension shape cannot be negative"
		self._resize_tensor(path, dims)

	# Must override in concrete implementation, to perform the actual work of deleting a tensor
	# When called, path and name are valid but workspace or its parents may not exist, and name may not be a tensor
	# If the tensor does not exist, do not throw an exception (just return silently)
	# The space used by the tensor may or may not be reclaimed
	def _delete_tensor(self, path: Path) -> None:
		pass

	def delete_tensor(self, path: Union[str, Path]) -> None:
		path = Path(self, path)
		self._delete_tensor(path)


	# Must override in concrete implementation, to perform the actual work of reading from a tensor
	# When called, path and name are valid but workspace or its parents may not exist, and name may not be a tensor
	# If the tensor does not exist, throw an exception
	# If any indices are outside the tensor shape, raise an exception
	# The implementation should read from the tensor, then extand or truncate the resulting np.ndarray so as to conform to the tensor shape 
	@abstractmethod
	def _read_tensor(self, path: Path, indices: FancyIndices) -> np.ndarray:
		pass

	# Read from the tensor by indexing along each dimension
	# Indices must be arrays of integers, and must be ordered without duplicates
	def read_tensor(self, path: Union[str, Path], indices: FancyIndices) -> np.ndarray:
		path = Path(self, path)
		return self._read_tensor(path, indices)

	# Must override in concrete implementation, to perform the actual work of writing to a tensor
	# When called, path and name are valid but workspace or its parents may not exist, and name may not be a tensor
	# If the tensor does not exist, throw an exception
	# If any indices are outside the tensor shape, raise an exception
	# If the data dtype is different from the tensor dtype, raise an exception (do not silently cast)
	# The implementation should write to the tensor
	@abstractmethod
	def _write_tensor(self, path: Path, indices: FancyIndices, data: np.ndarray) -> None:
		pass

	# Read from the tensor by indexing along each dimension
	# Indices must be arrays of integers, and must be ordered without duplicates
	def write_tensor(self, path: Union[str, Path], indices: FancyIndices, data: np.ndarray) -> None:
		path = Path(self, path)
		return self._write_tensor(path, indices, data)
