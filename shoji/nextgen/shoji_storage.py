import h5py
import numpy as np
from typing import Tuple, Union, Iterator, Optional, List, Any
from .storage import Storage, Path, Dimension, Tensor, FancyIndices
from pytypes import typechecked, override
import fdb
from ..io.enums import Compartment
import pickle


@fdb.transactional
def delete_workspace(tr, root: Any, path: Path) -> None:
	subdir = root.subspace(path.segments_())
	tr.clear_range_startswith(subdir.key())

@fdb.transactional
def list_workspaces(tr, root: Any, path: Path) -> List[Path]:
	subdir = root.subspace(path.segments_())
	return [Path(p) for p in subdir.list(tr)]

@fdb.transactional
def create_dimension(tr, root: Any, path: Path, length: int) -> None:
	path, name = path.split_()
	subdir = root.subspace(path.segments_())
	tr[subdir[Compartment.Dimensions][name]] = length.to_bytes(length=8, byteorder="big")

@fdb.transactional
def get_dimension(tr, root: Any, path: Path) -> int:
	path, name = path.split_()
	subdir = root.subspace(path.segments_())
	length = tr[subdir[Compartment.Dimensions][name]]
	if length.present():
		return int.from_bytes(length, byteorder="big")
	else:
		raise KeyError(f"Dimension {path} does not exist")

@fdb.transactional
def list_dimensions(tr, root: Any, path: Path) -> Tuple[List[str], List[int]]:
	path, name = path.split_()
	subdir = root.subspace(path.segments_())
	names = subdir[Compartment.Dimensions][name].list(tr)
	lens = [get_dimension(name) for name in names]
	return names, lens

@fdb.transactional
def delete_dimension(tr, root: Any, path: Path, name: str) -> None:
	subdir = root.subspace(path.segments_())
	tr.clear_range_startswith(subdir[Compartment.Dimensions][name].key())

@typechecked
class ShojiStorage(Storage):
	def __init__(self, cluster_file: Optional[str] = None, event_model = None, transaction_retry_limit = 1) -> None:
		super().__init__()
		self.db = fdb.open(cluster_file=cluster_file, event_model=event_model)
		self.db.transaction = self.db  # default to using the Database object for transactions
		self.db.options.set_transaction_retry_limit(transaction_retry_limit)  # Retry every transaction only once if it doesn't go through
		self.root = fdb.directory.create_or_open(self.db, ("shoji_nextgen",))

	@override
	def _create_workspace(self, path: Path) -> Path:
		self.root.create(self.db.transaction, tuple(path.segments_()))
		return path

	@override
	def _list_workspaces(self, path: Path) -> List[Path]:
		return list_workspaces(self.root, path)

	@override
	def _delete_workspace(self, path: Path) -> None:
		delete_workspace(self.root, path)

	@override
	def _create_dimension(self, path: Path, length: int) -> None:
		create_dimension(self.root, path, length)

	@override
	def _get_dimension(self, path: Path) -> "Dimension":
		parent = str(path.parent())
		name = path.name_
		length = get_dimension(self.root, path)
		return Dimension(self, parent, name, length)
		
	@override
	def _list_dimensions(self, path: Path) -> List[Dimension]:
		names, lens = list_dimensions(self.root, path)
		return [Dimension(self, path, name, length) for (name, length) in zip(names, lens)]

	@override
	def _resize_dimension(self, path: Path, length: int) -> Dimension:
		return self._create_dimension(path, length)

	@override
	def _delete_dimension(self, path: Path) -> None:
		parent = str(path.parent())
		name = path.name_
		if parent in self.h5 and isinstance(self.h5[parent], h5py.Group):
			if "@dim" + name not in self.h5[parent].attrs:
				raise ValueError(f"Dimension '{name}' does not exist in workspace '{parent}'")
			# Make sure none of the tensors in the group use the dimension
			in_use_by = []
			for tensor in self._list_tensors(parent):
				for dim in tensor.dims:
					if name == dim:
						in_use_by.append(tensor.name)
			if len(in_use_by) > 0:
				raise ValueError(f"Cannot delete dimension '{name}' because it is in use by tensors: {', '.join(in_use_by)}")
			del self.h5[parent].attrs["@dim" + name]
		raise ValueError(f"Workspace '{parent}' does not exist")

	@override
	def _create_tensor(self, path: Path, dtype: str, dims: Tuple[Union[int, str], ...], *, chunks: Tuple[int, ...] = None, allow_index: bool = True) -> Tensor:
		parent = path.parent()

		shape = []
		for dim in dims:
			if isinstance(dim, int):
				shape.append(dim)
			else:
				shape.append(self._get_dimension(parent / dim).length)

		if parent.name_ in self.h5 and isinstance(self.h5[parent.name_], h5py.Group):
			ds = self.h5.create_dataset(path, shape, dtype, chunks=chunks)
			# Dimension specification is stored as attributes on the tensor, prefixed by @dim
			for i, dim in enumerate(dims):
				ds.attrs[f"@dim{i}"] = dim
				if isinstance(dim, str):
					self._get_dimension(parent / dim)  # Make sure the dimension exists
			return Tensor(self, path, dtype, dims, chunks=chunks, allow_index=allow_index)
		else:
			raise ValueError(f"Workspace '{path}' does not exist")
		
	def _get_tensor_dataset(self, path: Path) -> Tuple[h5py.Dataset, Tensor]:
		if str(path) not in self.h5 or not isinstance(self.h5[str(path)], h5py.Dataset):
			raise ValueError(f"Tensor '{path.name_}' does not exist")

		ds = self.h5[str(path)]
		if ds.ndim > 0 and "@dim0" in ds.attrs:
			dims = [ds.attrs[f"@dim{i}"] for i in range(ds.ndim)]
		else:
			dims = ds.shape
		return ds, Tensor(self, path, ds.dtype, dims, chunks=ds.chunks, allow_index=True)

	@override
	def _get_tensor(self, path: Path) -> "Tensor":
		_, tensor = self._get_tensor_dataset(path)
		return tensor

	@override
	def _list_tensors(self, path: Path) -> Iterator["Tensor"]:
		for item in self.h5[str(path)]:
			fullpath = path / item
			if isinstance(self.h5[str(fullpath)], h5py.Dataset):
				yield self._get_tensor(fullpath)

	@override
	def _resize_tensor(self, path: Path, newdims: Tuple[Union[int, str], ...]) -> None:
		ds, _ = self._get_tensor_dataset(path)
		shape = []
		for i, dim in enumerate(newdims):
			ds.attrs[f"@dim{i}"] = dim
			if isinstance(dim, str):
				shape.append(self._get_dimension(path.parent_() / dim).length)  # Make sure the dimension exists
			else:
				shape.append(dim)
		ds.resize(shape)

	@override
	def _delete_tensor(self, path: Path) -> None:
		ds, _ = self._get_tensor_dataset(path)
		del ds

	@override
	def _read_tensor(self, path: Path, indices: FancyIndices) -> np.ndarray:
		ds, _ = self._get_tensor_dataset(path)
		return ds[indices]

	@override
	def _write_tensor(self, path: Path, indices: FancyIndices, data: np.ndarray) -> None:
		ds, _ = self._get_tensor_dataset(path)
		# TODO: validate size of data
		ds[indices] = data
