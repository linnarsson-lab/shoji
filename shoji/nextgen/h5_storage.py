import h5py
import numpy as np
from typing import Tuple, Union, Iterator
from .storage import Storage, Path, Dimension, Tensor, FancyIndices
from pytypes import typechecked, override


@typechecked
class H5Storage(Storage):
	def __init__(self, filename: str, mode: str) -> None:
		super().__init__()
		self.h5 = h5py.File(filename, mode)

	@override
	def _create_workspace(self, path: Path) -> Path:
		self.h5.create_group(str(path))
		return path

	@override
	def _list_workspaces(self, path: Path) -> Iterator[Path]:
		for item in self.h5[str(path)]:
			fullpath = path / item
			if isinstance(self.h5[str(fullpath)], h5py.Group):
				yield fullpath

	@override
	def _delete_workspace(self, path: Path) -> None:
		if str(path) not in self.h5 or not isinstance(self.h5[str(path)], h5py.Group):
			raise ValueError(f"Workspace '{path}' does not exist")
		del self.h5[str(path)]

	@override
	def _create_dimension(self, path: Path, shape: int) -> Dimension:
		# Dimensions are stored as attributes on the Group, prefixed by "@dim"
		parent = str(path.parent())
		name = path.name_
		if parent in self.h5 and isinstance(self.h5[parent], h5py.Group):
			if "@dim" + name in self.h5[parent].attrs:
				raise ValueError(f"Dimension '{name}' already exists in workspace '{parent}'")
			self.h5[parent].attrs["@dim" + name] = shape
			return Dimension(self, path, shape)
		raise ValueError(f"Workspace '{parent}' does not exist")

	@override
	def _get_dimension(self, path: Path) -> "Dimension":
		parent = str(path.parent())
		name = path.name_
		if parent in self.h5 and isinstance(self.h5[parent], h5py.Group):
			if "@dim" + name in self.h5[parent].attrs:
				return Dimension(self, path, self.h5[parent].attrs["@dim" + name])
			else:
				raise ValueError(f"Dimension '{name}' does not exist in workspace '{parent}'")
		raise ValueError(f"Workspace '{parent}' does not exist")
		
	@override
	def _list_dimensions(self, path: Path) -> Iterator[Dimension]:
		if str(path) in self.h5 and isinstance(self.h5[str(path)], h5py.Group):
			for attr in self.h5[str(path)].attrs.keys():
				if attr.startswith("@dim"):
					yield Dimension(self, path / attr[4:], self.h5[path].attrs[attr])
		else:
			raise ValueError(f"Workspace '{path}' does not exist")

	@override
	def _resize_dimension(self, path: Path, shape: int) -> Dimension:
		parent = str(path.parent())
		name = path.name_
		if parent in self.h5 and isinstance(self.h5[parent], h5py.Group):
			if "@dim" + name not in self.h5[parent].attrs:
				raise ValueError(f"Dimension '{name}' does not exist in workspace '{parent}'")
			self.h5[parent].attrs["@dim" + name] = shape
		raise ValueError(f"Workspace '{parent}' does not exist")

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
		
	@override
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
