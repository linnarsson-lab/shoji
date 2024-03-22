import scanpy as sc
import numpy as np
from typing import Tuple, Union, Iterator
from .storage import Storage, Path, Dimension, Tensor
from pytypes import typechecked, override


@typechecked
class AnnDataStorage(Storage):
	def __init__(self, source: Union[str, sc.AnnData]) -> None:
		super().__init__()
		if isinstance(source, sc.AnnData):
			self.h5ad = source
		else:
			self.h5ad = sc.read_h5ad(source)

	@override
	def _create_workspace(self, path: Path) -> Path:
		raise NotImplementedError("Workspaces are not supported in .h5ad storage")

	@override
	def _list_workspaces(self, path: Path) -> Iterator[Path]:
		if not path.isroot():
			raise NotImplementedError("Workspaces are not supported in .h5ad storage")

	@override
	def _delete_workspace(self, path: Path) -> None:
		raise NotImplementedError("Workspaces are not supported in .h5ad storage")

	@override
	def _create_dimension(self, path: Path, shape: int) -> Dimension:
		raise NotImplementedError("Cannot create new dimensions in .h5ad storage")

	@override
	def _get_dimension(self, path: Path) -> "Dimension":
		segments = path.segments()
		if len(segments) != 2:
			raise NotImplementedError("Workspaces are not supported in .h5ad storage")
		if segments[1] == "cells":
			return Dimension(self, path, "cells", self.h5ad.shape[0])
		elif segments[1] == "genes":
			return Dimension(self, path, "genes", self.h5ad.shape[1])
		else:
			raise NotImplementedError("Only built-in 'cells' and 'genes' dimensions are supported in .h5ad storage")
		
	@override
	def _list_dimensions(self, path: Path) -> Iterator[Dimension]:
		if not path.isroot():
			raise NotImplementedError("Workspaces are not supported in .h5ad storage")

		for i, name in enumerate("cells", "genes"):
			yield Dimension(self, path, name, self.h5ad.shape[i])

	@override
	def _resize_dimension(self, path: Path, shape: int) -> Dimension:
		raise NotImplementedError("Resizing is not supported in .h5ad storage")

	@override
	def _delete_dimension(self, path: Path) -> None:
		raise NotImplementedError("Dimensions cannot be deleted in .h5ad storage")

	@override
	def _create_tensor(self, path: Path, dtype: str, dims: Tuple[Union[int, str], ...], *, chunks: Tuple[int, ...] = None, allow_index: bool = True) -> Tensor:
		parent = path.parent()

		shape = []
		for dim in dims:
			if isinstance(dim, int):
				shape.append(dim)
			else:
				shape.append(self._get_dimension(parent / dim).length)

		if parent.name_ in self.h5ad and isinstance(self.h5ad[parent.name_], h5py.Group):
			ds = self.h5ad.create_dataset(path, shape, dtype, chunks=chunks)
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
		if str(path) not in self.h5ad or not isinstance(self.h5ad[str(path)], h5py.Dataset):
			raise ValueError(f"Tensor '{path.name_}' does not exist")

		ds = self.h5ad[str(path)]
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
		for item in self.h5ad[str(path)]:
			fullpath = path / item
			if isinstance(self.h5ad[str(fullpath)], h5py.Dataset):
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
