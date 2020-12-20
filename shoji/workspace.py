"""
Workspaces let you organise collections of data that belong together. Workspaces can be nested,
like folders in a file system.

A workspace contains Tensors and Dimensions that form a coherent dataset. Tensors can only use
dimensions that reside in the same workspace, and the implicit constraints imposed by the use
of common dimensions exist only within a single workspace. 

A Workspace is a bit like a collection of tables in a relational database, or a collection of
Pandas DataFrames. Each dimension in a workspace corresponds to a table, and each tensor with
that dimension as its first dimension corresponds to a column in the table. However, workspaces
can also contain tensors that link two or more dimensions.

For example, a simple table of samples, with columns for sample ID, name, age and description
could be modelled as follows:

```python
import shoji
db = shoji.connect()
db.cancer_project = shoji.Workspace()  # Create a new workspace
ws = db.cancer_project

ws.samples = shoji.Dimension(shape=None)  # Create a variable-length dimension
ws.SampleID = shoji.Tensor("uint32", ("samples",))  # Create a vector of sample IDs
ws.SampleName = shoji.Tensor("string", ("samples",))  # Create a vector of sample names
ws.Age = shoji.Tensor("uint32", ("samples",))  # Create a vector of ages
ws.Description = shoji.Tensor("string", ("samples",))  # Create a vector of descriptions
```

## Relationships

Relationships between tensors are established through shared dimensions. For example, we can add a
dimension that describes a set of protein measurements, and a 2D tensor that contains protein
measurements for each sample:

```python
# ...continued from above
ws.proteins = shoji.Dimension(shape=10)  # A fixed-length dimension (10 elements)
ws.ProteinName = shoji.Tensor("string", ("proteins",))
ws.ProteinData = shoji.Tensor("float32", ("samples", "proteins"))
```

In the above example, `ProteinData` is a tensor of protein measurements for each sample.
The shape of the tensor is `(n_samples, 10)`.

## Constraints

Within a workspace, the tensors that share a dimension are constrained to have the same number of
elements along that dimension. If a dimension is variable-length, elements can be added, but must
be added to all the tensors that share that dimension. Note that elements can be added only along
the first dimension of a tensor.

To add data while enforcing constraints, use the `append` method on the `shoji.dimension.Dimension`.

## Managing workspaces

When you connect to the Shoji database, you are connected to the root workspace:

```python
import shoji
db = shoji.connect()  # db is a shoji.Workspace object representing the root
```

Workspaces are created by assigning a newly created workspace object to a name on an existing
workspace. This allows you to create nested workspaces:

```python
db.scRNA = shoji.Workspace()
db.scRNA.analysis_20200601 = shoji.Workspace()  # Create a sub-workspace
```

You can delete a workspace using the `del` statement:

```python
del db.scRNA
```

**WARNING**: Deleting a workspace takes effect immediately and without confirmation. All sub-workspaces and all tensors and dimensions that they contain are deleted. The action cannot be undone.
"""
from typing import Any, Tuple, Union, List, Dict
import fdb
import os
import numpy as np
import logging
import loompy
import shoji
import shoji.io
import h5py
from copy import copy


class Workspace:
	"""
	Class representing a new Workspace. Use this to create new workspaces in Shoji.
	"""
	def __init__(self) -> None:
		pass


class WorkspaceManager:
	"""
	Class for managing workspaces. You should not create WorkspaceManager objects yourself.
	"""
	def __init__(self, db: fdb.impl.Database, subdir: fdb.directory_impl.DirectorySubspace, path: Union[Tuple, Tuple[str, ...]]) -> None:
		self._db = db
		self._subdir = subdir
		self._path = path
		self._name: str = ""

	def _move_to(self, new_path: Union[str, Tuple[str]]) -> None:
		if not isinstance(new_path, tuple):
			new_path = (new_path,)
		self._subdir = self._subdir.move_to(self._db.transaction, ("shoji",) + new_path)
		self._path = new_path
		
	def _create(self, path: Union[str, Tuple[str]]) -> "WorkspaceManager":
		if not isinstance(path, tuple):
			path = (path,)
		if self._subdir.exists(self._db.transaction, path):
			raise IOError(f"Workspace '{'/'.join(path)}' already exists")
		subdir = self._subdir.create(self._db.transaction, path)
		return WorkspaceManager(self._db.transaction, subdir, self._path + path)

	def _workspaces(self) -> List[str]:
		return self._subdir.list(self._db.transaction)

	def _get_workspace(self, name: str) -> "WorkspaceManager":
		ws = self[name]
		assert isinstance(ws, shoji.WorkspaceManager), f"'{name}' is not a workspace"
		return ws

	def _dimensions(self) -> List[str]:
		return [self._subdir["dimensions"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["dimensions"].range()]]

	def _get_dimension(self, name: str) -> shoji.Dimension:
		dim = self[name]
		assert isinstance(dim, shoji.Dimension), f"'{name}' is not a dimension"
		return dim

	def _tensors(self, include_not_ready: bool = False) -> List[str]:
		names = [self._subdir["tensors"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["tensors"].range()]]
		if include_not_ready:
			return names
		return [name for name in names if shoji.io.get_tensor(self._db.transaction, self, name) is not None]

	def _get_tensor(self, name: str, row: int = -1) -> shoji.Tensor:
		tensor = shoji.io.get_tensor(self._db.transaction, self, name, row)
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a tensor"
		return tensor

	def __dir__(self) -> List[str]:
		dimensions = [self._subdir["dimensions"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["dimensions"].range()]]
		tensors = [self._subdir["tensors"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["tensors"].range()]]
		return self._subdir.list(self._db.transaction) + dimensions + tensors + object.__dir__(self)

	def __iter__(self):
		for w in shoji.io.list_workspaces(self._db.transaction, self):
			yield w
		for t in shoji.io.list_tensors(self._db.transaction, self):
			yield t
		for d in shoji.io.list_dimensions(self._db.transaction, self):
			yield d

	def __contains__(self, name: str) -> bool:
		entity = shoji.io.get_entity(self._db.transaction, self, name)
		if entity is not None:
			return True
		parts = name.split(".")
		entity = shoji.io.get_entity(self._db.transaction, self, parts[0])
		if entity is None:
			return False
		if len(parts) == 1:
			return True
		else:
			if isinstance(entity, shoji.WorkspaceManager):
				return entity.__contains__(".".join(parts[1:]))
			else:
				raise ValueError("First part of a multi-part name must be a workspace")

	def __getattr__(self, name: str) -> Union["WorkspaceManager", shoji.Dimension, shoji.Tensor]:
		if name.startswith("_"):  # Jupyter calls this method with names like "__wrapped__" and we want to avoid a futile database roundtrip
			return super().__getattribute__(name)
		entity = shoji.io.get_entity(self._db.transaction, self, name)
		if entity is not None:
			return entity
		# The name could be a multi-part expression like x.y.z
		parts = name.split(".")
		entity = shoji.io.get_entity(self._db.transaction, self, parts[0])
		if entity is None:
			return super().__getattribute__(name)
		if len(parts) == 1:
			return entity
		else:
			if isinstance(entity, shoji.WorkspaceManager):
				return entity.__getattr__(".".join(parts[1:]))
			else:
				raise ValueError("First part of a multi-part name must be a workspace")

	def __getitem__(self, expr: Union[str, "shoji.Filter", slice]) -> Union["WorkspaceManager", shoji.Dimension, "shoji.View", "shoji.NonTransactionalView", shoji.Tensor]:
		# Try to read an attribute on the object
		if isinstance(expr, str):
			return self.__getattr__(expr)
		# Perhaps it's a view already (e.g. a slice of a dimension)
		if isinstance(expr, shoji.View):
			return expr
		# Maybe it's a Filter, or a tuple of Filters?
		if isinstance(expr, shoji.Filter):
			return shoji.View(self, (expr,))
		elif expr == ...:
			return shoji.NonTransactionalView(shoji.View(self, ()))
		elif isinstance(expr, tuple) and isinstance(expr[0], shoji.Filter):
			if len(expr) > 1 and expr[-1] == ...:
				return shoji.NonTransactionalView(shoji.View(self, expr[:-1]))
			return shoji.View(self, expr)
		# Or a slice?
		if isinstance(expr, slice):
			if expr.start is None and expr.stop is None:
				return shoji.View(self, ())
			else:
				raise KeyError("Cannot slice workspace directly (use a slice on a dimension instead)")
		raise KeyError(f"Invalid key '{expr}' (only filter expression or : allowed)")

	def __setattr__(self, name: str, value: Any) -> None:
		if "." in name:
			raise AttributeError(f"Invalid name '{name}' (names cannot contain periods (.))")
		if isinstance(value, Workspace):
			if name in self:
				raise AttributeError(f"Cannot overwrite existing entity with new workspace {name}")
			self._create(name)
		elif isinstance(value, shoji.Dimension):
			# Check that the first letter is lowercase
			if not name[0].islower():
				raise AttributeError("Dimension name must begin with a lowercase letter")
			shoji.io.create_dimension(self._db.transaction, self, name, value)
		elif isinstance(value, shoji.Tensor):
			tensor = value
			# Check that the first letter is uppercase
			if not name[0].isupper():
				raise AttributeError("Tensor name must begin with an uppercase letter")

			if name in self:
				if isinstance(self[name], shoji.Tensor):
					del self[name]
				else:
					raise AttributeError(f"Cannot create new tensor '{name}' because it would overwrite existing entity")
			shoji.io.create_tensor(self._db.transaction, self, name, tensor)
			shoji.io.initialize_tensor(self, name, tensor)
		elif isinstance(value, shoji.WorkspaceManager):
			raise ValueError("Cannot assign WorkspaceManager object to workspace (did you mean to use Workspace object?")
		else:
			super().__setattr__(name, value)
	
	def __setitem__(self, name: str, value: Any) -> None:
		self.__setattr__(name, value)

	def __delattr__(self, name: str) -> None:
		shoji.io.delete_entity(self._db.transaction, self, name)

	def __delitem__(self, name: str) -> None:
		shoji.io.delete_entity(self._db.transaction, self, name)

	def _from_loom(self, f: str, layers: List[str] = None, verbose: bool = False, dimension_names: Tuple[str, str] = None, *, fix_expression_dtype: bool = False) -> None:
		"""
		Load a loom files into a workspace

		Args:
			f						Filename (full path)
			layers					Layers to load, or None to load all layers
			verbose					If true, log progress
			dimension_names			2-tuple of strings to use as dimension names for (rows, cols), or None to use ("genes", "cells")
			fix_expression_dtype	If true, fix a legacy mistake in some loom files where the main matrix is float32 but should be uint16
		"""

		def fix_name(name, suffix, other_names):
			if name in other_names:
				name += "_" + suffix
			name = name[0].upper() + name[1:]
			if not name[0].isupper():
				name = "X_" + name
			return name

		if dimension_names is None:
			dimension_names = ("genes", "cells")
		genes_dim = dimension_names[0]
		cells_dim = dimension_names[1]
		with loompy.connect(f, validate=False) as ds:
			if layers is None:
				layers = list(ds.layers.keys())
			self[genes_dim] = shoji.Dimension(shape=None)
			self[cells_dim] = shoji.Dimension(shape=None)

			if verbose:
				logging.info("Loading global attributes")

			for key, val in ds.attrs.items():
				if not isinstance(val, np.ndarray):
					val = np.array(val)
				dtype = val.dtype.name
				if dtype.startswith("str"):
					dtype = "string"
					val = val.astype("object")
				name = fix_name(key, "global", ds.ca.keys() + ds.ra.keys() + ds.layers.keys())
				self[name] = shoji.Tensor("string" if dtype == "object" else dtype, val.shape, val)

			if verbose:
				logging.info("Loading row attributes")
			d = {}
			for key, vals in ds.ra.items():
				dtype = vals.dtype.name
				name = fix_name(key, genes_dim, ds.ca.keys() + ds.layers.keys() + ds.attrs.keys())
				d[name] = ds.ra[key]
				dims = (genes_dim, ) + vals.shape[1:]
				self[name] = shoji.Tensor("string" if dtype == "object" else dtype, dims=dims)
			self._get_dimension(genes_dim).append(d)
			self[genes_dim] = shoji.Dimension(shape=ds.shape[0])  # Set to a fixed shape to avoid jagged arrays below
			
			for key, vals in ds.ca.items():
				dtype = ds.ca[key].dtype.name
				name = fix_name(key, cells_dim, ds.ra.keys() + ds.layers.keys() + ds.attrs.keys())
				dims = (cells_dim, ) + vals.shape[1:]
				self[name] = shoji.Tensor("string" if dtype == "object" else dtype, dims=dims)

			if verbose:
				logging.info("Loading column attributes and matrix layers")
			STEP = 2000
			for i in range(0, ds.shape[1], STEP):
				d = {}
				for key in ds.ca.keys():
					name = fix_name(key, cells_dim, ds.ra.keys() + ds.layers.keys() + ds.attrs.keys())
					d[name] = ds.ca[key][i:i + STEP]
				for key in layers:
					name = "Expression" if key == "" else key
					name = fix_name(name, "layer", ds.ra.keys() + ds.ca.keys() + ds.attrs.keys())
					dtype = ds.layers[key].dtype.name
					if name == "Expression" and fix_expression_dtype:
						dtype = "uint16"
					if i == 0:
						self[name] = shoji.Tensor("string" if dtype == "object" else dtype, (cells_dim, genes_dim))
					d[name] = ds.layers[key][:, i:i + STEP].T
					if name == "Expression" and fix_expression_dtype:
						d[name] = d[name].astype("uint16")
				self._get_dimension(cells_dim).append(d)

	def _import(self, f: str, recursive: bool = False, group_name: str = "/"):
		"""
		Import a previously exported workspace

		Args:
			f		The file name (full path)
			recursive	If true, sub-workspaces will also be imported
			group_name	The name of the HDF5 group from which to import
		"""

		# TODO: rewrite this to load from HDF5 not the workspace
		h5 = h5py.File(f, "r")
		group = h5.require_group(group_name)
		for dname in self._dimensions():
			dim = self._get_dimension(dname)
			group.attrs["Dimension$" + dname] = (dim.shape if dim.shape is not None else -1, dim.length)

		tensors: Dict[str, shoji.Tensor] = {}
		for tname in self._tensors():
			tensor = self._get_tensor(tname)
			group.attrs["Tensor$" + tname] = np.array((tensor.dtype, tensor.rank, 1 if tensor.jagged else 0) + tensor.dims + tensor.shape, dtype=object)
			group.create_dataset(
				tname,
				tensor.shape,
				tensor.dtype # TODO: get this right
			)
			if tensor.rank > 0 and isinstance(tensor.dims[0], str):
				tensors[tensor.dims[0]] = tensor
			else:
				# Read the whole tensor
				data = tensor[:]		
		ds: h5py.Dataset = group[tname]
		n_rows_per_read = 1000
		ix = 0
		while True:
			try:
				data = tensor[ix: ix + n_rows_per_read]
				ds[ix: ix + n_rows_per_read] = data
			except fdb.impl.FDBError as e:
				if e.code == 1007:
					n_rows_per_read = max(1, n_rows_per_read // 2)
					continue
				raise e

		if recursive:
			for wsname in self._workspaces():
				self._get_workspace(wsname)._export(f, True, os.path.join(group_name, wsname))		
		h5.close()

	def _export(self, f: str, recursive: bool = False, group_name: str = "/"):
		"""
		Export the workspace to an HDF5 file

		Args:
			f			The file name (full path)
			recursive	If true, sub-workspaces will be exported to HDF5 sub-groups
			group_name	The name of the HDF5 group where the workspace should be stored

		Remarks:
			If the file does not exist, it will be created
		"""
		h5 = h5py.File(f, "a")
		group = h5.require_group(group_name)

		for dname in self._dimensions():
			dim = self._get_dimension(dname)
			group.attrs["Dimension$" + dname] = (dim.shape if dim.shape is not None else -1, dim.length)

		for tname in self._tensors():
			tensor = self._get_tensor(tname)
			group.attrs["Tensor$" + tname] = np.array((tensor.dtype, tensor.rank, 1 if tensor.jagged else 0) + tensor.dims + tensor.shape, dtype=object)
			group.create_dataset(tname, tensor.shape, tensor.dtype)
			ds: h5py.Dataset = group[tname]
			# Now read/write the dataset
			n_rows_per_read = 1000
			ix = 0
			while True:
				try:
					data = tensor[ix: ix + n_rows_per_read]
					ds[ix: ix + n_rows_per_read] = data
				except fdb.impl.FDBError as e:
					if e.code == 1007:
						n_rows_per_read = max(1, n_rows_per_read // 2)
						continue
					raise e

		if recursive:
			for wsname in self._workspaces():
				self._get_workspace(wsname)._export(f, True, os.path.join(group_name, wsname))
		
		h5.close()
			
	def __repr__(self) -> str:
		subdirs = self._workspaces()
		dimensions = [self._subdir["dimensions"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["dimensions"].range()]]
		tensors = [self._subdir["tensors"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["tensors"].range()]]
		s = f"Workspace with {len(subdirs)} workspaces, {len(dimensions)} dimensions and {len(tensors)} tensors:"
		for sub in subdirs:
			s += f"\n  {sub} <Workspace>" 
		for dname in dimensions:
			s += f"\n  {dname} {self[dname]}"
		for tname in tensors:
			s += f"\n  {tname} {self[tname]}"
		return s

	def _repr_html_(self):
		if len(self._path) == 0:
			s = f"<h4>(root) (shoji.Workspace)</h4>"
		else:
			s = f"<h4>{self._name} (shoji.Workspace)</h4>"
		
		subdirs = self._workspaces()
		if len(subdirs) > 0:
			s += f"<h5>Sub-workspaces</h5>"
			s += "<table><tr><th></th><th>Contents</th></tr>"
			for wsname in subdirs:
				ws = self[wsname]
				s += "<tr>"
				n_subdirs = len(ws._workspaces())
				n_dimensions = len(ws._dimensions())
				n_tensors = len(ws._tensors())
				s += f"<td align='left'><strong>{ws._name}</strong></td><td>{n_subdirs} workspaces, {n_dimensions} dimensions, {n_tensors} tensors</td>"
				s += "</tr>"
			s += "</table>"
	
		dimensions = self._dimensions()
		if len(dimensions) > 0:
			s += f"<h5>Dimensions</h5>"
			s += "<table><tr><th></th><th>shape</th><th>length</th></tr>"
			for dname in dimensions:
				dim = self[dname]
				s += "<tr>"
				s += f"<td align='left'><strong>{dim.name}</strong></td>"
				s += f"<td>{dim.shape:,}</td>" if dim.shape is not None else "<td>None</td>"
				s += f"<td>{dim.length:,}</td>"
				s += "</tr>"
			s += "</table>"

		tensors = self._tensors()
		if len(tensors) > 0:
			s += f"<h5>Tensors</h5>"
			s += "<table><tr><th></th><th>dtype</th><th>rank</th><th>dims</th><th>shape</th><th>(values)</th></tr>"
			for tname in tensors:
				t = self[tname]
				s += "<tr>"
				s += f"<td align='left'><strong>{t.name}</strong></td>"
				s += f"<td align='left'>{t.dtype}</td>"
				s += f"<td align='left'>{t.rank}</td>"
				if t.rank > 0:
					s += "<td>" + " ✕ ".join([(str(s) if s is not None else "__") for s in t.dims]) + "</td>"
					shps = []
					for i, shp in enumerate(t.shape):
						if t.dims[i] is None:
							shps.append("__")
						elif isinstance(t.dims[i], str) and self[t.dims[i]].shape is None:
							shps.append("__")
						else:
							shps.append("{:,}".format(shp))
					s += "<td>" + " ✕ ".join(shps) + "</td>"
				else:
					s += "<td>()</td>"
					s += "<td>()</td>"
				s += f"<td>{t._quick_look()}</td>"
				s += "</tr>"
			s += "</table>"
		return s