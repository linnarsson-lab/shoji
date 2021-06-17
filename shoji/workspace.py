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

You can also use brackets to create a workspace with a name defined by an expression:

```python
name = "Hello" + "World"
db[name] = shoji.Workspace()
```

You can list workspaces and check for the existence of a workspace in the usual ways:

```python
for ws in db:
	... # do something with the workspace object

if "samples" in db:
	... # the workspace existed
```

You can list the contents of a workspace:

```python
ws._workspaces()  # Returns a list of names of sub-workspaces
ws._tensors()     # Returns a list of names of tensors in the workspace
ws._dimensions()  # Returns a list of names of dimensions in the workspace
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
from shoji.io import Compartment
import h5py
import pickle


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

	def _move_to(self, new_path: Union[str, Tuple[str, ...]]) -> None:
		if isinstance(new_path, str):
			new_path = tuple(new_path.split("."))
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
		return [self._subdir[Compartment.Dimensions].unpack(k.key)[0] for k in self._db.transaction[self._subdir[Compartment.Dimensions].range()]]

	def _get_dimension(self, name: str) -> shoji.Dimension:
		dim = self[name]
		assert isinstance(dim, shoji.Dimension), f"'{name}' is not a dimension"
		return dim

	def _tensors(self, include_not_ready: bool = False) -> List[str]:
		names = [self._subdir[Compartment.Tensors].unpack(k.key)[0] for k in self._db.transaction[self._subdir[Compartment.Tensors].range()]]
		if include_not_ready:
			return names
		return [name for name in names if shoji.io.get_tensor(self._db.transaction, self, name) is not None]

	def _get_tensor(self, name: str, include_initializing: bool = False) -> shoji.Tensor:
		tensor = shoji.io.get_tensor(self._db.transaction, self, name, include_initializing=include_initializing)
		assert isinstance(tensor, shoji.Tensor), f"'{name}' is not a tensor"
		return tensor

	def __dir__(self) -> List[str]:
		dimensions = [self._subdir[Compartment.Dimensions].unpack(k)[0] for k,v in self._db.transaction[self._subdir[Compartment.Dimensions].range()]]
		tensors = [self._subdir[Compartment.Tensors].unpack(k)[0] for k,v in self._db.transaction[self._subdir[Compartment.Tensors].range()]]
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
		elif isinstance(expr, tuple) and isinstance(expr[0], shoji.Filter):
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
			name = name.capitalize()
			if not name[0].isupper():
				name = "X_" + name
			name = name.replace(".", "_")
			return name

		if dimension_names is None:
			dimension_names = ("genes", "cells")
		genes_dim = dimension_names[0]
		cells_dim = dimension_names[1]
		with loompy.connect(f, validate=False) as ds:
			if layers is None:
				layers = list(ds.layers.keys())
			self[genes_dim] = shoji.Dimension(shape=ds.shape[0])
			self[cells_dim] = shoji.Dimension(shape=ds.shape[1])

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
				self[name] = shoji.Tensor("string" if dtype == "object" else dtype, val.shape, inits=val)

			if verbose:
				logging.info("Loading row attributes")
			for key, vals in ds.ra.items():
				dtype = vals.dtype.name
				dtype = "string" if dtype == "object" else dtype
				name = fix_name(key, genes_dim, ds.ca.keys() + ds.layers.keys() + ds.attrs.keys())
				dims = (genes_dim, ) + vals.shape[1:]
				self[name] = shoji.Tensor(dtype, dims, inits=ds.ra[key])

			if verbose:
				logging.info("Loading column attributes")
			for key, vals in ds.ca.items():
				dtype = ds.ca[key].dtype.name
				dtype = "string" if dtype == "object" else dtype
				name = fix_name(key, cells_dim, ds.ra.keys() + ds.layers.keys() + ds.attrs.keys())
				dims = (cells_dim,) + vals.shape[1:]
				self[name] = shoji.Tensor(dtype, dims, inits=ds.ca[key])

			if verbose:
				logging.info("Loading layers")
			for key in layers:
				name = "Expression" if key == "" else key
				name = fix_name(name, "layer", ds.ra.keys() + ds.ca.keys() + ds.attrs.keys())
				dtype = ds.layers[key].dtype.name
				dtype = "string" if dtype == "object" else dtype
				if name == "Expression" and fix_expression_dtype:
					dtype = "uint16"
				self[name] = shoji.Tensor(dtype, (cells_dim, genes_dim), inits=ds.layers[key][:, :].T.astype(dtype))
			
	def __repr__(self) -> str:
		subdirs = self._workspaces()
		dimensions = [self._subdir[Compartment.Dimensions].unpack(k.key)[0] for k in self._db.transaction[self._subdir[Compartment.Dimensions].range()]]
		tensors = [self._subdir[Compartment.Tensors].unpack(k.key)[0] for k in self._db.transaction[self._subdir[Compartment.Tensors].range()]]
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
						# if t.dims[i] is None:
						# 	shps.append("__")
						# elif isinstance(t.dims[i], str) and self[t.dims[i]].shape is None:
						# 	shps.append("__")
						if i == 0 and t.jagged:
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

	def _import(self, f: str):
		"""
		Import a previously exported workspace

		Args:
			f		The file name (full path)
		"""

		h5 = h5py.File(f, "r")
		group = h5.require_group("shoji")
		for att in group.attrs:
			if att.startswith("Dimension$"):
				shape = group.attrs[att]
				if shape == -1:
					shape = None
				self[att[10:]] = shoji.Dimension(shape=shape)

		for tname in group:
			if tname.startswith("Tensor$"):
				data = group[tname][:]
				tensor = pickle.loads(group[tname])
				if tensor.dtype == "string":
					data = data.astype("object")
				tname = tname[7:]
				self[tname] = shoji.Tensor(tensor.dtype, tensor.dims, chunks=tensor.chunks, jagged=tensor.jagged, inits=data)

		h5.close()

	def _export(self, f: str):
		"""
		Export the workspace to an HDF5 file

		Args:
			f			The file name (full path)

		Remarks:
			If the file does not exist, it will be created
		"""
		h5 = h5py.File(f, "a")
		group = h5.require_group("shoji")

		for dname in self._dimensions():
			dim = self._get_dimension(dname)
			group.attrs["Dimension$" + dname] = (dim.shape if dim.shape is not None else -1, dim.length)

		for tname in self._tensors():
			tensor = self._get_tensor(tname)
			if tensor.jagged:
				logging.warning(f"Skipping '{tname}' because jagged tensors are not yet supported for export")
				continue
			group.attrs["Tensor$" + tname] = pickle.dumps(tensor)
			data: np.ndarray = self[tname][:]
			if tensor.dtype == "string":
				data = data.astype(np.string_)
			group.create_dataset(tname, data=data, compression="gzip")

		h5.close()
