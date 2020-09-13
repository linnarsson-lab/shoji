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
from typing import Any, Tuple, Union, List
import fdb
import numpy as np
import logging
from tqdm import trange
import loompy
import shoji
import timeit


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

	def _dimensions(self) -> List[str]:
		return [self._subdir["dimensions"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["dimensions"].range()]]

	def _tensors(self) -> List[str]:
		return [self._subdir["tensors"].unpack(k.key)[0] for k in self._db.transaction[self._subdir["tensors"].range()]]

	def __dir__(self) -> List[str]:
		dimensions = [self._subdir["dimensions"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["dimensions"].range()]]
		tensors = [self._subdir["tensors"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["tensors"].range()]]
		return self._subdir.list(self._db.transaction) + dimensions + tensors + object.__dir__(self)

	def __iter__(self):
		for s in self._subdir.list(self._db.transaction):
			yield self[s]
		dimensions = [self._subdir["dimensions"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["dimensions"].range()]]
		for d in dimensions:
			yield self[d]
		tensors = [self._subdir["tensors"].unpack(k)[0] for k,v in self._db.transaction[self._subdir["tensors"].range()]]
		for t in tensors:
			yield self[t]

	def __contains__(self, name: str) -> bool:
		return shoji.io.get_entity(self._db.transaction, self, name) is not None

	def __getattr__(self, name: str) -> Union["WorkspaceManager", shoji.Dimension, shoji.Tensor]:
		if name.startswith("_"):  # Jupyter calls this method with names like "__wrapped__" and we want to avoid a futile database roundtrip
			return super().__getattribute__(name)
		result = shoji.io.get_entity(self._db.transaction, self, name)
		if result is None:
			return super().__getattribute__(name)
		else:
			return result

	def __getitem__(self, expr: Union[str, "shoji.Filter", slice]) -> Union["WorkspaceManager", shoji.Dimension, "shoji.View", shoji.Tensor]:
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
		if isinstance(value, Workspace):
			self._create(name)
		elif isinstance(value, shoji.Dimension):
			# Check that the first letter is lowercase
			if not name[0].islower():
				raise AttributeError("Dimension name must begin with a lowercase letter")
			shoji.io.create_or_update_dimension(self._db.transaction, self, name, value)
		elif isinstance(value, shoji.Tensor):
			tensor = value
			# Check that the first letter is uppercase
			if not name[0].isupper():
				raise AttributeError("Tensor name must begin with an uppercase letter")

			# Note, this can fail as follows:
			#  * Before anything has been written
			#  * After the tensor has been full created but no values have been written
			#  * After one or more rows have been fully written (consistent with other tensors in the dimension)
			#  * After all rows have been fully written
			# In each case, the database state will be consistent
			shoji.io.create_tensor(self._db.transaction, self, name, tensor)
			# if tensor.inits is not None:
			# 	dim = None
			# 	if tensor.rank > 0 and isinstance(tensor.dims[0], str):
			# 		dim = shoji.io.get_dimension(self._db.transaction, self, tensor.dims[0])
			# 		if dim is not None:
			# 			# This ensures that all dimension constraints are properly checked
			# 			shoji.io.append_tensors(self._db.transaction, self, tensor.dims[0], {name: tensor.inits.values})
			# 			return
			# 	shoji.io.write_tensor_values(self._db.transaction, self, name, tensor.inits)
		else:
			super().__setattr__(name, value)
	
	def __setitem__(self, name: str, value: Any) -> None:
		self.__setattr__(name, value)

	def __delattr__(self, name: str) -> None:
		shoji.io.delete_entity(self._db.transaction, self, name)

	def __delitem__(self, name: str) -> None:
		shoji.io.delete_entity(self._db.transaction, self, name)

	def _from_loom(self, f: str, layers: List[str] = None, verbose: bool = False, dimension_names: Tuple[str, str] = None) -> None:
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
			self[genes_dim].append(d)
			self[genes_dim] = shoji.Dimension(shape=ds.shape[0])  # Set to a fixed shape to avoid jagged arrays below
			
			for key, vals in ds.ca.items():
				dtype = ds.ca[key].dtype.name
				name = fix_name(key, cells_dim, ds.ra.keys() + ds.layers.keys() + ds.attrs.keys())
				dims = (cells_dim, ) + vals.shape[1:]
				self[name] = shoji.Tensor("string" if dtype == "object" else dtype, dims=dims)

			if verbose:
				logging.info("Loading column attributes and matrix layers")
			STEP = 2000
			for i in trange(0, ds.shape[1], STEP):
				d = {}
				for key in ds.ca.keys():
					name = fix_name(key, cells_dim, ds.ra.keys() + ds.layers.keys() + ds.attrs.keys())
					d[name] = ds.ca[key][i:i + STEP]
				for key in layers:
					name = "Expression" if key == "" else key
					name = fix_name(name, "layer", ds.ra.keys() + ds.ca.keys() + ds.attrs.keys())
					dtype = ds.layers[key].dtype.name
					if i == 0:
						self[name] = shoji.Tensor("string" if dtype == "object" else dtype, (cells_dim, genes_dim))
					d[name] = ds.layers[key][:, i:i + STEP].T
				self[cells_dim].append(d)
			
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