"""
Workspaces let you organise collections of data that belong together. Workspaces can be nested,
like folders in a file system.

A workspace contains Tensors and Dimensions that form a coherent dataset. Tensors can only use
dimensions that reside in the same workspace, and the implicit constraints imposed by the use
of common dimensions exist only within a single workspace. 

A Workspace is a bit like a collection of tables in a relational database, or a collection of
Pandas DataFrames. Each dimension in a workspace corresponds to a table, and each tensor with
that dimension as its first dimension corresponds to a column in the table.

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
import shoji


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
	def __init__(self, db: fdb.impl.Database, subspace: fdb.directory_impl.DirectorySubspace, path: Union[Tuple, Tuple[str]]) -> None:
		self._db = db
		self._subspace = subspace
		self._path = path

	def _move_to(self, new_path: Union[str, Tuple[str]]) -> None:
		if not isinstance(new_path, tuple):
			new_path = (new_path,)
		self._subspace = self._subspace._move_to(self._db.transaction, ("shoji",) + new_path)
		self._path = new_path
		
	def _create(self, path: Union[str, Tuple[str]]) -> "WorkspaceManager":
		if not isinstance(path, tuple):
			path = (path,)
		subspace = self._subspace.create(self._db.transaction, path)
		return WorkspaceManager(self._db.transaction, subspace, self._path + path)

	def __dir__(self) -> List[str]:
		dimensions = [self._subspace["dimensions"].unpack(k)[0] for k,v in self._db.transaction[self._subspace["dimensions"].range()]]
		tensors = [self._subspace["tensors"].unpack(k)[0] for k,v in self._db.transaction[self._subspace["tensors"].range()]]
		return self._subspace.list(self._db.transaction) + dimensions + tensors + object.__dir__(self)

	def __contains__(self, name: str) -> bool:
		return shoji.io.read_entity(self._db.transaction, self, name) != None

	def __getattr__(self, name: str) -> Union["WorkspaceManager", shoji.Dimension, shoji.Tensor]:
		result = shoji.io.read_entity(self._db.transaction, self, name)
		if result is None:
			raise AttributeError(name)
		else:
			return result

	def __getitem__(self, expr: Union[str, "shoji.Filter", slice]) -> Union["WorkspaceManager", shoji.Dimension, "shoji.View", shoji.Tensor]:
		# Try to read an attribute on the object
		if isinstance(expr, str):
			return self.__getattr__(expr)
		# Maybe it's a Filter, or a tuple of Filters?
		if isinstance(expr, shoji.Filter):
			return shoji.View(self, (expr,))
		elif isinstance(expr, tuple) and isinstance(expr[0], shoji.Filter):
			return shoji.View(self, expr)
		# Or a slice?
		if isinstance(expr, slice) and expr.start is None and expr.stop is None:
			return shoji.View(self, ())
		raise IndexError()

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
			with shoji.Transaction(self):
				shoji.io.create_tensor(self._db.transaction, self, name, tensor)
				# Check all the dimension constraints
				for i, tdim in enumerate(tensor.dims):
					if isinstance(tensor.dims[0], str):
						dim = self[tensor.dims[i]]
						assert isinstance(dim, shoji.Dimension)
						if dim.shape is not None:
							if tensor.inits is None:
								raise ValueError(f"Tensor '{name}' with fixed-shape dimension {i} ('{tensor.dims[i]}') must be initialized when created")
							else:
								if tensor.jagged:
									for row in tensor.inits:
										if row.shape[i] != dim.shape:
											raise ValueError(f"Tensor '{name}' dimension {i} ('{tensor.dims[i]}') must be exactly {dim.shape} elements long")
								elif dim.shape != tensor.inits.shape[i]:  # type: ignore
									raise ValueError(f"Tensor '{name}' dimension {i} ('{tensor.dims[i]}') must be exactly {dim.shape} elements long")
				if tensor.inits is not None:
					if tensor.rank > 0 and isinstance(tensor.dims[0], str) and self[tensor.dims[0]].length == 0:
						shoji.io.append_tensors(self._db.transaction, self, tensor.dims[0], {name: tensor.inits})
					else:
						shoji.io.write_tensor_values(self._db.transaction, self, name, tensor)
		else:
			super().__setattr__(name, value)
	
	def __setitem__(self, name: str, value: Any) -> None:
		self.__setattr__(name, value)

	def __delattr__(self, name: str) -> None:
		shoji.io.delete_entity(self._db.transaction, self, name)

	def __repr__(self) -> str:
		subspaces = self._subspace.list(self._db.transaction)
		dimensions = [self._subspace["dimensions"].unpack(k.key)[0] for k in self._db.transaction[self._subspace["dimensions"].range()]]
		tensors = [self._subspace["tensors"].unpack(k.key)[0] for k in self._db.transaction[self._subspace["tensors"].range()]]
		s = f"Workspace with {len(subspaces)} subspaces, {len(dimensions)} dimensions and {len(tensors)} tensors:"
		for sub in subspaces:
			s += f"\n  {sub} <Workspace>" 
		for dname in dimensions:
			s += f"\n  {dname} {self[dname]}"
		for tname in tensors:
			s += f"\n  {tname} {self[tname]}"
		return s
