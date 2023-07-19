"""
Views let you work with a selected subset of a workspace. Reading from the view automatically
returns values from the selected subset of the database. Values written to the view
are automatically written to the corresponding subset of the database.

Views are *selections of elements* along one or more dimensions. 

View are created by filter expressions (see `shoji.filter`) on workspaces or dimensions. Once you 
have obtained a view, you can read from it just like you would from the workspace itself:

```python
view = ws.scRNA.cells[:1000]  # A view of the first 1000 rows along the cells dimension
ages = view.Age				  # The first 1000 values of the Age tensor, as np.ndarray
```

You can also write to the underlying workspace by assigning values to a view:

```python
new_ages = np.array(...)	  # A numpy array of values
view = ws.scRNA.cells[::2]    # A view of every other row along the cells dimension
view.Age = new_ages           # The corresponding rows in the underlying tensor are updated
```

Assigning values in this way is an atomic operation (it will either succeed or fail
completely), and is subject to the [size and time limits](file:///Users/stelin/shoji/html/shoji/index.html#limitations) of shoji transactions.

You can create a view that selects rows along more than one dimension, by providing two 
or more filter expressions separated by comma:

```python
ws = db.scRNA
ws.cells = shoji.Dimension(shape=None)
ws.genes = shoji.Dimension(shape=31768)
ws.Age = shoji.Tensor("string", ("cells",))
ws.Chromosome = shoji.Tensor("string", ("genes",))
# Slice both dimensions:
view = ws.scRNA[ws.Age > 10, ws.Chromosome == "chr1"]
```

## Grouping

Calling `.groupby(labels)` on a view creates a `shoji.groupby.GroupViewBy` object. 
The `labels` should be the name of a tensor which is used to group the view.
Calling an aggregation function such as `mean()` then returns the view but with
all elements that share the same value of the `label` replaced by their mean.

For example, suppose you have an Expression tensor with dimensions ("genes", "cells") and
a ClusterID tensor with dimension ("cells",). Suppose there are 10 distinct values of
ClusterID. The following code will return an np.ndarray of shape ("genes", 10) containing 
mean values for each ClusterID:

```python
grouped = ws[:].groupby("ClusterID")
(labels, values) = grouped.mean("Expression")
# labels is a list of distinct ClusterID values
# values is a np.ndarray where the "cells" dimension is replaced by the distinct cluster IDs
```


"""
from typing import Tuple, Callable, Union, List, Literal
import shoji
import shoji.io
from shoji.io import Compartment
import numpy as np
import scipy.sparse as sparse

class View:
	def __init__(self, wsm: shoji.WorkspaceManager, filters: Tuple[shoji.Filter, ...]) -> None:
		super().__setattr__("filters", {f.dim: f for f in filters})
		super().__setattr__("wsm", wsm)

	def groupby(self, labels: Union[str, np.ndarray], projection: Callable = None) -> shoji.GroupViewBy:
		return shoji.GroupViewBy(self, labels, projection)

	def get_length(self, dim: str) -> int:
		if dim in self.filters:
			return len(self.filters[dim].get_rows(self.wsm))
		else:
			return self.wsm._get_dimension(dim).length

	def get_shape(self, tensor: shoji.Tensor) -> Tuple[int, ...]:
		shape = []
		for i, dim in enumerate(tensor.dims):
			if isinstance(dim, str):
				shape.append(self.get_length(dim))
			else:
				shape.append(tensor.shape[i])
		return tuple(shape)

	def _read_batch(self, tensor: shoji.Tensor, start: int, end: int) -> np.ndarray:
		if tensor.jagged:
			raise ValueError(f"Cannot read batches of jagged tensor '{tensor.name}'")
		indices = []
		for i, dim in enumerate(tensor.dims):
			if dim in self.filters:
				indices.append(self.filters[dim].get_rows(self.wsm))
			else:
				indices.append(np.arange(tensor.shape[i]))
			if i == 0:
				indices[0] = indices[0][start:end]

		# Read the tensor (selected rows)
		result = shoji.io.read_at_indices(self.wsm, tensor.name, indices, tensor.chunks, False)
		return result

	def __getattr__(self, name: str) -> np.ndarray:
		return shoji.io.read_filtered(self.wsm, name, self.filters.values())

	def __getitem__(self, expr: Union[str, slice]) -> np.ndarray:
		if isinstance(expr, slice):
			raise KeyError("Cannot slice a view (not implemented)")
		return self.__getattr__(expr)

	def __setattr__(self, name: str, vals: Union[List[np.ndarray], np.ndarray]) -> None:
		return shoji.io.write_filtered(self.wsm._db, self.wsm, name, vals, self.filters.values())

	def __setitem__(self, name: str, vals: np.ndarray) -> None:
		return self.__setattr__(name, vals)

	def anndata(self, 
		 *,
	     X: str = "Expression",
		 var: Union[Literal["auto"], Tuple[str]] = "auto",
		 obs: Union[Literal["auto"], Tuple[str]] = (),
		 uns: Union[Literal["auto"], Tuple[str]] = (),
		 varm: Union[Literal["auto"], Tuple[str]] = (),
		 obsm: Union[Literal["auto"], Tuple[str]] = (),
		 var_key: str = "Accession",
	     obs_key: str = "CellID",
	     layers: Union[Literal["auto"], Tuple[str]] = ()
		 ):
		"""
		Create an anndata object by collecting tensors according to the specification

		Args:
			X		The main expression matrix, usually "Expression" (which is the default)
			var		Rank-1 tensors along the genes dimension, e.g. ("Gene", "Chromosome", "Start"); "auto" collects all rank-1 tensors on the genes dimension
			obs		Rank-1 tensors along the cells dimension, e.g. ("Tissue", "TotalUMI"); "auto" collects all rank-1 tensors on the cells dimension
			varm	Rank 2 tensors along the genes dimension, e.g. ("Loadings",); "auto" collects all rank 2 tensors on the genes dimension
			obsm	Rank 2 tensors along the cells dimension, e.g. ("Loadings",); "auto" collects all rank 2 tensors on the cells dimension (but not those on ("cells", "genes"))
			var_key	The unique var primary key, which must be one of the var tensors; default is "Accession"
			obs_key	The unique obs primary key, which must be one of the obs tensors; default is "CellID"
			layers	Additional expression matrices, e.g. ("Unspliced",)

		Remarks:
			To rename a tensor, use "OldName->NewName" notation. For example, "Embedding->X_embedding" can be used
			to create an embedding compatible with cellxgene. If you use "auto" to automatically collect relevant tensors,
			you can still rename tensors by adding them after "auto" in a tuple: ("auto", "TotalUMIs->UMI_count")
		"""
		import scanpy as sc
		import pandas as pd

		n_cells = self.get_length("cells")
		n_genes = self.get_length("genes")

		def load_csr(tname):
			result = None
			n_cells_per_batch = 1000
			for ix in range(0, n_cells, n_cells_per_batch):
				batch = sparse.csr_matrix(self._read_batch(tname, ix, ix + n_cells_per_batch))
				if result is None:
					result = batch
				else:
					result = sparse.vstack(result, batch)
			return result

		def renamed(old):
			if "->" in old:
				old, new = old.split("->")
			else:
				new = old
			return old, new

		def load_dense(names, *, dim, rank):
			result = {}
			renamings = {}
			if isinstance(names, (tuple, list)):
				for old in names:
					if old == "auto":
						continue
					old, new = renamed(old)
					renamings[old] = new

			if names == "auto" or (isinstance(names, (tuple, list)) and names[0] == "auto"):
				names = []
				# Collect all relevant tensors
				for tname in self.wsm._tensors():
					tensor = self.wsm[tname]
					if tensor.rank == rank and tensor.dims[0] == dim:
						# Skip matrices that look like layers
						if tensor.rank == 2 and tensor.dims[1] == "genes":
							continue
						names.append(tname)
						if tname not in renamings:
							renamings[tname] = tname

			# Now names is a list or tuple of tensor names, and renamings map old to new names
			for name in names:
				result[renamings[name]] = self[name]
			return result

		X_data = load_csr(X)
		var_data = load_dense(var, dim="genes", rank=1)
		obs_data = load_dense(var, dim="cells", rank=1)
		varm_data = load_dense(var, dim="genes", rank=2)
		obsm_data = load_dense(var, dim="genes", rank=2)

		layers_data = {}
		for layer in layers:
			old, new = renamed(layer)
			layers_data[new] = load_csr(old)

		return sc.AnnData(X=X_data, obs=pd.DataFrame(obs_data).set_index(obs_key), var=pd.DataFrame(var_data).set_index(var_key), obsm=obsm_data, varm=varm_data, layers=layers_data)
