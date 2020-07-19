
Enums?
Pinning (prevent deletion) workspaces, dimensions and tensors
Maybe: use shoji as backing for xarray via dask
View.groupby() and aggregate (or maybe Dimension.groupby()?)
.aspandas(), .asxarray() etc.
Move .from_loompy() to .toshoji() on loompy

Make sure jagged & scalar tensors work as intended
Exploit parallelism with futures
Tensor shape property
Accelerate tensors by chunking along all dimensions, before compressing (like HDF5)
Filter by slice
Filter by bool array or index array
Write to database using assignment through a selector
Use ranges to speed up filtered reads
Append method on workspace
Docs (pdoc --html shoji)
Dimensions
Tensors
Transactions (context manager)
Selectors on workspaces
Indexes
Store strings properly (1D and nD)
Lazy-read tensor values with selectors


Shortcomings of loompy
======================

Graphs support only a single weight attribute (arbitrary number and type of attributes are not supported)
No good way to store dendrograms
Layers are supported, but only if they share both dimensions; not good for dual RNA+ATAC-seq or RNA+protein
No good way to store aggregated data
Not good for storing images (need 2D rows)
No way to store jagged arrays
No transactions -> risk data loss or errors
No read/write concurrency
Slow over network
No read parallelism even on a single computer
Weird bugs in HDF5 cause problems