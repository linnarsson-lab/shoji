Initialize tensor by writing a stream of chunks (not row by row)
Pinning (prevent deletion) workspaces, dimensions and tensors
Maybe: use shoji as backing for xarray via dask
Maybe: Move .from_loom() to .toshoji() on loompy
Maybe: datetime and timeinterval datatypes (but which ones?)
Maybe: direct sparse matrix support (at least for reading)
Maybe: Enums
GUI
Cookbook: how to model different kinds of data (table, relational db, sparse matrix, graph)
Maybe: enforce consistent jaggedness if a jagged dimension is named (and if performance allows)

Improved chunking for partial reads
BUG: Tensor.shape should not be an exact value for dims that are jagged
BUG: Dimension.shape is not updated when it is the 2nd or higher dimension of some tensors
View.groupby() and aggregate (or maybe Dimension.groupby()?)
Minimize number of roundtrips per operation to reduce latency
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
Layers are supported, but only if they share both dimensions; not good for dual RNA+ATAC-seq or RNA+protein
No good way to store aggregated data
Not good for storing images (need 2D rows)
No way to store jagged arrays
No transactions -> risk data loss or errors
No read/write concurrency
Slow over network
No parallelism even on a single computer
Weird bugs in HDF5 cause problems