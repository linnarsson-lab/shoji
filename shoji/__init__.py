"""
Shoji is a tensor database, suitable for storing and working with very large-scale datasets
organized as vectors, matrices and higher-dimensional tensors. 

## Key features

- Multi-petabyte scalable, distributed, high-performance database
- Data modelled as N-dimensional tensors with boolean, string or numeric elements
- Supports both regular and jagged tensors
- Automatic data compression
- Relationships expressed through shared (and named) dimensions
- Read and write data through filter expressions (similar to SQL SELECT)
- Data safety through transactions and [ACID](https://en.wikipedia.org/wiki/ACID) properties (atomicity, consistency, isolation, durability)
- Concurrent access, with consistent serial reads and writes
- Elegant, convenient Python API, aligned with numpy

Oh, and it's fast.

## Data model

In Shoji, data is stored as tensors, and relationships are expressed using shared dimensions. 

The fundamental unit of data storage in Shoji is the *row* (and its generalization to N dimensions). 
Data is added to, or removed from, tensors by rows; columns cannot be added or removed. 

Dimensions can be named, and named dimensions express relationships and constraints between tensors.
Tensors that share a named dimension must have the same length along that dimension (and this relationship
is enforced when adding or removing rows).

You can think of rows as your data *objects*, dimensions as object *types*, and the tensors as object
*attributes*. For example, a set of vectors defined on a samples dimension could be seen as the attributes
of samples (e.g. SampleID, Age, Tissue, Date), and an individual sample would correspond to an individual
row across all tensors.

Tensors can also be related to multiple named dimensions. For example, omics data (e.g. gene expression)
is often represented as matrices, which can be represented in Shoji as rank-2 tensors with two named
dimensions, e.g. cells and genes. Metadata about cells and genes would be stored as rank-1 tensors
(vectors) along the cells and genes dimensions, respectively. 

Similarly, multichannel timelapse image data can be represented as high-rank tensors with dimensions
such as x, y, channel, and timepoint.

## Important classes

`shoji.workspace.Workspace`:    Workspaces let you organise collections of data that belong together.

`shoji.tensor.Tensor`:  N-dimensional arrays of numbers, booleans and strings.

`shoji.dimension.Dimension`:    Named dimensions that constrain tensors.

`shoji.filter.Filter`:  Expressions used to read subsets of tensors.

`shoji.view.View`:  A view of the database through a set of filter expressions.

`shoji.transaction.Transaction`:    Read and write data using atomic operations.

"""
import fdb
fdb.api_version(620)

from .codec import Codec
from .dimension import Dimension
from .tensor import Tensor
from .workspace import Workspace, WorkspaceManager
from .connect import connect
from .filter import Filter, CompoundFilter, TensorFilter, ConstFilter, DimensionBoolFilter, DimensionIndicesFilter, DimensionSliceFilter
from .transaction import Transaction
from .view import View
from .io import __nil__
