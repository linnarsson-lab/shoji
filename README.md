    💡 This is an early preview version, subject to breaking changes. 

# Shoji

Shoji is a tensor database, suitable for storing and working with very large-scale datasets organized as vectors, matrices and higher-dimensional tensors.

## Key features

* Multi-petabyte scalable, distributed, high-performance database
* Data modelled as N-dimensional tensors with boolean, string or numeric elements
* Supports both regular and jagged tensors
* Automatic chunking and compression
* Relationships expressed through shared named dimensions
* Read and write data through views created by powerful filter expressions
* Automatic indexing for fast filtering
* Data safety through transactions and ACID properties (atomicity, consistency, isolation, durability)
* Concurrent read/write access
* Elegant, convenient Python API, aligned with numpy

Oh, and it's pretty fast.

## Overview

### Data model

In Shoji, data is stored as tensors, and relationships are expressed using shared dimensions.

Dimensions can be named, and named dimensions express relationships and constraints between tensors. Tensors that share a named dimension must have the same length along that dimension (and this relationship is enforced when adding data).

You can think of rows as your data objects, dimensions as object types, and the tensors as object attributes. For example, a set of vectors (e.g. SampleID, Age, Tissue, Date) defined on a samples dimension could be seen as the attributes of samples, and an individual sample would correspond to an individual row across all tensors.

Tensors can also be related to multiple named dimensions. For example, omics data (e.g. gene expression) is often represented as matrices, which can be represented in Shoji as rank-2 tensors with two named dimensions, e.g. cells and genes. Metadata about cells and genes would be stored as rank-1 tensors (vectors) along the cells and genes dimensions, respectively. Similarly, multichannel timelapse image data can be represented as high-rank tensors with dimensions such as x, y, channel, and timepoint. This makes Shoji fundamentally different from tabular (relational) databases, which struggle to represent multidimensional data.

The fundamental operations in shoji are: creating a tensor, appending values, reading values, updating values. Tensors can be deleted, but individual tensor rows cannot.

### ACID guarantees
Shoji treats the slice as the atomic unit when writing data. This means that if your program crashes in the middle of an operation, you are guaranteed that there will be no half-created rows, or partially updated elements in the database.

When more than one tensor shares their first dimension, the atomic unit for writing new data (i.e. for Dimension.append()) is a slice across all tensors that share the same first dimension. In other words, if your program crashes in the middle of an append() operation, shoji guarantees that some number of complete indices (or nothing at all) will have been written across all the relevant tensors, ensuring that they stay in sync.

If you need stronger guarantees, you can wrap multiple database operations in a shoji.transaction.

### Limitations

Shoji is built on FoundationDB, a powerful open-source key-value store developed by Apple. It is FoundationDB that gives Shoji a solid foundation of performance, scalability and ACID guarantees. In order to gain these features, there are a few limitations though:

Transactions cannot exceed 5 seconds. If a transaction takes longer, it's terminated and rolled back. For Shoji, this limits the total feasible size of a slice (or a set of rows for append operations), since Shoji reads and writes slices transactionally.

Transactions exceeding 1 MB can cause performance issues, and transactions cannot exceed 10 MB. This also limits the total feasible size of a tensor slice, since Shoji reads and writes slices transactionally to ensure consistency.

FoundationDB is optimized to run on SSDs. Running on mechanical disks is discouraged.

For more details about these and some other limitations, see the FoundationDB docs

# Getting started with Shoji

Shoji requires Python 3.7+ (we recommend Anaconda)

First, install the FoundationDb client:

1. [Download FoundationDB](https://apple.github.io/foundationdb/downloads.html)
2. Double-click on FoundationDB-6.#.##.pkg and follow the instructions
    
    <aside>
    💡 If you get a security error (“FoundationDB-6.2.27.pkg” cannot be opened because it is from an unidentified developer) go to Settings → Security & Privacy → General and click on Open Anyway and then on Open (in the dialog).
    
    </aside>
    
Next, in your terminal, install the foundationdb and shoji Python packages:

```
$ pip install foundationdb
$ git clone https://github.com/linnarsson-lab/shoji.git
$ pip install -e shoji
```

Check that you can now connect to the database:

```python
import shoji
db = shoji.connect()
db
```

Typing db alone at the last line above should return a representation of the contents of the database (which might be empty at this point).

## Documentation

Clone the repository, and then go to the `shoji/html` folder to browse the docs.

# Setting up a Shoji database on macOS

<aside>
❗ This page is for setting up a new Shoji database. If you're just going to use an existing Shoji database, e.g. on monod, there is no need to set up FoundationDB or Shoji locally.

</aside>

### Install FoundationDB

Shoji is based on FoundationDB. You can easily set up a local Shoji database by following these instructions:

1. [Download FoundationDB](https://apple.github.io/foundationdb/downloads.html)
2. Double-click on FoundationDB-6.#.##.pkg and follow the instructions
    
    <aside>
    💡 If you get a security error (“FoundationDB-6.2.27.pkg” cannot be opened because it is from an unidentified developer) go to Settings → Security & Privacy → General and click on Open Anyway and then on Open (in the dialog).
    
    </aside>
    
3. In your Terminal, type `fdbcli` and then `status` to confirm that the database is up and running
4. Still in `fdbcli`, type `configure ssd` to change the storage engine to ssd-2 
5. After a few minutes, `status` should again show as `Healthy`

### Install the Python libraries

```bash
pip install -U foundationdb
git clone https://github.com/linnarsson-lab/shoji.git
cd shoji
pip install -e .
```

### Verify the installation

```python
>>> import shoji
>>> db = shoji.connect()
>>> db
(root) (shoji.Workspace)
```

### Where is everything?

Documentation: `shoji/html/shoji/index.html`

FDB cluster file: `/usr/local/etc/foundationdb/fdb.cluster`

FDB config: `/usr/local/etc/foundationdb/foundationdb.conf`

Data: `/usr/local/foundationdb/`
