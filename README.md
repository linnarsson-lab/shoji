> [!IMPORTANT]
> Nobody is actively developing or supporting Shoji. It was created for internal use in the Linnarsson Lab at Karolinska Institutet, Sweden. It is provided here as-is for your enjoyment.


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



# Installing Shoji on MacOS

> [!NOTE]
> This page shows how to install the client and set up a new Shoji database on MacOS. If you're just going to connect to an existing Shoji database, there is no need to set up FoundationDB locally. Pay attention below and just skip the steps related to the server.

> [!WARNING]
> FoundationDB on MacOS is intended only as a playground, but lacks data replication and **is not failure tolerant**. For a production environment, FoundationDB and Shoji should be installed on a Linux server. For best reliability and performance, configure multiple machines and one process per core per machine. See the FoundationDB [Administration Guide](https://apple.github.io/foundationdb/administration.html) to get started.


### Install FoundationDB

Shoji is based on FoundationDB. You can easily set up a local Shoji database by following these instructions:

**Download the installer** from the [FoundationDB release page](https://github.com/apple/foundationdb/releases)

Get a stable release, e.g. FoundationDB-7.3.69_arm64.pkg.

> [!NOTE]
> If you are using a Mac with Apple Silicon CPU (e.g. M1, M2, etc.) you should install from an **amd64** build, otherwise use an **x86** build.


> [!CAUTION]
> Do not download the installer from the [official downloads page](https://apple.github.io/foundationdb/downloads.html). It is very old and will not work on Apple Silicon.


**Double-click on FoundationDB-7.3.69.pkg** and follow the instructions.

> [!TIP]
> If you get a security error (e.g. *“FoundationDB-7.3.69_arm64.pkg cannot be opened because it is from an unidentified developer"*) go to Settings → Security & Privacy → General and click on Open Anyway and then on Open (in the dialog).

If you want to use a local server on your Mac, install both the server and the client. If you want to connect to a remote server, install only the client.

**Confirm that your installation was successful:**

In your Terminal, type `fdbcli` and then `status` to confirm that the database is up and running

**Change to store the data on disk**
1. Still in `fdbcli`, type `configure ssd` to change the storage engine to ssd-2 (otherwise, FoundationDb will use an in-memory database)
2. After a few minutes, `status` should again show as `Healthy`

### Install the Python libraries

> [!IMPORTANT]
> The client and server major and minor versions must match the major and minor version of the Python library. If you installed FoundationDB 7.3.xx above, you should install the most recent Python library in the 7.3.x series, but not anything in the 7.4.x series.

```bash
pip install "foundationdb>=7.3.0,<7.4.0" ## Most recent version in the 7.3 series
```

### Install Shoji

> [!IMPORTANT]
> Do not attempt to `pip install shoji`. That's a completely unrelated library. Our Shoji is not in pip.

Clone the repository and install it locally using pip:

```bash
git clone git@github.com:linnarsson-lab/shoji.git  # Clone using SSH; use https://github.com/linnarsson-lab/shoji.git to clone via HTTPS
pip install -e shoji
```

### Verify the installation
Run this code in a Python interpreter or Jupyter notebook:

```python
>>> import shoji
>>> db = shoji.connect()
>>> db
(root) (shoji.Workspace)
```

### What's next? Get started with example data

Walk through the [GettingStarted_Shoji.ipynb](https://github.com/linnarsson-lab/shoji/blob/master/notebooks/GettingStarted_Shoji.ipynb) Jupyter notebook in the `/notebooks` directory of the Shoji repository. It demonstrates how to import an .h5ad file and plot some data.

### Where is everything?

Documentation: `shoji/html/shoji/index.html` (in this repository)

FDB cluster file: `/usr/local/etc/foundationdb/fdb.cluster`

FDB config: `/usr/local/etc/foundationdb/foundationdb.conf`

Data: `/usr/local/foundationdb/`
