from typing import List, Tuple, Any, Union, Optional, Type
import numpy as np
import fdb
import blosc
import shoji
import pickle


"""
# 2. Chunked storage API

Shoji uses FoundationDB, a scalable and resilient key-value database, as its backing store. In order to bridge the mismatch between
a key-value store and a tensor database, the chunked storage API layer implements an N-dimensional compressed chunk storage layer.

Chunks are arbitrary numpy arrays, although they are intended to store chunks of N-dimensional tensors. Chunks are 
addressed using N-tuples of ordered integers, such as (0, 10, 9). Addresses can be viewed simply as abstract pointers, 
although they are intended to correspond to chunk offsets along each dimension of an N-dimensional tensor.

Chunks are stored in a FoundationDB subspace and under a specific key prefix (intended to store a single tensor). All chunks that 
are in the same subspace and using the same key prefix must use addresses of the same length (intended to correspond to the rank of a tensor). 

The chunked storage API layer provides functions for reading and writing sets of chunks, optionally using on-the-fly compression.

Note that the chunk address space need not be densely filled. That is, if a chunk exists at (10, 9, 3), this does not mean that 
chunks must exist at (9, 8, 1) or any other address. Reading from an empty address returns None. Writing to a non-empty address
silently overwrites the existing chunk.

"""


@fdb.transactional
def write_chunks(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], addresses: np.ndarray, chunks: List[Union[np.ndarray, np.ma.ndarray]], compression: bool = True) -> int:
	"""
	Write a list of chunks to the database, optionally using mask to write only partially

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		addresses: An (n_chunks, n_dim) numpy array giving the addresses of the desired chunks, along each dimension
		chunks: List of chunks, each of which can optionally be a numpy masked array
		compression: If true, compress each chunk when writing

	Returns:
		The number of bytes written
	
	Remarks:
		Chunks can be given as numpy masked arrays, and masked values will be filled by the corresponding values from 
		the current chunk at the same address (which must exist). This can be used to selectively update
		only parts of chunks, e.g. when updating part of a tensor or appending values that are nonaligned with chunk edges.
	"""
	n_bytes_written = 0
	for address, chunk in zip(addresses, chunks):
		key = subspace.pack(key_prefix + tuple(address))
		mask = np.ma.getmask(chunk)
		if mask != False:
			prev_value = read_chunks(tr, subspace, key_prefix, address[None, :], compression)[0]
			if prev_value is None:
				raise IOError("Attempt to write masked chunk to an empty address")
			chunk[mask] = prev_value
			chunk = chunk.data
		if compression:
			encoded = blosc.pack_array(chunk)
		else:
			encoded = chunk
		n_bytes_written += len(key) + len(encoded)
		tr[key] = encoded
	return n_bytes_written

@fdb.transactional
def read_chunks(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], addresses: np.ndarray, compression: bool = True) -> List[np.ndarray]:
	"""
	Read a list of chunks from the database, using a transaction

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		addresses: An (n_chunks, n_dim) numpy array giving the addresses of the desired chunks, along each dimension
		compression: If true, decompress chunks when reading

	Returns:
		chunks: A list of np.ndarray objects representing the desired chunks
	
	Remarks:
		Chunks that don't exist in the database are returned as None
	"""
	chunks = []
	for address in addresses:
		key = subspace.pack(key_prefix + tuple(address))
		if compression:
			decoded = blosc.unpack_array(tr[key])
		else:
			decoded = tr[key]
		chunks.append(decoded)
	return chunks

# Note: no @fdb.transactional decorator since this uses multiple transanctions inside the function
def read_chunks_multibatch(db: fdb.impl.Database, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any, ...], addresses: np.ndarray, compression: bool = True) -> List[np.ndarray]:
	n = len(addresses) # Start by attempting to read everything
	n_total = n
	ix = 0
	chunks = []
	while ix < n_total:
		try:
			chunks += read_chunks(db, subspace, key_prefix, addresses[ix: ix + n], compression)
		except fdb.impl.FDBError as e:
			if e.code in (1004, 1007, 1031, 2101) and n > 1:  # Too many bytes or too long time, so try again with less
				n = max(1, n // 2)
				continue
			else:
				raise e
	return chunks
