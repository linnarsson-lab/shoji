from typing import List, Tuple, Any, Union
import numpy as np
import fdb
import blosc

	
"""
# Chunked storage API layer

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

In addition, the chunked storage API handles reading and writing subsets of tensors defined by indices along each dimension,
which are translated to and from chunks as needed.
"""

# TODO: read_chunks_nontransactionally using dynamic adjustment of the number of chunks per transaction


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


@fdb.transactional
def write_indices(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], indices: List[np.ndarray], chunk_sizes: Tuple[int], values: np.ndarray, compression: bool = True) -> int:
	"""
	Write values corresponding to indices along each dimension (row indices, column indices, ...), automatically managing chunks as needed

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		indices: A list of numpy arrays giving the indices of the desired chunks
		chunk_sizes: A tuple of ints giving the size of chunks in each dimension
		values: An ndarray of values corresponding to the intersection of indices
		compression: If true, decompress chunks when reading

	Returns:
		The number of bytes written
	"""
	# Figure out which chunks need to be written
	addresses_per_dim = [np.unique(ind // sz) for ind, sz in zip(indices, chunk_sizes)]
	# All combinations of addresses along each dimension
	addresses = np.array(np.meshgrid(*addresses_per_dim)).T.reshape(-1, len(indices))
	chunks = []
	for address in addresses:
		# At this point, we have a chunk address, and we have the indices
		# into the whole tensor. We need to figure out the relevant indices for this chunk,
		# and their offsets in the chunk, so that we can place the right values at the right place in
		# the chunk for writing. We also need to construct a mask if the chunk is not fully covered
		chunk_indices = [ind[(ind // sz) == a] - a * sz for a, ind, sz in zip(address, indices, chunk_sizes)]
		chunk = np.empty_like(values, shape=chunk_sizes)
		tensor_indices = [ind[(ind // sz) == a] for a, ind, sz in zip(address, indices, chunk_sizes)]
		chunk[np.ix_(*tensor_indices)] = values[np.ix_(*tensor_indices)]

		mask = np.ones(chunk_sizes, dtype=bool)
		mask[np.ix_(*chunk_indices)] = False
		if np.any(mask):
			chunks.append(np.ma.array(chunk, mask))
		else:
			chunks.append(chunk)

	return write_chunks(tr, subspace, key_prefix, addresses, chunks, compression)


@fdb.transactional
def read_indices(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], indices: List[np.ndarray], chunk_sizes: Tuple[int], compression: bool = True) -> np.ndarray:
	"""
	Read values corresponding to indices along each dimension (row indices, column indices, ...), automatically manging chunks as needed

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		indices: A list of numpy arrays giving the indices of the desired chunks
		chunk_sizes: A tuple of ints giving the size of chunks in each dimension
		compression: If true, decompress chunks when reading

	Returns:
		data: The values at the intersection of each set of indices
	
	Remarks:
		All the relevant chunks must exist, or this function will throw an exception
	"""
	# Figure out which chunks need to be read
	addresses_per_dim = [np.unique(ind // sz) for ind, sz in zip(indices, chunk_sizes)]
	# All combinations of addresses along each dimension
	addresses = np.array(np.meshgrid(*addresses_per_dim)).T.reshape(-1, len(indices))
	# Read the chunk data and unravel it into the result ndarray
	chunks = read_chunks(tr, subspace, key_prefix, addresses, compression)	
	result = np.empty_like(chunks[0], shape=[len(i) for i in indices])
	for address, chunk in zip(addresses, chunks):
		# At this point, we have a chunk at a particular address, and we have the indices
		# into the whole tensor. We need to figure out the relevant indices for this chunk,
		# and their offsets in the chunk, so that we can extract the right values from
		# the chunk. 
		chunk_indices = [ind[(ind // sz) == a] - a * sz for a, ind, sz in zip(address, indices, chunk_sizes)]
		chunk_extract = chunk[np.ix_(*chunk_indices)]
		# We then need to figure out the offsets of those indices into the
		# result tensor so that we can write the values in the right place.
		# The chunk_extract should be placed as a dense ndarray into the result ndarray,
		# so we only need to figure out the offsets along each dimension. This is
		# equivalent to the number of indices belonging to lower addresses
		# in all dimensions.
		lowest_indices = [min(i) for i in [ind[(ind // sz) == a] for a, ind, sz in zip(address, indices, chunk_sizes)]]
		offsets = [(ind < min_ind).sum() for ind, min_ind in zip(indices, lowest_indices)]
		result[tuple([slice(a, a + b) for a, b in zip(offsets, chunk_sizes)])] = chunk_extract
	return result
