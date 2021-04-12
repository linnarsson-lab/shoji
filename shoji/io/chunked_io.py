from typing import List, Tuple, Any, Union
import numpy as np
import fdb
import blosc
from numpy.ma.core import MaskedArray
import time
from multiprocessing import Pool, cpu_count
from itertools import chain


"""
# Chunked storage API

Shoji uses FoundationDB, a scalable and resilient key-value database, as its backing store. In order to bridge the mismatch between
a key-value store and a tensor database, the chunked storage API layer implements an N-dimensional compressed chunk storage layer.

Chunks are arbitrary numpy arrays, although they are intended to store chunks of N-dimensional tensors. Chunks are 
addressed using N-tuples of ordered integers, such as (0, 10, 9). Addresses can be viewed simply as abstract pointers, 
although they are intended to correspond to chunk offsets along each dimension of an N-dimensional tensor.

Chunks are stored in a FoundationDB subspace and under a specific key prefix (intended to store a single tensor). All chunks that 
are in the same subspace and using the same key prefix must use addresses of the same length (intended to correspond to the rank of a tensor). 

The chunked storage API layer provides functions for reading and writing sets of chunks using on-the-fly compression.

Note that the chunk address space need not be densely filled. That is, if a chunk exists at (10, 9, 3), this does not mean that 
chunks must exist at (9, 8, 1) or any other address. Reading from an empty address returns None. Writing to a non-empty address
silently overwrites the existing chunk.

"""


@fdb.transactional
def write_chunks(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], addresses: np.ndarray, chunks: List[Union[np.ndarray, MaskedArray]]) -> int:
	"""
	Write a list of chunks to the database, optionally using mask to write only partially

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		addresses: An (n_chunks, n_dim) numpy array giving the addresses of the desired chunks, along each dimension
		chunks: List of chunks, each of which can optionally be a numpy masked array

	Returns:
		The number of bytes written
	
	Remarks:
		Chunks can be given as numpy masked arrays, and masked values will be filled by the corresponding values from 
		the current chunk at the same address (which must exist). This can be used to selectively update
		only parts of chunks, e.g. when updating part of a tensor or appending values that are nonaligned with chunk edges.
	"""
	# logging.info(f"Writing {addresses.shape[0]} chunks starting at {addresses[0]}")
	n_bytes_written = 0
	if len(addresses) == 0:  # writing a scalar
		key = subspace.pack(key_prefix)
		encoded = blosc.pack_array(chunks[0])
		n_bytes_written += len(key) + len(encoded)
		tr[key] = encoded
		return n_bytes_written

	for address, chunk in zip(addresses, chunks):
		key = subspace.pack(key_prefix + tuple(int(x) for x in address))
		if isinstance(chunk, np.ma.MaskedArray):
			mask = np.ma.getmask(chunk)
			if np.any(mask):
				prev_value = read_chunks(tr, subspace, key_prefix, address[None, :])[0]
				if prev_value is not None:
					chunk[mask] = prev_value[mask]
			chunk = chunk.data
		# chunk is now an ndarray (not masked)
		encoded = blosc.pack_array(chunk)
		n_bytes_written += len(key) + len(encoded)
		tr[key] = encoded
	# logging.info(f"Wrote {addresses.shape[0]} chunks starting at {addresses[0]}, total of {n_bytes_written:,} bytes")
	return n_bytes_written

@fdb.transactional
def read_chunks(tr: fdb.impl.Transaction, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any], addresses: np.ndarray) -> List[np.ndarray]:
	"""
	Read a list of chunks from the database, using a transaction

	Args:
		tr: Transaction object
		subspace: The fdb DirectorySubspace under which the chunks are stored
		key_prefix: The tuple to use as prefix when storing the chunks
		addresses: An (n_chunks, n_dim) numpy array giving the addresses of the desired chunks, along each dimension

	Returns:
		chunks: A list of np.ndarray objects representing the desired chunks
	
	Remarks:
		Chunks that don't exist in the database are returned as None
	"""
	# logging.info(f"Reading {addresses.shape[0]} chunks starting at {addresses[0]}")
	n_bytes_read = 0

	if len(addresses) == 0:  # writing a scalar
		key = subspace.pack(key_prefix)
		data = tr[key].value
		decoded = blosc.unpack_array(data)
		n_bytes_read += len(key) + len(data)
		return [decoded]

	chunks: List[np.ndarray] = []
	for address in addresses:
		key = subspace.pack(key_prefix + tuple(int(x) for x in address))
		data = tr[key].value
		if data is None:
			chunks.append(None)
		else:
			decoded = blosc.unpack_array(data)
			n_bytes_read += len(key) + len(data)
			chunks.append(decoded)
	# logging.info(f"Read {addresses.shape[0]} chunks starting at {addresses[0]}, total of {n_bytes_read:,} bytes")
	return chunks


# Note: no @fdb.transactional decorator since this uses multiple transanctions inside the function
def read_chunks_multibatch(db: fdb.impl.Database, subspace: fdb.directory_impl.DirectorySubspace, key_prefix: Tuple[Any, ...], addresses: np.ndarray) -> List[np.ndarray]:
	chunks = []
	n_total = len(addresses)
	n = min(200, n_total)
	# Read the first 100 chunks and measure the time it takes
	time_at_start = time.time()
	try:
		chunks += read_chunks(db, subspace, key_prefix, addresses[:200])
		ix = n
		time_per_chunk = (time.time() - time_at_start) / n
		# Aim for 2s batches
		n = max(2, min(int(1 / time_per_chunk), n_total))
	except fdb.impl.FDBError as e:
		if e.code not in (1004, 1007, 1031, 2101):  # Too many bytes or too long time
			raise e
		ix = 0
		time_per_chunk = (time.time() - time_at_start) / n
		# Aim for 0.2s batches
		n = max(1, min(int(0.2 / time_per_chunk), n_total))
	# print(f"Reading {key_prefix[-1]} at {time_per_chunk * 1000:.2} ms/chunk")
	while ix < n_total:
		try:
			chunks += read_chunks(db, subspace, key_prefix, addresses[ix: ix + n])
		except fdb.impl.FDBError as e:
			if e.code in (1004, 1007, 1031, 2101) and n > 1:  # Too many bytes or too long time, so try again with less
				n = max(1, n // 2)
				continue
			else:
				raise e
		ix += n
	return chunks
