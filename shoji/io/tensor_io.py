from typing import List, Tuple, Any, Union, Type, Dict
import numpy as np
import fdb
import shoji
import pickle
from .enums import Compartment


"""
# Tensor storage API

The tensor storage API handles reading and writing subsets of tensors defined by indices along each dimension,
which are translated to and from chunks as needed.
"""

@fdb.transactional
def create_tensor(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str, tensor: shoji.Tensor) -> None:
	"""
	Creates a new tensor (but does not write the inits)

	If inits were provided, the tensor is marked as initializing, and will be invisible until the inits have been written
	"""
	subdir = wsm._subdir
	# Check that name doesn't already exist
	existing = shoji.io.get_entity(tr, wsm, name)
	if existing is not None:
		raise AttributeError(f"Cannot overwrite {type(existing)} '{existing}' with a new shoji.Tensor (you must delete it first)")
	else:
		# Check that the dimensions of the tensor exist
		for ix, d in enumerate(tensor.dims):
			if isinstance(d, str):
				dim = shoji.io.get_dimension(tr, wsm, d)
				if dim is None:
					raise KeyError(f"Tensor dimension '{d}' is not defined")
				if dim.shape is not None:  # This is a fixed-length dimension
					if tensor.inits is not None and tensor.shape[ix] != dim.shape:
						raise IndexError(f"Mismatch between the declared shape {dim.shape} of dimension '{d}' and the shape {tensor.shape} of values")
	key = subdir.pack((Compartment.Tensors, name))
	tensor.shape = (0,) + tensor.shape[1:]  # The first dimension will be updated by write_tensor_values after values have been actually appended
	if tensor.inits is not None:
		tensor.initializing = True
	tr[key] = pickle.dumps(tensor)
		

def initialize_tensor(wsm: "shoji.WorkspaceManager", name: str, tensor: shoji.Tensor):
	if tensor.inits is not None:
		if tensor.rank == 0:
			write_at_indices(wsm._db.transaction, wsm, (Compartment.TensorValues, name), indices=None, chunk_sizes=None, values=tensor.inits.values, compression=tensor.compressed)
		else:
			# Hide the true dimensions so the append will not fail due to consistency checks
			dims = tensor.dims
			tensor.dims = (None,) * tensor.rank
			append_values_multibatch(wsm, [name], [tensor.inits], axes=(0,))
			tensor.dims = dims
		# Complete the intitalization in one atomic operation
		finish_initialization(wsm._db.transaction, wsm, name)


@fdb.transactional
def finish_initialization(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str) -> None:
	tensor = shoji.io.get_tensor(tr, wsm, name, include_initializing=True)
	assert tensor.initializing
	tensor.initializing = False
	# Update the tensor definition to clear the Initializing flag
	subdir = wsm._subdir
	key = subdir.pack((Compartment.Tensors, name))
	tr[key] = pickle.dumps(tensor)

	# Update the first dimension
	if tensor.rank > 0:
		for shape, dname in zip(tensor.shape, tensor.dims):
			if isinstance(dname, str):
				dim = wsm._get_dimension(dname)
				if dim.length == 0:
					dim.length = shape
					shoji.io.create_dimension(tr, wsm, dname, dim)
				elif dim.length != shape:
					raise ValueError(f"Length {shape} of new tensor '{name}' does not match length {dim.length} of dimension '{dname}' ")


@fdb.transactional
def update_tensor(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str, tensor: shoji.Tensor) -> None:
	# Store tensor definition (overwriting any existing definition)

	subdir = wsm._subdir
	key = subdir.pack((Compartment.Tensors, name))
	tr[key] = pickle.dumps(tensor)


@fdb.transactional
def write_at_indices(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", key_prefix: Tuple[Any], indices: List[np.ndarray], chunk_sizes: Tuple[int], values: np.ndarray, compression: bool = True) -> int:
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
	subspace = wsm._subdir
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
		# These are the indices into the whole tensor that fall within the chunk
		tensor_indices = [ind[(ind // sz) == a] for a, ind, sz in zip(address, indices, chunk_sizes)]
		# Now figure out which part of the values tensor they correspond to (always a dense sub-tensor)
		starts = [(i < min(j)).sum() for i, j in zip(indices, tensor_indices)]
		lengths = [len(j) for j in tensor_indices]
		values_slices = tuple(slice(s, s + l) for s, l in zip(starts, lengths))

		# Finally, copy the correct subtensor of values into the right slots in the chunk
		chunk[np.ix_(*chunk_indices)] = values[values_slices]

		mask = np.ones(chunk_sizes, dtype=bool)
		mask[np.ix_(*chunk_indices)] = False
		if np.any(mask):
			chunks.append(np.ma.masked_array(chunk, mask=mask))
		else:
			chunks.append(chunk)

	return shoji.io.write_chunks(tr, subspace, key_prefix, addresses, chunks, compression)


def read_at_indices(wsm: "shoji.WorkspaceManager", tensor: str, indices: List[np.ndarray], chunk_sizes: Tuple[int, ...], compression: bool = True, transactional: bool = True) -> np.ndarray:
	"""
	Read values corresponding to indices along each dimension (row indices, column indices, ...), automatically managing chunks as needed

	Args:
		wsm: workspace 
		tensor: name of the tensor
		indices: A list of numpy arrays giving the indices of the desired chunks
		chunk_sizes: A tuple of ints giving the size of chunks in each dimension
		compression: If true, decompress chunks when reading
		transactional: If false, read chunks in multiple batches adaptively

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
	if transactional:
		chunks = shoji.io.read_chunks(wsm._db.transaction, wsm._subdir, (Compartment.TensorValues, tensor), addresses, compression)
	else:
		chunks = shoji.io.read_chunks_multibatch(wsm._db.transaction, wsm._subdir, (Compartment.TensorValues, tensor), addresses, compression)
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


def dtype_class(dtype) -> Union[Type[int], Type[float], Type[bool], Type[str]]:
	if dtype in ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"):
		return int
	elif dtype in ("float16", "float32", "float64"):
		return float
	elif dtype == "bool":
		return bool
	elif dtype == "string":
		return str
	else:
		raise TypeError()
	

@fdb.transactional
def append_values(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", names: List[str], values: List[shoji.TensorValue], axes: Tuple[int]) -> int:
	"""
	Returns:
		Number of bytes written
	
	Remarks:
		This function uses a transaction to preserve the invariant that all tensors that share a dimension have the same length
		along that dimension (or zero length)
	"""
	subspace = wsm._subdir
	# Check that all the input values have same length along the relevant axis
	n_rows = -1
	for name, vals, axis in zip(names, values, axes):
		assert isinstance(vals, shoji.TensorValue), f"Input values must be numpy shoji.TensorValue, but '{name}' was {type(vals)}"
		assert vals.rank >= 1, f"Input values must be at least 1-dimensional, but '{name}' was scalar"
		if n_rows == -1:
			n_rows = vals.shape[axis]
		elif vals.shape[axis] != n_rows:
			raise ValueError(f"Length (along relevant axis) of all tensors must be the same when appending")

	n_bytes_written = 0
	all_tensors = {t.name: t for t in shoji.io.list_tensors(tr, wsm, include_initializing=True)}
	tensors: Dict[str, shoji.Tensor] = {}

	# Check that all the tensors exist, and have the right dimensions
	dname = None
	for name, axis in zip(names, axes):
		if name not in all_tensors:
			raise NameError(f"Tensor '{name}' does not exist in the workspace")
		tensor = all_tensors[name]
		tensors[name] = tensor
		if tensor.rank == 0:
			raise ValueError(f"Cannot append to scalar tensor '{name}'")
		if tensor.dims[axis] is not None:
			if isinstance(tensor.dims[axis], int):
				raise ValueError(f"Cannot append to fixed-length axis {axis} of tensor '{name}'")
			if dname is None:
				dname = tensor.dims[axis]
			elif tensor.dims[axis] != dname:
				raise ValueError(f"Cannot append to axis {axis} of tensor '{name}' because its dimension '{tensor.dims[axis]}' conflicts with dimension '{dname}' of another tensor")

	# Check the rank of the values, and the size along each axis
	new_length = 0
	for name, tensor, vals, axis in zip(names, tensors.values(), values, axes):
		if tensor.jagged:
			for row in vals:
				if tensor.rank != row.ndim + 1:  # type: ignore
					raise ValueError(f"Tensor '{name}' of rank {tensor.rank} cannot be appended with rank-{row.ndim + 1} array")  # type: ignore
		else:
			if tensor.rank != vals.rank:  # type: ignore
				raise ValueError(f"Tensor '{name}' of rank {tensor.rank} cannot be appended with rank-{vals.rank} array")  # type: ignore
		for i in range(tensor.rank):
			if i == axis:
				new_length = tensor.shape[i] + vals.shape[i]  # new_length will be the same for every tensor, since they start the same, and we checked above that the values are the same length
			else:
				if tensor.shape[i] != vals.shape[i]:
					raise ValueError(f"Cannot append values of shape {vals.shape} to tensor of shape {tensor.shape} along axis {axis}")

	# Check that all relevant tensors will have the right length after appending
	all_tensors = {n: t for n, t in all_tensors.items() if not t.initializing}  # Omit initializing tensors
	if dname is not None:
		for tensor in all_tensors.values():
			if tensor.name not in tensors:
				length_along_dim = tensor.shape[tensor.dims.index(dname)]
				if dname in tensor.dims and length_along_dim != new_length and not np.prod(tensor.shape) == 0:
					raise ValueError(f"Length {length_along_dim} of tensor '{tensor.name}' along dimension '{dname}' would conflict with length {new_length} after appending")

	for name, tensor, vals, axis in zip(names, tensors.values(), values, axes):
		if tensor.jagged:
			added_shape = vals.shape[axis]
			# Write row by row
			for i, row in enumerate(vals):
				ix = tensor.shape[axis] + i
				if axis == 0:
					indices = [np.array([ix])] + [np.arange(l) for l in row.shape]  # Just fill all the axes
				else:
					indices = [np.array([0])] + [np.arange(l) for l in row.shape]  # Just fill all the axes
					indices[axis] += tensor.shape[axis]  # Except the one we're appending, which starts at end of axis
				n_bytes_written += write_at_indices(tr, wsm, (Compartment.TensorValues, name), indices, tensor.chunks, row[None, ...], tensor.compressed)
				# Update row tensor shape
				key = subspace.pack((Compartment.TensorRowShapes, name, ix))
				tr[key] = fdb.tuple.pack(tuple(int(x) for x in row.shape))

			# Update tensor shape (use max length for non-first dimensions since this tensor is jagged)
			shape = list(tensor.shape)
			shape[axis] += added_shape
			tensor.shape = tuple(shape)
			update_tensor(tr, wsm, tensor.name, tensor)
		else:
			indices = [np.arange(l) for l in vals.shape]  # Just fill all the axes
			indices[axis] += tensor.shape[axis]  # Except the one we're appending, which starts at end of axis
			if tensor.rank == 1:
				# Write the index
				for i, value in enumerate(vals):
					casted_value = dtype_class(tensor.dtype)(value)
					key = subspace.pack((Compartment.TensorIndex, name, casted_value, int(indices[0][i])))
					n_bytes_written += len(key)
					tr[key] = b''
			n_bytes_written += write_at_indices(tr, wsm, (Compartment.TensorValues, name), indices, tensor.chunks, vals.values, tensor.compressed)
			# Update tensor shape
			shape = list(tensor.shape)
			shape[axis] += vals.shape[axis]
			tensor.shape = tuple(shape)
			update_tensor(tr, wsm, tensor.name, tensor)
	return n_bytes_written

def append_values_multibatch(wsm: "shoji.WorkspaceManager", tensors: List[str], values: List[np.ndarray], axes: Tuple[int]) -> int:
	"""
	Append values to a set of tensors, using multiple batches (transactions) if needed.

	Args:
		tensors: the names of the tensors to which values will be appended
		values: a list of ndarray objects to append (in the same order as the tensors)
		axes: a tuple giving the axis to which values should be appended, for each tensor

	Remarks:
		Values are appended along the given axis on each  tensor
		
		The batch size used when appending is adapted dynamically to maximize performance,
		while ensuring that the same number of (generalized) rows are appended to each tensor
		in each transaction.

		For each batch (transaction), the validity of appending values will be validated, to
		ensure safe concurrency
	"""
	n = values[0].shape[axes[0]]  # Start by attempting to append everything
	n_total = n
	n_bytes_written = 0
	ix = 0
	while ix < n_total:
		try:
			n_bytes_written += append_values(wsm._db.transaction, wsm, tensors, values, axes)
		except fdb.impl.FDBError as e:
			if e.code in (1004, 1007, 1031, 2101) and n > 1:  # Too many bytes or too long time, so try again with less
				n = max(1, n // 2)
				continue
			else:
				raise e
		ix += n
	return n_bytes_written
