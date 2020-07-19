"""
Internal low-level I/O routines, not intended for end users.
"""

from typing import Union, Optional, Tuple, Dict, List
import fdb
import numpy as np
import shoji
import numba
import pickle
import copy


CHUNK_SIZE = 2_000


@fdb.transactional
def get_entity(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[str]:
	s = get_subspace(tr, wsm, name)
	if s is not None:
		return s
	d = get_dimension(tr, wsm, name)
	if d is not None:
		return d
	t = get_tensor(tr, wsm, name)
	if t is not None:
		return t
	return None


@fdb.transactional
def get_subspace(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[shoji.WorkspaceManager]:
	subspace = wsm._subspace
	if subspace.exists(tr, name):
		child = subspace.open(tr, name)
		wsm = shoji.WorkspaceManager(wsm._db, child, wsm._path + (name,))
		wsm._name = name
		return wsm
	return None


@fdb.transactional
def get_dimension(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[shoji.Dimension]:
	subspace = wsm._subspace
	val = tr[subspace["dimensions"][name]]
	if val.present():
		val = tr[subspace["dimensions"][name]]
		dim = shoji.Dimension(shape=int.from_bytes(val[:8], "little", signed=True), length=int.from_bytes(val[8:], "little", signed=False))
		dim.name = name
		dim.wsm = wsm
		return dim
	return None


@fdb.transactional
def get_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[shoji.Tensor]:
	subspace = wsm._subspace
	# ("tensors", name) = (tensor.dtype, tensor.rank, tensor.jagged) + tensor.dims + shape
	# where shape[0] == -1 if jagged
	val = tr[subspace.pack(("tensors", name))]
	if val.present():
		t = pickle.loads(val.value)
		if t[1] > 0:
			shape = t[-t[1]:]
		else:
			shape = ()
		tensor = shoji.Tensor(t[0], t[3:3 + t[1]], shape=shape)
		tensor.jagged = t[2] == 1
		tensor.name = name
		tensor.wsm = wsm
		return tensor
	return None


@fdb.transactional
def delete_entity(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> None:
	subspace = wsm._subspace
	if subspace.exists(tr, name):
		subspace.open(tr, name).remove(tr)
	elif tr[subspace["dimensions"][name]].present():
		tr.clear_range_startswith(subspace["dimensions"][name].key())
	elif tr[subspace["tensors"][name]].present():
		tr.clear_range_startswith(subspace["tensors"][name].key())
		tr.clear_range_startswith(subspace["tensor_values"][name].key())


@fdb.transactional
def create_or_update_dimension(tr, wsm: shoji.WorkspaceManager, name: str, dim: shoji.Dimension):
	subspace = wsm._subspace
	# Check that name doesn't already exist
	existing = get_entity(tr, wsm, name)
	if existing is not None:
		if not isinstance(existing, shoji.Dimension):
			raise AttributeError(f"Name already exists (as {existing})")
		# Update an existing dimension
		prev_dim = existing
		if prev_dim.shape != dim.shape:
			if isinstance(dim.shape, int):
				# Changing to a fixed shape, so we must check that all relevant tensors agree
				all_tensors: List[str] = [subspace["tensors"].unpack(k)[0] for k,v in tr[subspace["tensors"].range()]]
				for tensor_name in all_tensors:
					tensor = get_tensor(tr, wsm, tensor_name)
					if tensor.rank > 0 and tensor.dims[0] == name:
						if tensor.shape[0] != dim.shape:
							raise AttributeError(f"New shape of existing dimension '{name}' conflicts with length {tensor.shape[0]} of existing tensor '{tensor_name}'")
				dim.length = dim.shape
			else:
				dim.length = prev_dim.length
	# Create or update the dimension
	tr[subspace["dimensions"][name]] = (dim.shape if dim.shape is not None else -1).to_bytes(8, "little", signed=True) + dim.length.to_bytes(8, "little", signed=False)


@fdb.transactional
def create_or_update_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor) -> None:
	# Check that name doesn't already exist
	existing = get_entity(tr, wsm, name)
	if existing is not None:
		if not isinstance(existing, shoji.Tensor):
			raise AttributeError(f"Name already exists (as {existing})")
	else:
		# Check that the dimensions of the tensor exist
		for ix, d in enumerate(tensor.dims):
			if isinstance(d, str):
				dim = get_dimension(tr, wsm, d)
				if dim is None:
					raise KeyError(f"Tensor dimension '{d}' is not defined")
				if dim.shape is not None and tensor.inits is not None:
					if tensor.jagged:
						assert isinstance(tensor.inits, list)
						for row in tensor.inits:
							if row.shape[ix] != dim.shape:
								raise ValueError(f"Tensor '{name}' dimension {ix} ('{tensor.dims[ix]}') must be exactly {dim.shape} elements long")
					elif dim.shape != tensor.inits.shape[ix]:  # type: ignore
						raise ValueError(f"Tensor '{name}' dimension {ix} ('{tensor.dims[ix]}') must be exactly {dim.shape} elements long")
				# Check that the dimensions of the tensor match the shape of the tensor
				if dim.shape is None:
					if ix > 0:
						tensor.jagged = True
				else:
					if tensor.inits is not None and tensor.shape[ix] != dim.shape:
						raise IndexError(f"Mismatch between the declared shape {dim.shape} of dimension '{d}' and the shape {tensor.shape} of values")
	# Store tensor definition (overwriting any existing definition)
	# ("tensors", name) = (dtype, rank, jagged) + dims + shape  
	# where shape[0] == -1 if jagged, and tuple value is encoded using pickle
	subspace = wsm._subspace
	shape = tensor.shape if existing is not None else (0,) + tensor.shape[1:]  # if this is a new tensor, use shape (0, ...)
	tr[subspace.pack(("tensors", name))] = pickle.dumps((tensor.dtype, tensor.rank, 1 if tensor.jagged else 0) + tensor.dims + shape)


def coerce_dtype(dtype, v) -> Union[int, float, bool, str]:
	if dtype in ("uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64"):
		return int(v)
	elif dtype in ("float16", "float32", "float64"):
		return float(v)
	elif dtype == "bool":
		return bool(v)
	elif dtype == "string":
		return str(v)
	else:
		raise TypeError()


@fdb.transactional
def write_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, in_tensor: shoji.Tensor, indices: np.ndarray = None):
	"""
	Args:
		tr          Transaction
		wsm         WorkspaceManager
		name        Name of the tensor in the database
		in_tensor      A tensor with inits 
		indices     A vector of row indices where the inits should be written, or None to append at end of tensor
	"""
	subspace = wsm._subspace
	tensor = get_tensor(tr, wsm, name)
	codec = shoji.Codec(tensor.dtype)
	assert in_tensor.inits is not None

	if tensor.rank == 0:  # It's a scalar value
		key = subspace.pack(("tensor_values", name) + (0, 0))
		tr[key] = codec.encode(np.array(in_tensor.inits))
		return
		
	is_update = True  
	if indices is None:  # We're appending to the end of the tensor
		is_update = False
		new_length = len(in_tensor) + tensor.shape[0]
		indices = np.arange(tensor.shape[0], new_length)
		# Update the tensor length
		new_tensor = copy.copy(tensor)
		new_tensor.shape = (new_length,) + in_tensor.shape[1:]
		create_or_update_tensor(tr, wsm, name, new_tensor)

	# Update the index
	if tensor.rank == 1:
		if is_update:
			old_vals = read_tensor_values(tr, wsm, name, tensor, indices)
			for i, ix in enumerate(indices):
				key = subspace.pack(("tensor_indexes", name, coerce_dtype(tensor.dtype, old_vals[i]), int(ix)))
				del tr[key]

		for i, value in enumerate(in_tensor.inits):
			key = subspace.pack(("tensor_indexes", name, coerce_dtype(tensor.dtype, value), int(indices[i])))
			tr[key] = b''

	if not tensor.jagged:
		assert(isinstance(in_tensor.inits, np.ndarray))
		rows_per_chunk = max(1, int(np.floor(CHUNK_SIZE / (in_tensor.inits.size // in_tensor.inits.shape[0]))))
		if rows_per_chunk > 1:
			chunks = indices // rows_per_chunk
			for chunk in np.unique(chunks):
				vals = in_tensor.inits[chunks == chunk]
				if len(vals) < rows_per_chunk:
					# Need to read the previous tensor values and update them first
					prev = tr[subspace.pack(("tensor_values", name, int(chunk), 0))]
					if prev.present():
						prev_vals = codec.decode(prev.value)
					else:
						prev_vals = np.zeros((rows_per_chunk,) + in_tensor.shape[1:], dtype=tensor.numpy_dtype())
					ixs = indices[chunks == chunk]
					prev_vals[np.mod(ixs, rows_per_chunk)] = vals
					vals = prev_vals
				key = subspace.pack(("tensor_values", name, int(chunk), 0))
				tr[key] = codec.encode(vals)
			return
		# Falls through to the code below

	# Jagged or only one row per chunk
	for i, ix in enumerate(indices):
		encoded = codec.encode(np.array(in_tensor.inits[i]))
		for j in range(0, len(encoded), CHUNK_SIZE):
			key = subspace.pack(("tensor_values", name, int(ix), j // CHUNK_SIZE))
			tr[key] = encoded[j:j + CHUNK_SIZE]


@numba.jit
def compute_ranges(elements):
    elements = np.sort(elements)
    ranges = []
    start = elements[0]
    for ix in range(1, len(elements)):
        if elements[ix] != elements[ix - 1] + 1:
            ranges.append((start, elements[ix - 1]))
            start = elements[ix]
    ranges.append((start, elements[-1]))
    return ranges


@fdb.transactional
def read_chunked_rows(tr: fdb.impl.Transaction, subspace: fdb.subspace_impl.Subspace, name: str, i: int, j: int, codec: shoji.Codec) -> List[np.ndarray]:
	start = subspace.range(("tensor_values", name, i)).start
	stop = subspace.range(("tensor_values", name, j)).stop
	result = []
	ix = i
	encoded = bytearray()
	# TODO: use parallelism with futures instead
	for k, v in tr.get_range(start, stop, streaming_mode=fdb.StreamingMode.want_all):
		row = subspace.unpack(k)[-2]
		if row != ix:
			result.append(codec.decode(bytes(encoded)))
			encoded = bytearray()
			ix = row
		encoded += v
	result.append(codec.decode(bytes(encoded)))
	return result


@fdb.transactional
def read_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor = None, indices: np.ndarray = None) -> np.ndarray:
	subspace = wsm._subspace
	if tensor is None:
		tensor = get_tensor(tr, wsm, name)

	# Convert the list of indices to ranges as far as possible
	if indices is None:
		if tensor.rank > 0:
			n_rows = tensor.shape[0]
			indices = np.arange(0, n_rows)
			ranges = [(0, n_rows)]
		else:
			n_rows = 1
			indices = np.array([0], dtype="int")
			ranges = [(0, 0)]
	else:
		n_rows = len(indices)
		ranges = compute_ranges(indices)

	codec = shoji.Codec(tensor.dtype)

	if tensor.rank == 0:  # It's a scalar value
		key = subspace.pack(("tensor_values", name) + (0, 0))
		return codec.decode(tr[key].value).item()

	if tensor.jagged:
		resultj: List[np.ndarray] = []
		for (start, stop) in ranges:
			resultj += read_chunked_rows(tr, subspace, name, start, stop, codec)
		return resultj
	rows_per_chunk = max(1, int(np.floor(CHUNK_SIZE / (np.prod(tensor.shape) // tensor.shape[0]))))
	if rows_per_chunk == 1:
		result = np.empty((n_rows,) + tensor.shape[1:], dtype=tensor.numpy_dtype())
		ix = 0
		for (start, stop) in ranges:
			vals = read_chunked_rows(tr, subspace, name, start, stop, codec)
			result[ix: ix + len(vals)] = vals
		return result
	else:  # A dense array (not jagged or scalar) with more than one row per chunk
		chunks = indices // rows_per_chunk
		result = np.empty((n_rows,) + tensor.shape[1:], dtype=tensor.numpy_dtype())
		i = 0
		# Use parallelism with futures
		r = {}
		unique_chunks = np.unique(chunks)
		for chunk in unique_chunks:
			key = subspace.pack(("tensor_values", name, int(chunk), 0))
			r[chunk] = tr[key]  # This returns a Future
		for chunk in unique_chunks:
			vals = codec.decode(r[chunk].value)  # This blocks reading the Future
			ixs = indices[chunks == chunk]
			vals = vals[np.mod(ixs, rows_per_chunk)]  # Extract the relevant rows from the chunk
			result[i: i + len(vals)] = vals
			i += len(vals)
		return result


@fdb.transactional
def const_compare(tr, wsm: shoji.WorkspaceManager, name: str, operator: str, const: Tuple[int, str, float]) -> np.ndarray:
	"""
	Compare a tensor to a constant value, and return all indices that match
	"""
	# Code for range, equality and inequality filters
	tensor = wsm[name]
	assert isinstance(tensor, shoji.Tensor)
	const = tensor.python_dtype()(const)  # Cast the const to string, float or int
	index = wsm._subspace["tensor_indexes"][name]
	eq_range = index[const].range()
	all_range = index.range()
	start, stop = all_range.start, all_range.stop
	if operator == "==":
		start, stop = eq_range.start, eq_range.stop
	if operator == ">=":
		start = eq_range.start
	elif operator == ">":
		start = tr.get_key(fdb.KeySelector.first_greater_than(index[const]))
	elif operator == "<=":
		stop = eq_range.stop
	elif operator == "<":
		stop = tr.get_key(fdb.KeySelector.last_less_than(index[const]))
	return np.array([index.unpack(k)[1] for k, _ in tr[start:stop]], dtype="int64")


@fdb.transactional
def append_tensors(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, dname: str, vals: Dict[str, Union[List[np.ndarray], np.ndarray]]) -> None:
	"""
	Append values to a set of named tensors, which share their first dimension

	Args:
		tr: Transaction
		wsm: `shoji.workspace.WorkspaceManager`
		dname: Name of the dimension along which the values should be appended
		vals: dict of tensor names and corresponding values (np.ndarray or list of np.ndarray)

	Remarks:
		The function will check for the existence of all the tensors, that they have the same first dimension,
		and that no other tensor in the workspace has the same first dimension. It will also check that the values
		given have the same length along the first dimension, and that all other dimensions of each tensor match the 
		definitions of those tensors.
	"""
	subspace = wsm._subspace
	tensors: Dict[str, shoji.Tensor] = {}

	# Check that all named tensors exist, and have the right first dimension
	for name in vals.keys():
		t = get_tensor(tr, wsm, name)
		if t is None:
			raise NameError(f"Tensor '{name}' does not exist in the workspace")
		tensors[name] = t  # type: ignore
		if tensors[name].rank == 0 or tensors[name].dims[0] != dname:
			raise ValueError(f"Tensor '{name}' does not have '{dname}' as first dimension")

	# Check the rank of the values
	new_length = 0
	for name, values in vals.items():
		if tensors[name].jagged:
			for row in values:
				if tensors[name].rank != row.ndim + 1:  # type: ignore
					raise ValueError(f"Tensor '{name}' of rank {tensors[name].rank} cannot be initialized with rank-{row.ndim + 1} array")  # type: ignore
		else:
			if tensors[name].rank != values.ndim:  # type: ignore
				raise ValueError(f"Tensor '{name}' of rank {tensors[name].rank} cannot be initialized with rank-{values.ndim} array")  # type: ignore
		new_length = tensors[name].shape[0] + len(values)

	# Check that all relevant tensors will have the right shape after appending
	all_tensors: List[str] = [subspace["tensors"].unpack(k)[0] for k,v in tr[subspace["tensors"].range()]]
	for tensor_name in all_tensors:
		if tensor_name not in vals:
			tensor: shoji.Tensor = wsm[tensor_name]  # type: ignore
			if tensor.rank != 0 and tensor.dims[0] == dname:
				if tensor.shape[0] != new_length:
					raise ValueError(f"Length {tensor.shape[0]} of tensor '{tensor.name}' would conflict with dimension '{dname}' length {new_length} after appending {','.join(vals.keys())}")

	# Check that all the other dimensions are the correct shape according to their definitions
	for name, values in vals.items():
		for i, idim in enumerate(tensors[name].dims):
			if i == 0 or idim is None:
				continue
			elif isinstance(idim, int):  # Anonymous fixed-shape dimension
				target_size = idim
			elif isinstance(idim, str):  # Named dimension
				dim = get_dimension(tr, wsm, idim)
				if dim is None:
					raise KeyError(f"Dimension {idim} is undefined")
				if dim.shape is None:
					continue
				target_size = dim.shape
			if tensors[name].jagged:
				for row in values:
					if row.shape[i] != target_size: 
						raise ValueError(f"Tensor '{name}' dimension '{idim}' must be exactly {target_size} elements long")
			elif target_size != values.shape[i]:  # type: ignore
				raise ValueError(f"Tensor '{name}' dimension {i} ('{idim}') must be exactly {target_size} elements long")
				
	# Write the values
	for name, values in vals.items():
		tensors[name].inits = values
		if isinstance(values, (list, tuple)):
			tensors[name].shape = (len(values), ) + (None, ) * (values[0].ndim - 1)  # TODO: handle jagged tensors
		else:
			tensors[name].shape = values.shape  # TODO: handle jagged tensors
		write_tensor_values(tr, wsm, name, tensors[name])

	# Update the first dimension
	dim = wsm[dname]  # type: ignore
	dim.length = dim.length + len(values)
	create_or_update_dimension(tr, wsm, dname, dim)


def __nil__():
	pass
