"""
Internal low-level I/O routines, not intended for end users.
"""

from typing import Union, Optional, Tuple, Dict, List, Type
from enum import IntFlag
import fdb
import logging
import numpy as np
import blosc
import shoji
import pickle
import copy
CHUNK_SIZE = 1_000


class TensorFlags(IntFlag):
	Empty = 0
	Jagged = 1
	Initializing = 2  # Set if the tensor is not yet fully initialized
	Chunked = 4  # New style of chunking for improved random-access along second dimension


@fdb.transactional
def get_entity(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[str]:
	t = get_tensor(tr, wsm, name)
	if t is not None:
		return t
	d = get_dimension(tr, wsm, name)
	if d is not None:
		return d
	s = get_subdir(tr, wsm, name)
	if s is not None:
		return s
	return None


@fdb.transactional
def get_subdir(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[shoji.WorkspaceManager]:
	subdir = wsm._subdir
	if subdir.exists(tr, name):
		child = subdir.open(tr, name)
		wsm = shoji.WorkspaceManager(wsm._db, child, wsm._path + (name,))
		wsm._name = name
		return wsm
	return None


@fdb.transactional
def get_dimension(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[shoji.Dimension]:
	subdir = wsm._subdir
	val = tr[subdir["dimensions"][name]]
	if val.present():
		val = tr[subdir["dimensions"][name]]
		dim = shoji.Dimension(shape=int.from_bytes(val[:8], "little", signed=True), length=int.from_bytes(val[8:], "little", signed=False))
		dim.name = name
		dim.wsm = wsm
		return dim
	return None


@fdb.transactional
def get_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, include_initializing: bool = False) -> Optional[shoji.Tensor]:
	subdir = wsm._subdir
	# ("tensors", name) = (tensor.dtype, tensor.rank, tensor.jagged) + tensor.dims + shape + chunks
	# where shape[0] == -1 if jagged
	val = tr[subdir.pack(("tensors", name))]
	if val.present():
		t = pickle.loads(val.value)
		rank = t[1]
		if rank > 0:
			shape = t[3 + rank: 3 + 2 * rank]
		else:
			shape = ()
		tensor = shoji.Tensor(t[0], t[3:3 + rank], shape=shape)
		flags = TensorFlags(t[2])
		tensor.jagged = TensorFlags.Jagged in flags
		if TensorFlags.Initializing in flags and not include_initializing:
			return None
		tensor.name = name
		tensor.wsm = wsm
		return tensor
	return None


@fdb.transactional
def list_workspaces(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager) -> List[str]:
	return wsm._subdir.list(tr)


@fdb.transactional
def list_dimensions(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager) -> List[shoji.Dimension]:
	result = []
	for kv in tr[wsm._subdir["dimensions"].range()]:
		dim = shoji.Dimension(shape=int.from_bytes(kv.value[:8], "little", signed=True), length=int.from_bytes(kv.value[8:], "little", signed=False))
		dim.name = wsm._subdir["dimensions"].unpack(kv.key)[0]
		dim.wsm = wsm
		result.append(dim)
	return result


@fdb.transactional
def list_tensors(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, include_initializing: bool = False) -> List[shoji.Tensor]:
	result = []
	for kv in tr[wsm._subdir["tensors"].range()]:
		t = pickle.loads(kv.value)
		rank = t[1]
		if rank > 0:
			shape = t[3 + rank: 3 + 2 * rank]
		else:
			shape = ()
		tensor = shoji.Tensor(t[0], t[3:3 + rank], shape=shape)
		flags = TensorFlags(t[2])
		tensor.jagged = TensorFlags.Jagged in flags
		tensor.name = wsm._subdir["tensors"].unpack(kv.key)[0]
		if TensorFlags.Initializing in flags and not include_initializing:
			continue
		tensor.wsm = wsm
		result.append(tensor)
	return result


@fdb.transactional
def delete_entity(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> None:
	subdir = wsm._subdir
	if subdir.exists(tr, name):
		subdir.open(tr, name).remove(tr)
	elif tr[subdir["dimensions"][name]].present():
		tr.clear_range_startswith(subdir["dimensions"][name].key())
	elif tr[subdir["tensors"][name]].present():
		tr.clear_range_startswith(subdir["tensors"][name].key())
		tr.clear_range_startswith(subdir["tensor_values"][name].key())
		tr.clear_range_startswith(subdir["tensor_indexes"][name].key())


@fdb.transactional
def create_or_update_dimension(tr, wsm: shoji.WorkspaceManager, name: str, dim: shoji.Dimension):
	subdir = wsm._subdir
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
				all_tensors: List[shoji.Tensor] = list_tensors(tr, wsm)
				for tensor in all_tensors:
					if tensor.rank > 0 and tensor.dims[0] == name:
						if tensor.shape[0] != dim.shape:
							raise AttributeError(f"New shape {dim.shape} of existing dimension '{name}' conflicts with length {tensor.shape[0]} of existing tensor '{tensor.name}'")
				dim.length = dim.shape
			else:
				dim.length = prev_dim.length
	# Create or update the dimension
	tr[subdir["dimensions"][name]] = (dim.shape if dim.shape is not None else -1).to_bytes(8, "little", signed=True) + dim.length.to_bytes(8, "little", signed=False)


@fdb.transactional
def create_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor) -> None:
	"""
	Creates a new tensor (but does not write the inits)

	If inits were provided, the tensor is marked as Initializing, and will be invisible until the inits have been written
	"""
	subdir = wsm._subdir
	# Check that name doesn't already exist
	existing = get_entity(tr, wsm, name)
	if existing is not None:
		raise AttributeError(f"Name already exists (as {existing})")
	else:
		# Check that the dimensions of the tensor exist
		for ix, d in enumerate(tensor.dims):
			if isinstance(d, str):
				dim = get_dimension(tr, wsm, d)
				if dim is None:
					raise KeyError(f"Tensor dimension '{d}' is not defined")
				if dim.shape is not None:  # This is a fixed-length dimension
					if ix == 0 and tensor.inits is None:
							raise ValueError(f"Values (inits) must be provided for Tensor '{name}' with first dimension '{dim.name}' declared with fixed length {dim.shape}")
					elif tensor.inits is not None:
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
	key = subdir.pack(("tensors", name))
	tensor.shape = (0,) + tensor.shape[1:]  # The first dimension will be updated by write_tensor_values after values have been actually appended
	flags = TensorFlags.Empty
	if tensor.jagged:
		flags |= TensorFlags.Jagged
	if tensor.inits is not None:
		flags |= TensorFlags.Initializing
	if tensor.chunked:
		flags |= TensorFlags.Chunked
	data = pickle.dumps((tensor.dtype, tensor.rank, int(flags)) + tensor.dims + tensor.shape)
	tr[key] = data


def initialize_tensor(wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor):
	if tensor.inits is not None:
		if tensor.rank == 0:
			write_tensor_values(wsm._db.transaction, wsm, name, tv=tensor.inits, tensor=tensor)
		else:
			n_rows = len(tensor.inits)
			n_bytes = tensor.inits.size_in_bytes()

			if isinstance(tensor.dims[0], str):
				dim = wsm._get_dimension(tensor.dims[0])
				if dim.length != 0 and dim.length != len(tensor.inits):
					raise ValueError(f"Length {len(tensor.inits)} of new tensor '{name}' does not match length {dim.length} of first dimension '{tensor.dims[0]}' ")

			n_bytes_per_transaction = 1_000_000  # Starting point, but we'll adapt it below
			ix: int = 0
			while ix < n_rows:
				n_rows_per_transaction = int(max(1, n_rows / (n_bytes / n_bytes_per_transaction)))
				batch = tensor.inits.values[ix: ix + n_rows_per_transaction]
				try:
					tensor.shape = (ix,) + tensor.inits.shape[1:]
					n_bytes_written = write_tensor_values(wsm._db.transaction, wsm, name, tv=shoji.TensorValue(batch), tensor=tensor, initializing=True)
				except fdb.impl.FDBError as e:
					if e.code in (1004, 1007, 1031, 2101) and n_rows_per_transaction > 1:  # Too many bytes or too long time, so try again with less
						n_bytes_per_transaction = max(1, n_bytes_per_transaction // 2)
						continue
					else:
						raise e
				if n_bytes_written < n_bytes_per_transaction // 2:  # Not enough bytes, so increase for next round
					n_bytes_per_transaction *= 2
				ix += n_rows_per_transaction
		# Complete the intitalization in one atomic operation
		finish_initialization(wsm._db.transaction, wsm, name, tensor)


@fdb.transactional
def finish_initialization(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor) -> None:
	assert tensor.inits is not None
	# Update the tensor definition to clear the Initializing flag
	subdir = wsm._subdir
	key = subdir.pack(("tensors", name))
	flags = TensorFlags.Empty
	if tensor.jagged:
		flags |= TensorFlags.Jagged
	if tensor.chunked:
		flags |= TensorFlags.Chunked
	data = pickle.dumps((tensor.dtype, tensor.rank, int(flags)) + tensor.dims + tensor.inits.shape)
	tr[key] = data

	# Update the first dimension
	if tensor.rank > 0 and isinstance(tensor.dims[0], str):
		dim = wsm._get_dimension(tensor.dims[0])
		# Redo this test here, since dimension might have changed while we were writing the inits
		if dim.length != 0 and dim.length != len(tensor.inits):
			raise ValueError(f"Length {len(tensor.inits)} of new tensor '{name}' does not match length {dim.length} of first dimension '{tensor.dims[0]}' ")
		if dim.length == 0:
			dim.length = len(tensor.inits)
			create_or_update_dimension(tr, wsm, tensor.dims[0], dim)


@fdb.transactional
def update_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor, initializing: bool = False) -> int:
	# Store tensor definition (overwriting any existing definition)
	# ("tensors", name) = (dtype, rank, jagged) + dims + shape  
	# where shape[0] == -1 if jagged, and tuple value is encoded using pickle
	subdir = wsm._subdir
	key = subdir.pack(("tensors", name))
	flags = TensorFlags.Empty
	if tensor.jagged:
		flags |= TensorFlags.Jagged
	if initializing:
		flags |= TensorFlags.Initializing
	if tensor.chunked:
		flags |= TensorFlags.Chunked
	data = pickle.dumps((tensor.dtype, tensor.rank, int(flags)) + tensor.dims + tensor.shape)
	tr[key] = data

	return len(key) + len(data)


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
def write_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tv: shoji.TensorValue, indices: np.ndarray = None, tensor: shoji.Tensor = None, initializing: bool = False) -> int:
	"""
	Args:
		tr          Transaction
		wsm         WorkspaceManager
		name        Name of the tensor in the database
		tv      	Tensor values to write
		indices     A vector of row indices where the inits should be written, or None to append at end of tensor
	"""
	n_bytes_written = 0

	subdir = wsm._subdir
	if tensor is None:
		tensor = get_tensor(tr, wsm, name)
	
	if tensor.rank == 0:  # It's a scalar value
		key = subdir.pack(("tensor_values", name) + (0, 0))
		encoded = blosc.pack_array(np.array(tv.values))
		tr[key] = encoded
		n_bytes_written += len(encoded) + len(key)
		return n_bytes_written
		
	is_update = True
	if indices is None:  # We're appending to the end of the tensor
		is_update = False
		new_length = len(tv) + tensor.shape[0]
		indices = np.arange(tensor.shape[0], new_length)
		# Update the tensor length
		new_tensor = copy.copy(tensor)
		if tensor.shape[0] == 0:
			new_tensor.shape = tv.shape
		else:
			new_tensor.shape = (new_length,) + tensor.shape[1:]
		n_bytes_written += update_tensor(tr, wsm, name, new_tensor, initializing=initializing)

	# Update the index
	tc = dtype_class(tensor.dtype)
	if tensor.rank == 1:
		if is_update:
			# Delete the old index keys
			old_vals = read_tensor_values(tr, wsm, name, tensor, indices)
			for i, ix in enumerate(indices):
				key = subdir.pack(("tensor_indexes", name, tc(old_vals[i]), int(ix)))
				n_bytes_written += len(key)
				del tr[key]

		for i, value in enumerate(tv.values):
			key = subdir.pack(("tensor_indexes", name, tc(value), int(indices[i])))
			n_bytes_written += len(key)
			tr[key] = b''

	if not tensor.jagged:
		assert isinstance(tv.values, np.ndarray)
		rows_per_chunk = max(1, int(np.floor(CHUNK_SIZE / (tv.values.size // tv.values.shape[0]))))

		if rows_per_chunk > 1:
			chunks = indices // rows_per_chunk
			for chunk in np.unique(chunks):
				vals = tv.values[chunks == chunk]
				if len(vals) < rows_per_chunk:
					# Need to read the previous tensor values and update them first
					prev = tr[subdir.pack(("tensor_values", name, int(chunk), 0))]
					if prev.present():
						prev_vals = blosc.unpack_array(prev.value)
					else:
						if tensor.dtype == "string":
							prev_vals = np.full((rows_per_chunk,) + vals.shape[1:], "", dtype="object")
						else:
							prev_vals = np.zeros((rows_per_chunk,) + vals.shape[1:], dtype=tensor.numpy_dtype())
					ixs = indices[chunks == chunk]
					prev_vals[np.mod(ixs, rows_per_chunk)] = vals
					vals = prev_vals
				key = subdir.pack(("tensor_values", name, int(chunk), 0))
				encoded = blosc.pack_array(vals)
				tr[key] = encoded
				n_bytes_written += len(key) + len(encoded)
			return n_bytes_written
		# Falls through to the code below

	# Jagged or only one row per chunk
	for i, ix in enumerate(indices):
		if tensor.chunked:
			for j in range(0, len(vals), CHUNK_SIZE):
				encoded = blosc.pack_array(vals[j: j + CHUNK_SIZE])
				key = subdir.pack(("tensor_values", name, int(ix), int(j // CHUNK_SIZE)))
				n_bytes_written += len(key) + len(encoded)
				tr[key] = encoded
		else:
			encoded = blosc.pack_array(np.array(tv.values[i]))
			for j in range(0, len(encoded), CHUNK_SIZE):
				key = subdir.pack(("tensor_values", name, int(ix), int(j // CHUNK_SIZE)))
				n_bytes_written += len(key) + CHUNK_SIZE
				tr[key] = encoded[j:j + CHUNK_SIZE]
	return n_bytes_written

#@numba.jit
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
def read_chunked_rows(tr: fdb.impl.Transaction, subdir: fdb.directory_impl.DirectorySubspace, name: str, i: int, j: int, chunked: bool = False, indices_2nd_axis: np.ndarray = None) -> List[np.ndarray]:
	start = subdir.range(("tensor_values", name, i)).start
	stop = subdir.range(("tensor_values", name, j)).stop
	result = []
	ix = i
	if chunked:
		data: List[np.ndarray] = []
		for k, v in tr.get_range(start, stop, streaming_mode=fdb.StreamingMode.want_all):
			row = subdir.unpack(k)[-2]
			if row != ix:
				result.append(np.concatenate(data))
				data = []
				ix = row
			data.append(blosc.unpack_array(v))

		result.append(np.concatenate(data)[0])
	else:
		encoded = bytearray()
		for k, v in tr.get_range(start, stop, streaming_mode=fdb.StreamingMode.want_all):
			row = subdir.unpack(k)[-2]
			if row != ix:
				result.append(blosc.unpack_array(bytes(encoded)))
				encoded = bytearray()
				ix = row
			encoded += v
		result.append(blosc.unpack_array(bytes(encoded)))
	return result


@fdb.transactional
def read_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor = None, indices: np.ndarray = None, indices_2nd_axis: np.ndarray = None) -> np.ndarray:
	subdir = wsm._subdir
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
		if n_rows == 0:
			return np.zeros(0, dtype=tensor.numpy_dtype())
		ranges = compute_ranges(indices)

	if tensor.rank == 0:  # It's a scalar value
		key = subdir.pack(("tensor_values", name) + (0, 0))
		return blosc.unpack_array(tr[key].value).item()

	if not tensor.jagged and np.prod(tensor.shape) == 0:
		return np.zeros((n_rows,) + tensor.shape[1:], dtype=tensor.numpy_dtype())
	if tensor.jagged and tensor.shape[0] == 0:
		return [np.zeros(tensor.shape[1:], dtype=tensor.numpy_dtype()) for _ in range(n_rows)]

	if tensor.jagged:
		resultj: List[np.ndarray] = []
		for (start, stop) in ranges:
			resultj += read_chunked_rows(tr, subdir, name, int(start), int(stop), tensor.chunked)
		return resultj
	rows_per_chunk = max(1, int(np.floor(CHUNK_SIZE / (np.prod(tensor.shape) // tensor.shape[0]))))
	if rows_per_chunk == 1:
		result = np.empty((n_rows,) + tensor.shape[1:], dtype=tensor.numpy_dtype())
		ix = 0
		for (start, stop) in ranges:
			vals = read_chunked_rows(tr, subdir, name, int(start), int(stop), tensor.chunked, indices_2nd_axis)
			result[ix: ix + len(vals)] = vals
			ix += len(vals)
		return result
	else:  # A dense array (not jagged or scalar) with more than one row per chunk
		chunks = indices // rows_per_chunk
		result = np.empty((n_rows,) + tensor.shape[1:], dtype=tensor.numpy_dtype())
		i = 0
		# Use parallelism with futures
		r = {}
		unique_chunks = np.unique(chunks)
		for chunk in unique_chunks:
			key = subdir.pack(("tensor_values", name, int(chunk), 0))
			r[chunk] = tr[key]  # This returns a Future
		for chunk in unique_chunks:
			vals = blosc.unpack_array(r[chunk].value)  # This blocks reading the Future
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
	tensor = get_tensor(tr, wsm, name)
	const = tensor.python_dtype()(const)  # Cast the const to string, float or int
	index = wsm._subdir["tensor_indexes"][name]
	eq_range = index[const].range()
	all_range = index.range()
	start, stop = all_range.start, all_range.stop
	if operator == "!=":
		stop = tr.get_key(fdb.KeySelector.last_less_than(eq_range.start))
		a = np.array([index.unpack(k)[1] for k, _ in tr[start:stop]], dtype="int64")
		start = tr.get_key(fdb.KeySelector.first_greater_than(eq_range.stop))
		stop = all_range.stop
		b = np.array([index.unpack(k)[1] for k, _ in tr[start:stop]], dtype="int64")
		return np.concatenate([a, b])
	if operator == "==":
		start, stop = eq_range.start, eq_range.stop
	elif operator == ">=":
		start = eq_range.start
	elif operator == ">":
		start = tr.get_key(fdb.KeySelector.first_greater_than(eq_range.stop))
	elif operator == "<=":
		stop = eq_range.stop
	elif operator == "<":
		stop = tr.get_key(fdb.KeySelector.last_less_than(eq_range.start))
	return np.array([index.unpack(k)[1] for k, _ in tr[start:stop]], dtype="int64")


@fdb.transactional
def append_tensors(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, dname: str, vals: Dict[str, Union[List[np.ndarray], np.ndarray]]) -> int:
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
	n_bytes_written = 0
	all_tensors = {t.name: t for t in list_tensors(tr, wsm)}
	all_dims = {d.name: d for d in list_dimensions(tr, wsm)}
	tensors: Dict[str, shoji.Tensor] = {}

	# Check that all named tensors exist, and have the right first dimension
	for name in vals.keys():
		if name not in all_tensors:
			raise NameError(f"Tensor '{name}' does not exist in the workspace")
		tensors[name] = all_tensors[name]
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

	# Check that all relevant tensors will have the right length after appending
	for tensor in all_tensors.values():
		if tensor.name not in vals:
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
				if idim not in all_dims:
					raise KeyError(f"Dimension {idim} is undefined")
				dim = all_dims[idim]
				if dim.shape is None:
					continue
				target_size = dim.shape
			if tensors[name].jagged:
				for row in values:
					if row.shape[i - 1] != target_size: 
						raise ValueError(f"Tensor '{name}' dimension {i} ('{idim}') must be exactly {target_size} elements long, but shape was {row.shape}")
			elif target_size != values.shape[i]:  # type: ignore
				raise ValueError(f"Tensor '{name}' dimension {i} ('{idim}') must be exactly {target_size} elements long, but shape was {values.shape[i]}")

	# Write the values
	for name, values in vals.items():
		tv = shoji.TensorValue(values)
		n_bytes_written += write_tensor_values(tr, wsm, name, tv, None, all_tensors[name])

	# Update the first dimension
	dim = all_dims[dname]
	dim.length = dim.length + len(values)
	create_or_update_dimension(tr, wsm, dname, dim)

	return n_bytes_written

def __nil__():
	pass
