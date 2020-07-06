"""
Internal low-level I/O routines, not intended for end users.
"""

from typing import Union, Optional, Tuple, Dict, List
import fdb
import numpy as np
import shoji
import numba
import struct


@fdb.transactional
def name_exists(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Optional[str]:
	subspace = wsm._subspace
	if subspace.exists(tr, wsm._path + (name,)):
		return "Workspace"
	if tr[subspace["dimensions"][name]].present():
		return "Dimension"
	for _ in tr[subspace.range(("tensors", name))]:
		return "Tensor"
	return None

@fdb.transactional
def read_entity(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str) -> Union[shoji.Dimension, shoji.WorkspaceManager, shoji.Tensor, None]:
	subspace = wsm._subspace
	if subspace.exists(tr, name):
		child = subspace.open(tr, name)
		return shoji.WorkspaceManager(wsm._db, child, wsm._path + (name,))
	elif tr[subspace["dimensions"][name]].present():
		val = tr[subspace["dimensions"][name]]
		dim = shoji.Dimension(shape=int.from_bytes(val[:8], "little", signed=True), length=int.from_bytes(val[8:], "little", signed=False))
		dim.assigned_name = name
		dim.wsm = wsm
		return dim
	else:
		# key_tuple = ("tensors", name, tensor.dtype, tensor.rank) + tensor.dims + (jagged, length)
		tensor_tuples = tr[subspace.range(("tensors", name))]
		for k, _ in tensor_tuples:
			key = subspace.unpack(k)
			tensor = shoji.Tensor(key[2], key[4:4+key[3]], length=key[-1])
			tensor.jagged = key[-2] == 1
			tensor.assigned_name = name
			tensor.wsm = wsm
			return tensor
	raise KeyError(f"{name} not found")

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
	existing_name = name_exists(tr, wsm, name)
	if existing_name is not None:
		if existing_name != "Dimension":
			raise AttributeError(f"Name already exists (as {existing_name})")
		# Update an existing dimension
		prev_dim: shoji.Dimension = read_entity(tr, wsm, name)
		if prev_dim.shape != dim.shape:
			raise AttributeError(f"Cannot modify shape of existing dimension '{name}'")
	# Create or update the dimension
	tr[subspace["dimensions"][name]] = (dim.shape if dim.shape is not None else -1).to_bytes(8, "little", signed=True) + dim.length.to_bytes(8, "little", signed=False)


@fdb.transactional
def create_tensor(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor) -> None:
	subspace = wsm._subspace
	# Check that name doesn't already exist
	existing_name = name_exists(tr, wsm, name)
	if existing_name is not None:
		raise AttributeError(f"Name already exists (as {existing_name})")
	# Check that the dimensions of the tensor exist
	for ix, d in enumerate(tensor.dims):
		if isinstance(d, str):
			dim = wsm[d]
			assert isinstance(dim, shoji.Dimension)
			# Check that the dimensions of the tensor match the shape of the tensor
			if dim.shape is None:
				if ix > 0:
					tensor.jagged = True
			else:
				if tensor.inits is not None and tensor.shape[ix] != dim.shape:
					raise IndexError(f"Mismatch between the declared shape {dim.shape} of dimension '{d}' and the shape {tensor.shape} of values")
	# Store tensor definition
	# ("tensors", name, dtype, rank) + dims + (jagged, length)
	key_tuple = ("tensors", name, tensor.dtype, tensor.rank) + tensor.dims + (1 if tensor.jagged else 0, 0)
	key = subspace.pack(key_tuple)
	tr[key] = b''


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
def write_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, tensor: shoji.Tensor) -> None:
	CHUNK_SIZE = 1_000
	subspace = wsm._subspace
	if tensor.inits is not None:
		codec = shoji.Codec(tensor.dtype)
		if np.ndim(tensor.inits) == 0:  # It's a scalar value
			key = subspace.pack(("tensor_values", name) + (0, 0))
			tr[key] = codec.encode(np.array(tensor.inits))
		else:
			# Update the length
			old_length: int = wsm[name].length  # type: ignore
			length = len(tensor.inits) + old_length  # type: ignore
			tr.clear_range_startswith(subspace["tensors"][name].key())
			key_tuple = ("tensors", name, tensor.dtype, tensor.rank) + tensor.dims + (1 if tensor.jagged else 0, length)
			key = subspace.pack(key_tuple)
			tr[key] = b''
			if tensor.rank == 1:
				# Save the values unchunked
				for i in range(len(tensor.inits)):
					key = subspace.pack(("tensor_values", name, i + old_length, 0))
					x = tensor.inits[i]
					if tensor.dtype == "string":
						tr[key] = x.encode()
					elif tensor.dtype == "float32":
						tr[key] = struct.pack("f", x)
					elif tensor.dtype == "float64":
						tr[key] = struct.pack("d", x)
					elif tensor.dtype == "uint16":
						tr[key] = int(x).to_bytes(2, "little", signed=False)
					elif tensor.dtype == "uint32":
						tr[key] = int(x).to_bytes(4, "little", signed=False)
					elif tensor.dtype == "uint64":
						tr[key] = int(x).to_bytes(8, "little", signed=False)
					elif tensor.dtype == "int16":
						tr[key] = int(x).to_bytes(2, "little", signed=True)
					elif tensor.dtype == "int32":
						tr[key] = int(x).to_bytes(4, "little", signed=True)
					elif tensor.dtype == "int64":
						tr[key] = int(x).to_bytes(8, "little", signed=True)
					elif tensor.dtype == "bool":
						tr[key] = int(x).to_bytes(1, "little", signed=False)

				# Create an index
				values = [coerce_dtype(tensor.dtype, v) for v in tensor.inits]
				for i, value in enumerate(values):
					key = subspace.pack(("tensor_indexes", name, value, int(i + old_length)))
					tr[key] = b''
			else:
				for i in range(len(tensor.inits)):
					encoded = codec.encode(np.array(tensor.inits[i]))
					for j in range(0, len(encoded), CHUNK_SIZE):
						key = subspace.pack(("tensor_values", name, i + old_length, j // CHUNK_SIZE))
						tr[key] = encoded[j:j+CHUNK_SIZE]


@fdb.transactional
def update_tensor_values(tr: fdb.impl.Transaction, wsm: shoji.WorkspaceManager, name: str, indices: Tuple[slice, np.ndarray], tensor: shoji.Tensor) -> None:
	CHUNK_SIZE = 1_000
	subspace = wsm._subspace
	codec = shoji.Codec(tensor.dtype)
	# Better read the tensor metadata again inside the transaction
	temp = tensor.inits
	tensor: shoji.Tensor = wsm[name]
	tensor.inits = temp

	if isinstance(indices[0], slice):
		s = indices[0].indices(tensor.length)
		rows = np.arange(s[0], s[1], s[2])
	else:
		rows = indices[0]

	if np.ndim(tensor.inits) == 0:  # It's a scalar value
		assert len(indices) != 0, "Cannot index scalar value"
		key = subspace.pack(("tensor_values", name) + (0, 0))
		tr[key] = codec.encode(np.array(tensor.inits))
	else:
		# Read the current data that we want to modify
		if isinstance(indices[0], slice):
			prev = read_filtered_tensor(tr, subspace, name, tensor, np.arange(indices[0].start, indices[0].stop, indices[0].step))
		else:
			prev = read_filtered_tensor(tr, subspace, name, tensor, indices[0])

		# Modify the data row by row (in case the tensor is jagged)
		for ix in range(len(prev)):
			row = prev[ix]
			
			if tensor.rank == 1:
				# Remove index entries
				key = subspace.pack(("tensor_indexes", name, coerce_dtype(tensor.dtype, row), int(rows[ix])))
				del tr[key]
				prev[ix] = tensor.inits[ix]
			else:
				# Expand any slices in the indices
				actual_indices = []
				for i, ind in enumerate(indices[1:]):
					if isinstance(ind, slice):
						s = ind.indices(prev.shape[i + 1])
						actual_indices.append(np.arange(s[0], s[1], s[2]))
					else:
						actual_indices.append(ind)
				row[actual_indices] = tensor.inits

		# Now save the tensor back to the appropriate rows
		if tensor.rank == 1:
			for i, j in enumerate(rows):
				key = subspace.pack(("tensor_values", name, int(j), 0))
				x = tensor.inits[i]
				if tensor.dtype == "string":
					tr[key] = x.encode()
				elif tensor.dtype == "float32":
					tr[key] = struct.pack("f", x)
				elif tensor.dtype == "float64":
					tr[key] = struct.pack("d", x)
				elif tensor.dtype == "uint16":
					tr[key] = int(x).to_bytes(2, "little", signed=False)
				elif tensor.dtype == "uint32":
					tr[key] = int(x).to_bytes(4, "little", signed=False)
				elif tensor.dtype == "uint64":
					tr[key] = int(x).to_bytes(8, "little", signed=False)
				elif tensor.dtype == "int16":
					tr[key] = int(x).to_bytes(2, "little", signed=True)
				elif tensor.dtype == "int32":
					tr[key] = int(x).to_bytes(4, "little", signed=True)
				elif tensor.dtype == "int64":
					tr[key] = int(x).to_bytes(8, "little", signed=True)
				elif tensor.dtype == "bool":
					tr[key] = int(x).to_bytes(1, "little", signed=False)

			# Create index entries
			values = [coerce_dtype(tensor.dtype, v) for v in tensor.inits]
			for i, value in enumerate(values):
				key = subspace.pack(("tensor_indexes", name, value, int(rows[i])))
				tr[key] = b''
		else:
			for i, j in enumerate(rows):
				encoded = codec.encode(np.array(tensor.inits[i]))
				for k in range(0, len(encoded), CHUNK_SIZE):
					key = subspace.pack(("tensor_values", name, int(j), k // CHUNK_SIZE))
					tr[key] = encoded[k:k + CHUNK_SIZE]


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
	for k, v in tr.get_range(start, stop):
		row = subspace.unpack(k)[-2]
		if row != ix:
			result.append(codec.decode(bytes(encoded)))
			encoded = bytearray()
			ix = row
		encoded += v
	result.append(codec.decode(bytes(encoded)))
	return result

@fdb.transactional
def read_unchunked_rows(tr: fdb.impl.Transaction, subspace: fdb.subspace_impl.Subspace, name: str, i: int, j: int, dtype: str) -> Union[List[float], List[int], List[str], List[bool]]:
	start = subspace.range(("tensor_values", name, i)).start
	stop = subspace.range(("tensor_values", name, j)).stop
	vals = [v for _,v in tr.get_range(start, stop)]
	if dtype == "string":
		return [x.decode() for x in vals]
	if dtype == "float32":
		return [struct.unpack("f", x)[0] for x in vals]
	if dtype == "float64":
		return [struct.unpack("d", x)[0] for x in vals]
	if dtype == "bool":
		return [bool(int.from_bytes(x, "little", signed=False)) for x in vals]
	signed = True
	if dtype[0] == "u":
		signed = False
	return [int.from_bytes(x, "little", signed=signed) for x in vals]


@fdb.transactional
def read_filtered_tensor(tr: fdb.impl.Transaction, subspace: fdb.subspace_impl.Subspace, name: str, tensor: shoji.Tensor, indices: np.ndarray = None) -> np.ndarray:
	assert tensor.length is not None
	# Convert the list of indices to ranges as far as possible
	if indices is None:
		ranges = [(0, tensor.length)]
		n_rows = tensor.length
	else:
		ranges = compute_ranges(indices)
		n_rows = len(indices)
	codec = shoji.Codec(tensor.dtype)

	if tensor.jagged:
		resultj: List[np.ndarray] = []
		for (start, stop) in ranges:
			resultj += read_chunked_rows(tr, subspace, name, start, stop, codec)
		return resultj
	else:
		if tensor.rank == 1:
			result = np.empty(n_rows, dtype=tensor.numpy_dtype())
			ix = 0
			for (start, stop) in ranges:
				rows = read_unchunked_rows(tr, subspace, name, start, stop, dtype=tensor.dtype)
				result[ix: ix + (stop - start + 1)] = rows
				ix += stop - start + 1
		else:
			result = None
			ix = 0
			for start, stop in ranges:
				rows = read_chunked_rows(tr, subspace, name, start, stop, codec)
				if result is None:
					result = np.empty((n_rows,) + rows[0].shape, dtype=rows[0].dtype)
				result[ix: ix + (stop - start + 1)] = np.array(rows)
				ix += (stop - start + 1)
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
		if name_exists(tr, wsm, name) != "Tensor":
			raise NameError(f"Tensor '{name}' does not exist in the workspace")
		tensors[name] = wsm[name]  # type: ignore
		if tensors[name].rank == 0 or tensors[name].dims[0] != dname:
			raise ValueError(f"Tensor '{name}' does not have '{dname}' as first dimension")

	# Check that all relevant tensors have been included in the values
	all_tensors: List[str] = [subspace["tensors"].unpack(k)[0] for k,v in tr[subspace["tensors"].range()]]
	for tensor_name in all_tensors:
		if tensor_name not in vals:
			tensor: shoji.Tensor = wsm[tensor_name]  # type: ignore
			if tensor.rank != 0 and tensor.dims[0] == dname:
				raise ValueError(f"Tensor '{tensor.assigned_name}' missing from values in append operation")

	# Check the rank of the values
	for name, values in vals.items():
		if tensors[name].jagged:
			for row in values:
				if tensors[name].rank != row.ndim + 1:  # type: ignore
					raise ValueError(f"Tensor '{name}' of rank {tensors[name].rank} cannot be initialized with rank-{row.ndim + 1} array")  # type: ignore
		else:
			if tensors[name].rank != values.ndim:  # type: ignore
				raise ValueError(f"Tensor '{name}' of rank {tensors[name].rank} cannot be initialized with rank-{values.ndim} array")  # type: ignore

	# Check that all the other dimensions are the correct shape according to their definitions
	for name, values in vals.items():
		for i, idim in enumerate(tensors[name].dims):
			if i == 0 or idim is None:
				continue
			elif isinstance(idim, int):  # Anonymous fixed-shape dimension
				pass  # This constraint was already checked when the tensor was created
			elif isinstance(idim, str):  # Named dimension
				dim: shoji.Dimension = wsm[idim]  # type: ignore
				if dim.shape is None:
					continue
				if tensors[name].jagged:
					for row in values:
						if row.shape[i] != dim.shape: 
							raise ValueError(f"Tensor '{name}' dimension '{idim}' must be exactly {dim.shape} elements long")
				elif dim.shape != values.shape[i]:  # type: ignore
					raise ValueError(f"Tensor '{name}' dimension {i} ('{idim}') must be exactly {dim.shape} elements long")
				
	# Write the values
	for name, values in vals.items():
		tensors[name].inits = values
		write_tensor_values(tr, wsm, name, tensors[name])

	# Update the first dimension
	dim = wsm[dname]  # type: ignore
	dim.length = dim.length + len(values)
	create_or_update_dimension(tr, wsm, dname, dim)


def __nil__():
	pass
