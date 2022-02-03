from typing import Tuple, Dict, List, Union
import shoji
import shoji.io
import fdb
import numpy as np
from .enums import Compartment


@fdb.transactional
def const_compare(tr, wsm: "shoji.Workspace", name: str, operator: str, const: Tuple[int, str, float]) -> np.ndarray:
	"""
	Compare a tensor to a constant value, and return all indices that match
	"""
	# Code for range, equality and inequality filters
	tensor = shoji.io.get_tensor(tr, wsm, name)
	const = tensor.python_dtype()(const)  # Cast the const to string, float or int
	index = wsm._subdir[Compartment.TensorIndex][name]
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


def const_compare_non_transactional(wsm: "shoji.Workspace", name: str, operator: str, const: Tuple[int, str, float]) -> np.ndarray:
	"""
	Compare a tensor to a constant value, and return all indices that match
	"""
	# Code for range, equality and inequality filters
	tensor = wsm._get_tensor(name)
	const = tensor.python_dtype()(const)  # Cast the const to string, float or int
	index = wsm._subdir[Compartment.TensorIndex][name]
	eq_range = index[const].range()
	all_range = index.range()
	start, stop = all_range.start, all_range.stop
	tr = wsm._db.transaction  # This will typically (except inside a Transaction scope) be set to the database, so that each time it's used it will create a separeate transaction
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

	tr = wsm._db.create_transaction()
	n = 100_000
	result = []
	next_start = b''
	while start < stop:
		try:
			temp = []
			for k, _ in tr.get_range(start, stop, limit=n):
				temp.append(index.unpack(k)[1])
			next_start = tr.get_key(fdb.KeySelector.first_greater_than(k)).value
			result += temp
		except fdb.impl.FDBError as e:
			if e.code in (1004, 1007, 1031, 2101) and n > 1:  # Too many bytes or too long time, so try again with less
				n = max(1, n // 2)
				tr = wsm._db.create_transaction()
				continue
			else:
				raise e
		start = next_start
	return np.array(result, dtype="int64")


def get_filtered_indices(wsm: "shoji.Workspace", tensor: "shoji.Tensor", filters: List["shoji.Filter"], axis: int, n_rows: int) -> np.ndarray:
	indices = None
	for f in filters:
		if isinstance(tensor.dims[axis], str) and tensor.dims[axis] == f.dim:
			indices = np.sort(f.get_rows(wsm))
		elif isinstance(f, (shoji.TensorBoolFilter, shoji.TensorIndicesFilter, shoji.TensorSliceFilter)) and f.tensor.name == tensor.name and f.axis == axis:
			indices = np.sort(f.get_rows(wsm, n_rows))
	if indices is None:
		indices = np.arange(n_rows)
	return indices


def read_filtered(wsm: "shoji.Workspace", name: str, filters: List["shoji.Filter"]) -> Union[np.ndarray, List[np.ndarray]]:
	tensor = wsm._get_tensor(name)
	subspace = wsm._subdir
	if tensor.jagged:
		rows = get_filtered_indices(wsm, tensor, filters, 0, tensor.shape[0])
		result = []
		for row in rows:
			row_shape = fdb.tuple.unpack(wsm._db.transaction[subspace.pack((Compartment.TensorRowShapes, name, int(row)))])
			indices = [np.array([row])]
			for axis in range(1, tensor.rank):
				indices.append(get_filtered_indices(wsm, tensor, filters, axis, row_shape[axis - 1]))
			result.append(shoji.io.read_at_indices(wsm, name, indices, tensor.chunks, False))
		return result
	else:
		indices = [get_filtered_indices(wsm, tensor, filters, i, tensor.shape[i]) for i in range(tensor.rank)]
		return shoji.io.read_at_indices(wsm, name, indices, tensor.chunks, False)


@fdb.transactional
def write_filtered(tr: fdb.impl.Transaction, wsm: "shoji.Workspace", name: str, vals: Union[np.ndarray, List[np.ndarray]], filters: List["shoji.Filter"]) -> None:
	tensor: shoji.Tensor = wsm._get_tensor(name)
	subspace = wsm._subdir
	assert isinstance(vals, (np.ndarray, list, tuple)), f"Value assigned to '{name}' is not a numpy array or a list or tuple of numpy arrays"

	if tensor.jagged:
		rows = get_filtered_indices(wsm, tensor, filters, 0, tensor.shape[0])
		for row in rows:
			row_shape = fdb.tuple.unpack(wsm._db.transaction[subspace.pack((Compartment.TensorRowShapes, name, int(row)))])
			indices = [np.array([row])]
			for axis in range(1, tensor.rank):
				indices.append(get_filtered_indices(wsm, tensor, filters, axis, row_shape[axis - 1]))
			shoji.io.write_at_indices(tr, wsm, (Compartment.TensorValues, name), indices, tensor.chunks, vals[row])
	else:
		indices = [get_filtered_indices(wsm, tensor, filters, i, tensor.shape[i]) for i in range(tensor.rank)]
		shoji.io.write_at_indices(tr, wsm, (Compartment.TensorValues, name), indices, tensor.chunks, vals)
