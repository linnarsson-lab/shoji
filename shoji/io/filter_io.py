from typing import Tuple
import shoji
import shoji.io
import fdb
import numpy as np


@fdb.transactional
def const_compare(tr, wsm: shoji.WorkspaceManager, name: str, operator: str, const: Tuple[int, str, float]) -> np.ndarray:
	"""
	Compare a tensor to a constant value, and return all indices that match
	"""
	# Code for range, equality and inequality filters
	tensor = shoji.io.get_tensor(tr, wsm, name)
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