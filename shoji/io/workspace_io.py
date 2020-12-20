from typing import List, Union, Optional
import fdb
import shoji
import pickle


"""
# General functions for managing workspaces and their content
"""

@fdb.transactional
def get_entity(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str) -> Optional[Union[shoji.Dimension, shoji.Tensor, "shoji.WorkspaceManager"]]:
	t = get_tensor(tr, wsm, name)
	if t is not None:
		return t
	d = get_dimension(tr, wsm, name)
	if d is not None:
		return d
	s = get_workspace(tr, wsm, name)
	if s is not None:
		return s
	return None


@fdb.transactional
def get_workspace(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str) -> Optional["shoji.WorkspaceManager"]:
	subdir = wsm._subdir
	if subdir.exists(tr, name):
		child = subdir.open(tr, name)
		wsm = shoji.WorkspaceManager(wsm._db, child, wsm._path + (name,))
		wsm._name = name
		return wsm
	return None


@fdb.transactional
def get_dimension(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str) -> Optional[shoji.Dimension]:
	subdir = wsm._subdir
	val = tr[subdir["dimensions"][name]]
	if val.present():
		dim = pickle.loads(val.value)
		dim.name = name
		dim.wsm = wsm
		return dim
	return None


@fdb.transactional
def get_tensor(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str, include_initializing: bool = False) -> Optional[shoji.Tensor]:
	subdir = wsm._subdir
	val = tr[subdir.pack(("tensors", name))]
	if val.present():
		tensor = pickle.loads(val.value)
		tensor.name = name
		tensor.wsm = wsm
		return tensor
	return None


@fdb.transactional
def list_workspaces(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager") -> List["shoji.WorkspaceManager"]:
	return [get_workspace(tr, wsm, name) for name in wsm._subdir.list(tr)]


@fdb.transactional
def list_dimensions(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager") -> List[shoji.Dimension]:
	result = []
	for kv in tr[wsm._subdir["dimensions"].range()]:
		dim = pickle.loads(kv.value)
		dim.name = wsm._subdir["dimensions"].unpack(kv.key)[0]
		dim.wsm = wsm
		result.append(dim)
	return result


@fdb.transactional
def list_tensors(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", include_initializing: bool = False) -> List[shoji.Tensor]:
	result = []
	for kv in tr[wsm._subdir["tensors"].range()]:
		tensor = pickle.loads(kv.value)
		if tensor.initializing and not include_initializing:
			continue
		tensor.name = wsm._subdir["tensors"].unpack(kv.key)[0]
		tensor.wsm = wsm
		result.append(tensor)
	return result


@fdb.transactional
def delete_entity(tr: fdb.impl.Transaction, wsm: "shoji.WorkspaceManager", name: str) -> None:
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
def create_dimension(tr, wsm: "shoji.WorkspaceManager", name: str, dim: shoji.Dimension):
	"""
	Create a dimension in the workspace, overwriting any existing dimension of the same name
	"""
	subdir = wsm._subdir
	# Check that name doesn't already exist
	existing = get_entity(tr, wsm, name)
	if existing is not None:
		if not isinstance(existing, shoji.Dimension):
			raise AttributeError(f"Cannot overwrite {type(existing)} '{existing}' with a new shoji.Dimension (you must delete it first)")
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
	tr[subdir["dimensions"][name]] = pickle.dumps(dim)
