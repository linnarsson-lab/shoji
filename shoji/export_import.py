"""
Exporting and importing workspaces
"""
import shoji
import h5py
import numpy as np


def import_(f: str, ws: shoji.WorkspaceManager):
	...

def export(ws: shoji.WorkspaceManager, f: str, recursive: bool = False, group_name: str = "/"):
	"""
	Export a Shoji workspace to an HDF5 file

	Args:
		ws			The workspace to export
		f			The file name (full path)
		recursive	If true, sub-workspaces will be exported to HDF5 sub-groups
		group_name	The name of the HDF5 group where the workspace should be stored

	Remarks:
		If the file does not exist, it will be created
	"""
	h5 = h5py.File(f)
	group = h5.require_group(group_name)

	for dname in ws._dimensions():
		dim = ws._get_dimension(dname)
		group.attrs["Dimension$" + dname] = (dim.shape if dim.shape is not None else -1, dim.length)

	for tname in ws._tensors():
		tensor = ws._get_tensor(tname)
		group.attrs["Tensor$" + tname] = np.array((tensor.dtype, tensor.rank, 1 if tensor.jagged else 0) + tensor.dims + tensor.shape, dtype=object)
		group.create_dataset(
			tname,
			tensor.shape,
			tensor.dtype # TODO: get this right
		)
		# Now read/write the dataset
	if recursive:
		for wsname in ws._workspaces():
			export(ws._get_workspace(wsname), f, True)
	
	h5.close()
