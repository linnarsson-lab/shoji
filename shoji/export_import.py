"""
Exporting and importing workspaces
"""
import shoji
import h5py
import numpy as np
import pickle
import blosc

def _import(f: str, ws: shoji.WorkspaceManager, recursive: bool = False, group_name: str = "/"):
	"""
	Import an HDF5 file to a Shoji workspace

	Args:
		f			The file name (full path)
		ws			The workspace to export
		recursive	If true, sub-workspaces will be imported from HDF5 sub-groups
		group_name	The name of the HDF5 group where the workspace is stored
	
	Remarks:
		If the workspace exists, it will be overwritten
	"""
	with h5py.File(f, mode="r") as h5:
		for subgroup in h5[group_name]:
			if isinstance(subgroup, h5py.Group):
				name = subgroup.name.split("/")[-1]
				if name.startswith("Dimension$"):
					shape, length = 
					ws[name.split("$")[-1]] = shoji.Dimension()

	# To be continued...


def export(ws: shoji.WorkspaceManager, f: str, recursive: bool = False, group_name: str = "/"):
	"""
	Export a Shoji workspace to an HDF5 file

	Args:
		ws			The workspace to export
		f			The file name (full path)
		recursive	If true, sub-workspaces will be exported to HDF5 sub-groups
		group_name	The name of the HDF5 group where the workspace should be stored

	Remarks:
		If the file exists, it will be overwritten.
	"""
	with h5py.File(f, mode="w") as h5:
		group = h5.require_group(group_name)

		for dname in ws._dimensions():
			dim = ws._get_dimension(dname)
			group.attrs["Dimension$" + dname] = (dim.shape if dim.shape is not None else -1, dim.length)

		for tname in ws._tensors():
			tensor = ws._get_tensor(tname)
			print(tname)
			group.create_dataset(
				"Tensor$" + tname,
				data = np.frombuffer(pickle.dumps(tensor), dtype="uint8")
			)
			data = np.frombuffer(blosc.pack_array(np.array(tensor[...])), dtype="uint8")
			group.create_dataset(
				tname,
				data=data
			)

		if recursive:
			for wsname in ws._workspaces():
				export(ws._get_workspace(wsname), f, True)
