
## Ideas

Use a path tuple to open all kinds of storage (+optional storage type)
    cluster_file, path/to/workspace  # Shoji (cluster_file is optional)
    /path/to/file, /path/to/workspace  # hdf5
    /path/to/file, raw   # or not raw; h5ad
    /path/to/file  # loom

shoji.connect(file_path, workspace_path, kind=None)


To support strong typing, distinguish between accessing workspace and accessing tensors and dimensions:

db = shoji.connect()  # db is now always a Workspace
ws = db / "builds" / "sten" / "gbm"  # access sub-workspaces
ws = db / "builds/sten/gbm"  # same result
ws.Tensor  # Only tensors accessed like this
ws.dims_("cells")  # The dimensions can be accessed like this

