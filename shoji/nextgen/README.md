
## Ideas

Use a path to open all kinds of storage (+optional storage type)
    cluster_file  # Shoji (cluster_file is optional)
    /path/to/file  # hdf5
    /path/to/file  # h5ad
    /path/to/file  # loom

shoji.connect(file_path, kind=None)


To support strong typing, distinguish between accessing workspace and accessing tensors and dimensions:

db = shoji.connect()  # db is now always a Workspace
ws = db.builds.sten.gbm  # access sub-workspaces, type is always Workspace
ws.d.cells  # The dimensions can be accessed like this
ws.d.genes
ws.t.Expression  # access tensors of this workspace, type is Tensors


## Enums

ws.t.IsValid = shoji.Tensor("cells", dtype="enum", values=["true", "false"], indexed=False)



## ShojiStorage contract with user

- Database structure is transactionally guaranteed
- Increasing the length of a dimension does not make any guarantee as to the newly exposed values
- Deleting a tensor reclaims all space occupied by that tensor (including any due to truncating dimensions)


## ShojiStorage physical layout

# Directories correspond to workspaces
("shoji_nextgen")
("shoji_nextgen", "path", "to", "workspace")

# Keys in directory
("shoji_nextgen", "path", "to", "workspace", Compartment.Dimensions, "cells") -> <length>

