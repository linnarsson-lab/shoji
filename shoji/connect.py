"""
Connecting to a Shoji database cluster.
"""
import fdb
import shoji


def connect(cluster_file=None, event_model=None) -> shoji.WorkspaceManager:
	"""
	Connect to a Shoji (i.e. [FoundationDB](https://www.foundationdb.org)) database cluster.

	Args:
		cluster_file: The FoundationDB cluster file to use (or None to use the default)
		event_model: Addtional FoundationDB parameters

	Remarks:
		The cluster file should normally not be explicitly provided as an argument to this function.
		Instead, it will be located automatically by searching these locations:

		1. The value of the FDB_CLUSTER_FILE environment variable, if it has been set;
		2. An fdb.cluster file in the current working directory, if one is present;
		3. The default file at its system-dependent location.

		See https://apple.github.io/foundationdb/administration.html for more information
	"""
	db = fdb.open(cluster_file=cluster_file, event_model=event_model)
	db.transaction = db  # default to using the Database object for transactions
	subspace = fdb.directory.create_or_open(db, ("shoji",))
	return shoji.WorkspaceManager(db, subspace, ())
