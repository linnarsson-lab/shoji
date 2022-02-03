"""
Connecting to a Shoji database cluster.

To connect to a shoji database, typically you just need to do this:

```python
db = shoji.connect()
```

This returns an object `db` representing the root `shoji.workspace.Workspace`.

"""
import fdb
import shoji


def connect(cluster_file=None, event_model=None) -> shoji.Workspace:
	"""
	Connect to a Shoji (i.e. [FoundationDB](https://www.foundationdb.org)) database cluster.

	Args:
		cluster_file: The FoundationDB cluster file to use (or None to use the default)
		event_model: Additional FoundationDB parameters

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
	db.options.set_transaction_retry_limit(1)  # Retry every transaction only once if it doesn't go through
	subdir = fdb.directory.create_or_open(db, ("shoji",))
	return shoji.Workspace._attach(db, subdir, ())
