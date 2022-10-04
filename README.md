

# Getting started with Shoji

Shoji requires Python 3.7+ (we recommend Anaconda)

First, in your terminal, install the shoji Python package:

```
$ git clone https://github.com/linnarsson-lab/shoji.git
$ pip install -e shoji
```

Check that you can now connect to the database:

```python
import shoji
db = shoji.connect()
db
```

Typing db alone at the last line above should return a representation of the contents of the database (which might be empty at this point).

# Setting up a Shoji database on macOS

<aside>
❗ This page is for setting up a new Shoji database. If you're just going to use an existing Shoji database, e.g. on monod, there is no need to set up FoundationDB or Shoji locally.

</aside>

### Install FoundationDB

Shoji is based on FoundationDB. You can easily set up a local Shoji database by following these instructions:

1. [Download FoundationDB](https://apple.github.io/foundationdb/downloads.html)
2. Double-click on FoundationDB-6.2.27.pkg and follow the instructions
    
    <aside>
    💡 If you get a security error (“FoundationDB-6.2.27.pkg” cannot be opened because it is from an unidentified developer) go to Settings → Security & Privacy → General and click on Open Anyway and then on Open (in the dialog).
    
    </aside>
    
3. In your Terminal, type `fdbcli` and then `status` to confirm that the database is up and running
4. Still in `fdbcli`, type `configure ssd` to change the storage engine to ssd-2 
5. After a few minutes, `status` should again show as `Healthy`

### Install the Python libraries

```bash
pip install -U foundationdb
git clone https://github.com/linnarsson-lab/shoji.git
cd shoji
pip install -e .
```

### Verify the installation

```python
>>> import shoji
>>> db = shoji.connect()
>>> db
(root) (shoji.Workspace)
```

### Where is everything?

Documentation: `shoji/html/shoji/index.html`

FDB cluster file: `/usr/local/etc/foundationdb/fdb.cluster`

FDB config: `/usr/local/etc/foundationdb/foundationdb.conf`

Data: `/usr/local/foundationdb/`
