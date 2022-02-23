

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

