import shoji
import pytest
import numpy as np


def test_create_tensor():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.Test = shoji.Tensor("uint32", (None,), inits=np.arange(100, dtype="uint32"))
	assert np.all(db.test.Test[:] == np.arange(100, dtype="uint32"))
	del db.test.Test
	del db.test

def test_modify_tensor():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.Test = shoji.Tensor("uint32", (None,), inits=np.arange(100, dtype="uint32"))
	db.test.Test[:10] = np.full(10, 91)
	assert np.all(db.test.Test[:10] == np.full(10, 91, dtype="uint32"))
	assert np.all(db.test.Test[10:] == np.arange(10, 100, dtype="uint32"))
	del db.test.Test
	del db.test



def test_read_at_indices():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(shape=1000)
	db.test.dim2 = shoji.Dimension(shape=300)
	db.test.Test = shoji.Tensor("float32", ("dim1", "dim2"), inits=np.arange(300_000, dtype="float32").reshape(1000,300))
	db.test.Test[0:1000:100, 0:300:100] = np.full((100, 3), 91, dtype="float32")
	#assert np.all(db.test.Test[0:1000:100, 0:3000:100] == np.full((100, 30), 91, dtype="float32"))
	del db.test.Test
	del db.test

def test_create_jagged():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(shape=10)
	inits = [np.arange(x * 4, dtype="float32").reshape(x, 4) for x in range(10)]
	db.test.Test = shoji.Tensor("float32", ("dim1", None, 4), jagged=True, inits=inits)
	assert isinstance(db.test.Test[:], list)
	assert all(a == b for a,b in zip(inits, db.test.Test[:]))
	del db.test.Test
	del db.test
