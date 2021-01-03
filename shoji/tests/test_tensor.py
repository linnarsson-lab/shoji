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
	del db.test.Test
	del db.test


def test_append():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(shape=1000)
	db.test.dim2 = shoji.Dimension(shape=None)
	db.test.Test1 = shoji.Tensor("float32", ("dim1", "dim2"), inits=np.arange(300_000, dtype="float32").reshape(1000, 300))
	db.test.Test2 = shoji.Tensor("int16", ("dim2", "dim1"), inits=np.arange(300_000, dtype="int16").reshape(300, 1000))
	with pytest.raises(AssertionError):
		db.test.dim1.append({
			"Test1": np.zeros((100, 300)),
			"Test2": np.zeros((100, 300))
		})
	with pytest.raises(ValueError):
		db.test.dim2.append({
			"Test1": np.zeros((100, 300)),
			"Test2": np.zeros((100, 300))
		})
	db.test.dim2.append({
		"Test1": np.zeros((1000, 200)),
		"Test2": np.zeros((200, 1000))
	})
	del db.test.Test
	del db.test


def test_jagged():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(shape=10)
	inits = [np.arange(x * 4, dtype="float32").reshape(x, 4) for x in range(1, 11)]
	db.test.Test = shoji.Tensor("float32", ("dim1", None, 4), inits=inits, jagged=True)
	assert isinstance(db.test.Test[:], list)
	assert all(np.all(a == b) for a,b in zip(inits, db.test.Test[:]))
	del db.test.Test
	del db.test


def test_filter():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(shape=10)
	db.test.Test = shoji.Tensor("float32", ("dim1",), inits=np.arange(10, dtype="float32"))
	assert np.all(db.test.Test[db.test.Test > 2] == np.arange(3, 10, dtype="float32"))
	del db.test.Test
	del db.test
