import shoji
import pytest
import numpy as np


def test_create_dimension():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	db.test.dim1 = shoji.Dimension(None)
	db.test.dim2 = shoji.Dimension(100)
	db.test.Test = shoji.Tensor("uint32", ("dim1", "dim2"), inits=np.zeros((1000, 100), dtype="uint32"))
	db.test.dim1.append({
		"Test": np.zeros((1000, 100), dtype="uint32")
	})
	assert db.test.dim1.length == 2000
	del db.test
