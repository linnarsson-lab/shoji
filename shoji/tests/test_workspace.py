import shoji
import pytest


def test_create_workspace():
	db = shoji.connect()
	if "test" in db:
		del db.test
	if "test2" in db:
		del db.test2
	db.test = shoji.Workspace()
	with pytest.raises(ValueError):
		db.test2 = shoji.WorkspaceManager(db, db._subdir, ("test2",))
	del db.test
	del db.test2


def test_create_workspace_collision():
	db = shoji.connect()
	if "test" in db:
		del db.test
	db.test = shoji.Workspace()
	with pytest.raises(AttributeError):
		db.test = shoji.Workspace()
	db.test.dim = shoji.Dimension(shape=10)
	with pytest.raises(AttributeError):
		db.test.dim = shoji.Workspace()
	del db.test


def test_move_workspace():
	db = shoji.connect()
	if "test" in db:
		del db.test
	if "test2" in db:
		del db.test2
	db.test = shoji.Workspace()
	db.test.dim = shoji.Dimension(shape=10)
	db.test._move_to(("test2",))
	assert "test" not in db
	assert "test2" in db
	assert "dim" in db.test2
	assert isinstance(db.test2.dim, shoji.Dimension)
	del db.test2


def test_move_workspace_collision():
	db = shoji.connect()
	if "test" in db:
		del db.test
	if "test2" in db:
		del db.test2
	db.test = shoji.Workspace()
	db.test2 = shoji.Workspace()
	with pytest.raises(ValueError):
		db.test._move_to(("test2",))
	del db.test
	del db.test2
