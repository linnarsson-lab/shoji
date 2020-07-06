from setuptools import find_packages, setup

# First update the version in loompy/_version.py, then:

# cd ~/code/loompy  (the directory where loompy resides)
# rm -r dist   (otherwise twine will upload the oldest build!)
# python setup.py sdist
# twine upload dist/*

# NOTE: Don't forget to update the release version at loompy.github.io (index.html)!

# pylint: disable=exec-used
__version__ = '0.0.0'
exec(open('shoji/_version.py').read())

setup(
	name="shoji",
	version=__version__,
	packages=find_packages(),
	python_requires='>=3.6',
	install_requires=[
	],
	# metadata for upload to PyPI
	author="Linnarsson Lab",
	author_email="sten.linnarsson@ki.se",
	description="Python API for shoji, a tensor database",
	license="BSD",
	keywords="shoji tensor omics transcriptomics bioinformatics microscopy",
	url="https://github.com/linnarsson-lab/shoji",
	download_url=f"https://github.com/linnarsson-lab/shoji/archive/{__version__}.tar.gz",
)
