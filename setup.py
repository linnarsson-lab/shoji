from setuptools import find_packages, setup

# First update the version in loompy/_version.py, then:

# cd ~/code/loompy  (the directory where loompy resides)
# rm -r dist   (otherwise twine will upload the oldest build!)
# python setup.py sdist
# twine upload dist/*

# NOTE: Don't forget to update the release version at loompy.github.io (index.html)!

# pylint: disable=exec-used
__version__ = '0.0.0'
exec(open('doompy/_version.py').read())

setup(
	name="doompy",
	version=__version__,
	packages=find_packages(),
	python_requires='>=3.6',
	install_requires=[

	],
	entry_points='''
		[console_scripts]
		loompy=loompy.commands:cli
	''',
	# metadata for upload to PyPI
	author="Linnarsson Lab",
	author_email="sten.linnarsson@ki.se",
	description="Python API for doom, a database for omics",
	license="BSD",
	keywords="doom omics transcriptomics bioinformatics",
	url="https://github.com/linnarsson-lab/doompy",
	download_url=f"https://github.com/linnarsson-lab/doompy/archive/{__version__}.tar.gz",
)
