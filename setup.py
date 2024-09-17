######!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2024 Xavier Robert <xavier.robert@ird.fr> and Benjamin Lehmann <lehmann.benj@gmail.com>
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib.util
from setuptools import setup


spec = importlib.util.spec_from_file_location(
    "simple_swath._version",
    "simple_swath/_version.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
VERSION = module.__version__

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='simple_swath',
	version=VERSION,
	description='Module that provides tools to extract swath profile using a shapefile',
	long_descritpion=open('README.rst').read(),
	url='https://github.com/robertxa/Simple_Swath',
	#dowload_url='https://github.com/robertxa/simple_swath/archive/master.zip',
	author='Xavier Robert',
	author_email='xavier.robert@ird.fr',
	license='GPL-V3.0',
    entry_points={
        "console_scripts": [
            "swath=swath.command_line:main",
        ],
    },
	packages=['simple_swath'],
	install_requires=[
        'numpy',
		'gdal',
        'shapely',
		'rasterstats',
		'matplotlib',
		'matplotlib_scalebar',
		'alive_progress',
		'csv',
		'pickle'
	],
	#classifiers=[
	#	"Programming language :: Python",
	#	"Operating System :: OS Independent",
	#	"Topic :: Caving",
	#],
	include_package_data=True,
	zip_safe=False)
      