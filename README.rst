Simple Swath
============

This module is designed to extract a swath profile from a raster using a line shapefile. 

Install
-------

To install it :
	pip install simple_swath

Usage
-----

Inside a (i)python environnement:

To import the module:
	>>> from simple_swath import swath
	
To 
	>>> swath(raster_file = 'Rastername', shapefile_file = 'shapename', outfile = 'prefixname', increment_value = 10, window_size = 100, bins = 40, meanmedian = 'mean', minmax = True, frequencyplot = True, TEMP = False, ylim = None, subplots = True)

Options/inputs
--------------

To use options or inputs, you need to set them as
	
	swath(option_name = option_value, [...])
	
Options/inputs are (option_names):

	1. raster_path (str)              :
  Path of the Raster to use.
  The raster needs to be projected; units should be meters
  2. shapefile_path (str)           : Path of the shapefile to use to extract the profile.
										The shapefile should be in the same projection than the input raster
	3. outfile (str)                  : Prefix to add to the outputs' names.
  4. spl_incrt (float)              : Increment along the profile (in m).
	5. spl_wndw (float)               : Width of the box to build (in m).
  6. bins (integer, optional)       : Number of categories to compute the histogram.
  7. minmax (bool, optional)        : True to plot the min/max. 
                                        Defaults to False.
  8. frequencyplot (bool, optional) : True to plot the frequency. 
                                        Defaults to False.
  9. TEMP (bool, optional)          : True to keep the temporary shapefiles,
                                        False to delete the folder TEMP/.
                                        Defaults to False.
  10. ylim (tuple, optional)        : Tuple of y limits for the graph profile;
                                        Defaults to None
  11. subplots (bool, optional)     : True to plot the raster and the swath on the same plot
                                        False to plot the raste and the swath as two separate plots
                                        default =  False

Help files
----------

To get help in your (i)python environnement:
	>>> help(swath)
			
Outputs
-------

The module outputs pdfs of graphs, csv files, and pickle binary files.

	1. pdfs are the plot of the raster with the line use for the swath, and the swath profile
	2. csv files are a record of the data use to plot the swath (max, min, mean, median, 1s)
	3. pickle binary files record the data used plot the frequency plot. There are two files. The file "rater_freqs.pickle" contains the frequency data, and the file "bin_edges.pickle" contains the bins extend of the frequency.
	

How to cite
-----------

.. image:: https://zenodo.org/badge/159739189.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.10020982


Licence
-------

Copyright (c) 2024 Xavier Robert <xavier.robert@ird.fr> and Benjamin Lehmann <lehmann.benj@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later
