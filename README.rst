Simple Swath
============

This module is designed to extract a swath profile from a raster using a line shapefile.
Finally, it has been completely recoded, but this is a simplification of the **pyswath** module (https://github.com/robertxa/pyswath). The main difference is that **pyswath** relies on profiles defined by points, **simple_swath** relies on a shapefile, and not only on points definition.

Install
-------

To install it :
::
	$ pip install simple_swath

Dependencies
------------

- numpy
- gdal
- shapely
- rasterstats
- alive_progress
- matplotlib
- matplotlib_scalebar
- csv
- pickle

Usage
-----

Inside a (i)python environnement:

To import the module:
	>>> from simple_swath import swath
	
To run the swath extraction:
	>>> swath(raster_path = 'Rasterpathandname', shapefile_path = 'shapepathandname', outfile = 'prefixname', increment_value = 10, window_size = 100, bins = 40, xshift = Non, nodata_value = -9999, meanmedian = 'mean', minmax = True, frequencyplot = True, TEMP = False, ylim = None, subplots = True)

Options/inputs
--------------

To use options or inputs, you need to set them as:
	>>> swath(option_name = option_value, [...])
	
Options/inputs are (option_names):

#. **raster_path (str)**              : Path of the Raster to use. The raster needs to be projected; the units of the grid should be meters (m)
  
#. **shapefile_path (str)**           : Path of the shapefile to use to extract the profile. The shapefile should be in the same projection than the input raster. The shapefile should have a line or polyline geometry (2D or 3D), and can have several features.
	
#. **outfile (str)**                  : Prefix to add to the outputs' names.
  
#. **spl_incrt (float)**              : Increment along the profile (in m).
	
#. **spl_wndw (float)**               : Width of the box to build (in m).
  
#. **bins (integer, optional)**       : Number of categories to compute the histogram.

#. **xshift (real or list of reals, optional)**: if you need to shift a profile along the disance axis
                                                for 1 profile, provide a real (or same shifting applied to all the profiles)
                                                for n profiles in the shp, provide a list of n reals (len(xshit) == n)
                                                Default to None. 

#. **nodata_value (integer)**         : set the input raster's nodata value; default to None
  
#. **minmax (bool, optional)**        : True to plot the min/max. Defaults to False.
  
#. **frequencyplot (bool, optional)** : True to plot the frequency. Defaults to False.
  
#. **TEMP (bool, optional)**          : True to keep the temporary shapefiles, False to delete the folder TEMP/. Defaults to False.
  
#. **ylim (tuple, optional)**        : Tuple of y limits for the graph profile. Defaults to None
  
#. **subplots (bool, optional)**     : True to plot the raster and the swath on the same plot, False to plot the raste and the swath as two separate plots. Default =  False

#. **map_plot (dict, optional)**     : Dictionnary to tune the plot of the raster. It should be as 
                                       >>>map_plot = {'cmap' : "terrain", 'alphaM' : 0.7, 'alphaH' : 1, 'hshd'     : True, 'hshd_az'  : 315, 'hshd_alt' : 45}
                                       where **cmap** is the cmap used to plot the DEM, **alphaM** and **alphaH** the transparency of the DEM and hillshade, **hshd** is True if you want an hillshade above the DEM, and hshd_az and hshd_alt the Azimuth and altitude used to compute the hillshade
                                       Default to None ; in that case, the values given in the ex. are used


.. Caution::
  Be carefull, the raster should be projected and the grid should be in meters (m)...


Help files
----------

To get help in your (i)python environnement:
	>>> help(swath)
			
Outputs
-------

The module outputs pdfs of graphs, csv files, and pickle binary files.

1. **pdfs** are the plot of the raster with the line use for the swath, and the swath profile

2. **csv files** are a record of the data use to plot the swath (max, min, mean, median, 1s). There is one file for each profile/feature. If a profile is shifted along distances, this is writen as a comment in the first line of the csv file

3. **pickle binary files** record the data used plot the frequency plot. There are two files. The file "rater_freqs.pickle" contains the frequency data, and the file "bin_edges.pickle" contains the bins extend of the frequency.
	

How to cite
-----------

.. image:: https://zenodo.org/badge/751342655.svg
  :target: https://zenodo.org/doi/10.5281/zenodo.10606462

Licence
-------

**Copyright (c)** 2024 Xavier Robert <xavier.robert@ird.fr> and Benjamin Lehmann <lehmann.benj@gmail.com>
SPDX-License-Identifier: **GPL-3.0-or-later**
