#!/usr/bin/env python
# coding: utf-8

##################################################
# Copyrights:
#   (c) 2024 - Benjamin Lehmann, ISTerre, UGA <lehmann.benj@gmail.com>
#   (c) 2024 - Xavier Robert, ISterre, IRD <xavier.robert@ird.fr>
#
# SPDX-License-Identifier: GPL-3.0-or-later
##################################################

# Import modules
import os, math
from shutil import rmtree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from osgeo import gdal, ogr
from shapely.geometry import LineString, Polygon
from rasterstats import zonal_stats
import csv
import pickle
from alive_progress import alive_bar              # https://github.com/rsalmei/alive-progress

#########################################################################################
def buildbox(raster_path, raster_dataset, line, distance, k, spl_wndw, spl_incrt):

    """
	Create a polygon shapefile perpendicular at a distance "distance" along the line "line" 
        with a lenght spl_incrt and a width spl_wndw.
    And then extract and return data and zonal statistics from raster_dataset inside the polygon built.
    It also records the shapefile of the box computed.

    This function is an adaptation of the makeshape function from pyswath (https://zenodo.org/badge/latestdoi/81983899)
	
	INPUTS:
	   raster_path (str)            : path to the raster file
	   raster_dataset (gdal raster) : raster from which to extract data
       line (Shapely LineString)    : Shapely line shapefile object along which distances and positions are defined
       distance (float)             : distance from the begining of the line, where to build the box (in m)
       spl_wndw (float)             : width of the box to build (in m)
	   spl_incrt (float)            : length of the box to build (in m)
	
	OUTPUTS:
	   line.shp (ESRI shapefile)         : shapefile int Temp_XX/ folder
       values_around_point (numpy array) : extraction of the values of the raster that are inside the built polygon
       stats (dictionnary)               : zonal statistics in values_around_point_ array

	
	USAGE:
	  values_around_point, stats = buildbox(raster_path, raster_dataset, line, distance, k, spl_wndw, spl_incrt)
	
	"""

    # find the points's coordinates of the segment
    if distance == 0:   # For the first point of the line
        point1 = line.interpolate(distance)
        point2 = line.interpolate(distance + spl_incrt/2)
    else:   # if the point is not the first point of the line
        point1 = line.interpolate(distance - spl_incrt/2)
        if distance == int(line.length):    # For the last point of the line
            point2 = line.interpolate(distance)
        else:   # if the point is not the last point of the line
            point2 = line.interpolate(distance + spl_incrt/2)

	# beta is the angle between the parallel/horizontal and the transect
    beta = math.atan((point2.y - point1.y) / (point2.x - point1.x))
	
	# Recompute the coordinates of the corner :
    xxx = spl_wndw / 2 * np.sin(beta)
    yyy = spl_wndw / 2 * np.cos(beta)
    
    # create the corner points of the shapefile
    aa = [point1.x - xxx, point1.y + yyy]
    dd = [point1.x + xxx, point1.y - yyy]
    cc = [point2.x + xxx, point2.y - yyy]
    bb = [point2.x - xxx, point2.y + yyy]
	
	# from http://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
	# Create the polygon to be used for the shapefile
    poly = Polygon([(aa[0],aa[1]), (bb[0],bb[1]), (cc[0],cc[1]), (dd[0],dd[1]), (aa[0],aa[1])])
	# Now convert it to a shapefile with OGR    
    shp = 'TEMP_' + str(k) + '/shp_' + str(distance) + '.shp'
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(shp)
	# Set the Reference
    layer = ds.CreateLayer('zoup', raster_dataset.GetSpatialRef(), ogr.wkbPolygon)
	# Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
	# Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)
	# Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(poly.wkb)
    feat.SetGeometry(geom)
    layer.CreateFeature(feat)
    feat = geom = None  # destroy these
	# Save and close everything
    ds = layer = feat = geom = None

    # Extract values included in the box
    stats = zonal_stats(vectors = shp,
			            raster = raster_path, 
			            #nodata_value = -9999,
                        nodata_value = np.nan,
                        #nodata_value = raster_dataset.GetRasterBand(1).GetNoDataValue(),
			            raster_out = True,
		                stats = ['min', 'max', 'median', 'mean', 'count'])
    
    values_around_point = [f['mini_raster_array'] for f in stats][0].compressed()
    # set -9999 values to NaN
    values_around_point[values_around_point == -9999] = np.nan

    return values_around_point, stats


#########################################################################################
def swath(raster_path, shapefile_path, outfile, 
          spl_incrt, spl_wndw, bins = None, 
          meanmedian = 'median', minmax = False, frequencyplot = False, TEMP = False,
          ylim = None, subplots = False):
    """
    
    Args:
        raster_path (str)              : Path of the Raster to use.
        shapefile_path (str)           : Path of the shapefile to use to extract the profile.
        outfile (str)                  : Prefix to add to the outputs' names.
        spl_incrt (float)              : Increment along the profile (in m).
        spl_wndw (float)               : Width of the box to build (in m).
        bins (integer, optional)       : Number of categories to compute the histogram.
        minmax (bool, optional)        : True to plot the min/max. 
                                         Defaults to False.
        frequencyplot (bool, optional) : True to plot the frequency. 
                                         Defaults to False.
        TEMP (bool, optional)          : True to keep the temporary shapefiles,
                                         False to delete the folder TEMP/.
                                         Defaults to False.
        ylim (tuple)                   : Tuple of y limits for the graph profile;
                                         Defaults to None
        subplots (bool, optional)      : True to plot the raster and the swath on the same plot
                                         False to plot the raste and the swath as two separate plots
                                         default =  False

    Raises:
        ValueError : if the specified path to the raster or the shapefile does not exists
    """

    # Print header
    print('#################################################################')
    print('                   Build swath profile                           ')
    print('              (c) 2024 B. Lehmann & X. Robert                     ')
    print('\n  - Raster: %s' %(raster_path))
    print('   - Shapefile: %s' %(shapefile_path))
    print('   - width: %s m' %(spl_wndw))
    print('   - increment: %s m\n' %(spl_incrt))
    print('################################################################\n')

    # Remove gdal warnings
    gdal.UseExceptions()

    print('\x1b[32;1m- Reading data...\x1b[0m')
    # Open the raster file
    raster_dataset = gdal.Open(raster_path)
    if raster_dataset is None:
        raise ValueError(f"Error: Unable to open the raster file at {raster_path}.")

    # Read raster data
    raster_array = raster_dataset.ReadAsArray()
    # Get the transformation parameters
    transform = raster_dataset.GetGeoTransform()
    px_size = transform[1]

    # Open the shapefile
    shapefile_dataset = ogr.Open(shapefile_path)
    if shapefile_dataset is None:
        raise ValueError(f"Error: Unable to open the shapefile at {shapefile_path}.")
    # Get the first layer of the shapefile
    layer = shapefile_dataset.GetLayerByIndex(0)

    if not os.path.isdir('Outs'):
        # Check if the output folder "Out/" exists
        # And if not, make it
        os.mkdir('Outs')

    # Initiate variables
    if minmax:
        raster_mins = []
        raster_maxs = []
    if meanmedian == 'median':
        raster_medians = []
    else:
        raster_means = []
    raster_std_devs = []
    distances = []

    if frequencyplot:
        # Compute the bins intervals for the histogram
        bin_min = np.nanmin(raster_array)
        bin_max = np.nanmax(raster_array)
        bin_inc = (bin_max - bin_min) / bins
        bins_values = np.arange(start = bin_min, #start = bin_min - bin_inc,
                                stop = bin_max, #bin_max + bin_inc,
                                step = bin_inc)

    # Initiate loop on features
    k = 0
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if not os.path.isdir('TEMP_' + str(k)):
            # For each feature, if it does not exists,create a temporary folder, 
            # where intermediary shapefiles will be stored.
            # It will be erase at the end of the Loop if TEMP == False (default)
            os.mkdir('TEMP_' + str(k))

        # Check geometry type
        if geometry.GetGeometryType() == ogr.wkbLineString or geometry.GetGeometryType() == ogr.wkbPoint:
            coords = np.array(geometry.GetPoints())

            # Create a Shapely LineString
            line = LineString(coords)
            
            if frequencyplot:
                i = 0
            # Define the progress-bar
            with alive_bar(len(range(0, int(line.length), spl_incrt)), title = "\x1b[32;1m- Processing profile\x1b[0m", length = 35) as bar:    
                # Sample raster values at each point along the LineString
                for distance in range(0, int(line.length), spl_incrt):
                    
                    # Compute the box on which to compute stats
                    values_around_point, stats = buildbox(raster_path, raster_dataset, 
                                                          line, distance, k, 
                                                          spl_wndw, spl_incrt)

                    point = line.interpolate(distance)
                    x_coord, y_coord = point.x, point.y

                    px = int((x_coord - transform[0]) / transform[1])  # X coordinate in raster
                    py = int((y_coord - transform[3]) / transform[5])  # Y coordinate in raster

                    # Xav : Si tu veux connaître le nombre de NaN et de non Nan :
                    #print('NaN Values: ', np.count_nonzero(np.isnan(values_around_point)))
                    #print('Non NaN Values: ', np.count_nonzero(~np.isnan(values_around_point)))

                    if minmax:
                        # If plot min and max, update min/max arrays
                        min_value = np.nanmin(values_around_point)
                        max_value = np.nanmax(values_around_point)
                        raster_mins.append(min_value)
                        raster_maxs.append(max_value)
                    if meanmedian == 'median':
                        # Update median array
                        median_value = np.nanmedian(values_around_point)
                        raster_medians.append(median_value)
                    else:
                        # Update mean array
                        mean_value = np.nanmean(values_around_point)
                        raster_means.append(mean_value)
                    # Update standard deviation array
                    std_dev_value = np.nanstd(values_around_point)
                    raster_std_devs.append(std_dev_value)
                
                    if frequencyplot:
                        raster_freqs_values, bin_edges_values = np.histogram(values_around_point, 
                                                                    bins = bins_values,
                                                                    density = 'True')
                        if i == 0:
                            bin_edges = bin_edges_values
                            raster_freqs = raster_freqs_values/(values_around_point.size)
                        else:    
                            bin_edges = np.vstack((bin_edges, bin_edges_values))
                            raster_freqs = np.vstack((raster_freqs, raster_freqs_values/(values_around_point.size)))
                        i += 1
                    
                    # update bar
                    bar()
                    # update the distance array
                    distances.append(distance)

        # clean the TEMPs folder by default
        if not TEMP:
            print('\x1b[32;1m- Delating Temp...\x1b[0m')
            rmtree('TEMP_' + str(k))

        k += 1

    print('\x1b[32;1m- Plotting...\x1b[0m')
    
    if subplots:
        # Plot the raster with the shapefile
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    else:
        # just beging to plot the raster with the line
        fig, ax1 = plt.subplots()

    # First graph
    # Plot the raster
    img = ax1.imshow(raster_array, extent=(transform[0], transform[0] + transform[1] * raster_dataset.RasterXSize,
                                    transform[3] + transform[5] * raster_dataset.RasterYSize, transform[3]),
                    cmap='terrain', alpha=1)
    cbar = fig.colorbar(img, ax=ax1, label='Pixel Value', shrink=0.7)

    # Plot the shapefile polygons and segments
    for feature in layer:
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryType() == ogr.wkbPolygon:
            coords = np.array(geometry.GetGeometryRef(0).GetPoints())
            polygon = LineString(coords)

            # Plot the original polygon
            ax1.plot(coords[:, 0], coords[:, 1], linestyle='-', color='red')

            # Create a buffer around the polygon
            buffer_polygon = polygon.buffer(spl_wndw/2)

            # Plot the buffer
            buffer_x, buffer_y = buffer_polygon.exterior.xy
            ax1.fill(buffer_x, buffer_y, color='red', alpha=0.3)

        elif geometry.GetGeometryType() == ogr.wkbLineString or geometry.GetGeometryType() == ogr.wkbPoint:
            coords = np.array(geometry.GetPoints())
            line = LineString(coords)

            # Plot the original line
            ax1.plot(coords[:, 0], coords[:, 1], linestyle='-', color='red')

            # Create a buffer around the line
            buffer_line = line.buffer(spl_wndw/2)

            # Plot the buffer
            buffer_x, buffer_y = buffer_line.exterior.xy
            ax1.fill(buffer_x, buffer_y, color='red', alpha=0.3)        

    # Add scale bar to the map
    scalebar = ScaleBar(dx = px_size, location='lower right', 
                        length_fraction=0.1, width_fraction=0.01, 
                        border_pad=0.2,
                        box_alpha = 0.7)
    ax1.add_artist(scalebar)
    # Setting the number of ticks 
    ax1.locator_params(axis='both', nbins=6)
    for tick in ax1.get_yticklabels():
        tick.set_rotation(90)

    if not subplots:
        # Show the plots
        #plt.legend()
        plt.tight_layout()
        # save figure
        pdf_file_path = "Outs/output_Raster_" + outfile + ".pdf"
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
        # Close the figure to release resources
        plt.close(fig)
        # Reopen a new figure to plot the swath
        fig, ax2 = plt.subplots()

    # Second graph !
    # First, check if we plot in meters or kilometers
    km = False
    if np.max(distances) > 2500:
        distances = np.array(distances) / 1000
        km = True

    # Plot the graph distance = f(median raster value)
    if frequencyplot:   # Plot the frequency of raster values if needed
        img = ax2.imshow(raster_freqs.T, aspect = 'auto',
                         #aspect = 1000,
                         extent = (min(distances), max(distances),
                                   #np.nanmin(raster_array), np.nanmax(raster_array)),
                                   np.nanmin(bin_edges), np.nanmax(bin_edges)),
                         cmap = 'Greys', 
                         alpha = 1,
                         origin = "lower")
        cbar = fig.colorbar(img, ax=ax2, label='Frequency', shrink=0.7)
        # Save the stats in a pickle format
        f = open("Outs/rater_freqs_" + outfile + ".pickle", 'wb')
        pickle.dump(raster_freqs, f)
        f.close()
        f = open("Outs/bin_edges_" + outfile + ".pickle", 'wb')
        pickle.dump(bin_edges, f)

    std_dev_array = np.array(raster_std_devs)

    # plot medians
    if meanmedian == 'median':
        ax2.plot(distances, raster_medians, linestyle='-', color='red', label='median value')
        median_array = np.array(raster_medians)
        # fill in between
        ax2.fill_between(distances, median_array + std_dev_array, median_array - std_dev_array, color='r', alpha=0.3, label='±1σ')
    else:
        ax2.plot(distances, raster_means, linestyle='-', color='red', label='mean value')
        mean_array = np.array(raster_means)
        # fill in between
        ax2.fill_between(distances, mean_array + std_dev_array, mean_array - std_dev_array, color='r', alpha=0.3, label='±1σ')


    if minmax:
        # plot maxs
        ax2.plot(distances, raster_maxs, linestyle='--', linewidth = 0.5, color='blue', label='min/max value')
        # plot mins
        ax2.plot(distances, raster_mins, linestyle='--', linewidth = 0.5, color='blue')#, label='min value')

    #median_array = np.array(raster_medians)
    ## fill in between
    #ax2.fill_between(distances, median_array + std_dev_array, median_array - std_dev_array, color='r', alpha=0.3, label='±1σ')

    if km:
        ax2.set_xlabel('Distance along profile [km]') 
    else:
        ax2.set_xlabel('Distance [m]')
    ax2.set_ylabel('Median Raster Value')

    # Show the plots
    plt.legend()
    plt.tight_layout()
    # Impose y limits if needed
    if ylim != None:
        ax2.set_ylim((ylim))

    # Save the figure as PDF
    if subplots:
        pdf_file_path = "Outs/output_" + outfile + ".pdf"
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')
    else:
        pdf_file_path = "Outs/output_Swath_" + outfile + ".pdf"
        plt.savefig(pdf_file_path, format='pdf', bbox_inches='tight')

    ## Save the figure as PNG --> Xav: Je commente cet export pour gagner du temps/espace 
    #png_file_path = "Outs/output_" + outfile + "_figure.png"
    #plt.savefig(png_file_path, format='png', bbox_inches='tight')

    plt.close(fig)  # Close the figure to release resources
    
    # Write the data to a CSV file
    if meanmedian == 'median':
        csv_file_path = "Outs/median_raster_" + outfile + "_values_with_std_dev.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Distance [meters]', 'Median Raster Value', 'Standard Deviation'])
            for distance, median, std_dev in zip(distances, raster_medians, raster_std_devs):
                csv_writer.writerow([distance, median, std_dev])
    else:
        csv_file_path = "Outs/mean_raster_" + outfile + "_values_with_std_dev.csv"
        with open(csv_file_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Distance [meters]', 'Mean Raster Value', 'Standard Deviation'])
            for distance, mean, std_dev in zip(distances, raster_means, raster_std_devs):
                csv_writer.writerow([distance, mean, std_dev])

    # Close the datasets
    raster_dataset = None
    shapefile_dataset = None

    print('\x1b[32;1m- Done.\x1b[0m\n')
    return 

#########################################################################################

if __name__ == '__main__':
    
    # Example usage:
    
    raster_file = "Test/Rasters/DEM_clipped_EPSG6342.tif" # --> OK
    outfile = 'DEM'    # Je te rajoute une variable pour moduler directement le nom de sortie ici, sans aller modifier le code
    
    #raster_file = "Test/Rasters/20150910-20190914-velocity_EPSG6342.tif" # --> OK
    #outfile = 'velocity'

    #raster_file = "Test/Rasters/19790904-19930904_velocity_EPSG6342.tif" # --> OK
    #outfile = 'velocity-2'    

    #raster_file = "Test/Rasters/19581007-19790904_velocity_EPSG6342.tif" # --> OK
    #outfile = 'velocity-3'    

    shapefile_file = "Test/SHP/Longitudinal_transect_EPSG6342.shp"

    #increment_value = 10
    #window_size = 100

    increment_value = 20
    window_size = 100
    bins = 100  # Needs to be adjest in function of the min/max of the whole raster

    # to plot the mean or the median
    meanmedian = 'median'  # defaults value
    #meanmedian = 'mean'

    # To add the frequency plot
    frequencyplot = True
    #frequencyplot = False
    # To plot min-max values
    minmax = True

    # To keep temporary (shp) files :
    #TEMP = True
    TEMP = False

    # Set the ylim of the profil if needed; None by default
    #ylim = (0,2.6)
    ylim = None    # if option not used

    # Set if you want the rater plot and the swaht on the same fingure as sub-plots (True)
    subplots = False
    #subplots = True
    
    swath(raster_file, shapefile_file, outfile,
            increment_value, window_size, bins,
            meanmedian, minmax, frequencyplot, TEMP,
            ylim, subplots)

