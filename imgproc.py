''' This python library gathers a bunch of generic image processing routines '''

import numpy as np

# Astropy imports
import astropy.units as u
import astropy.coordinates as coord
import astropy.io.fits as pyf
from astropy import wcs
from scipy.special import hyp2f1
from mmm import mmm
# Photutils imports
from photutils import detect_sources, segment_properties, aperture_photometry, remove_segments
from photutils import CircularAperture, EllipticalAperture
from photutils import SkyCircularAperture, SkyEllipticalAperture
from photutils import CircularAnnulus, EllipticalAnnulus
from photutils import SkyCircularAnnulus, SkyEllipticalAnnulus
from photutils.geometry import circular_overlap_grid, elliptical_overlap_grid
from photutils.aperture_funcs import get_phot_extents
from astropy.stats import  gaussian_sigma_to_fwhm, sigma_clipped_stats, gaussian_fwhm_to_sigma

# Function to estimate the global background
# Taken from T. Shimizu's SPIRE photometry code
def estimate_bkg(image, clip=3.0, snr=2.0, npix=5, tol=1e-6, max_iter=10):

    # Create the NaN mask
    nanmask = np.isnan(image)
    
    # First estimate of the global background from sigma-clipping
    im_mean0, im_med0, im_std0 = sigma_clipped_stats(image, sigma=clip, mask=nanmask)
    
    # Create segmentation image using threshold = im_med + snr*im_std
    # Greater than npix pixels have to be connected to be a source
    thresh = im_med0 + snr*im_std0
    segm_img = detect_sources(image, thresh, npixels=npix)
    
    # Create new mask that masks all the detected sources
    mask = segm_img.astype(np.bool) | nanmask
    
    # Re-calculate background and rms
    im_mean, im_med, im_std = sigma_clipped_stats(image, sigma=clip, mask=mask)
    
    # Calculate percent change in the background estimate
    perc_change = np.abs(im_med0 - im_med)/ im_med
    
    if perc_change > tol:
    	im_med0 = im_med
        for i in range(max_iter):
 
    	    thresh = im_med + snr*im_std
            segm_img = detect_sources(image, thresh, npixels=npix)
            mask = segm_img.astype(np.bool) | nanmask
            im_mean, im_med, im_std = sigma_clipped_stats(image, sigma=clip, mask=mask)
            perc_change = np.abs(im_med0 - im_med)/ im_med

            if perc_change < tol:
            	break
            else:
            	im_med0 = im_med
            	
    return im_med, im_std


# Function to estimate the rms in an annulus by patching it with apertures
# Taken from T. Shimizu's SPIRE photometry code, slightly modified 
def calc_bkg_rms(ap, image, src_ap_area, rpsrc, mask=None, min_ap=6):

    if isinstance(ap, CircularAnnulus):
        aback = bback =  ap.r_in + rpsrc
        ap_theta = 0
    elif isinstance(ap, EllipticalAnnulus):
        aback = ap.a_in + rpsrc
        bback = ap.b_in + rpsrc
        ap_theta = ap.theta
    
    ecirc = ellip_circumference(aback, bback)
    diam = 2*rpsrc
    
    # Estimate the number of background apertures that can fit around the source
    # aperture.
    naps = np.int(np.round(ecirc/diam))
    
    # Use a minimum of 6 apertures
    naps = np.max([naps, min_ap])
    #naps = 6
    
    theta_back = np.linspace(0, 2*np.pi, naps, endpoint=False)
    
    # Get the x, y positions of the background apertures
    x, y = ellip_point(ap.positions[0], aback, bback, ap_theta, theta_back)

    # Create the background apertures and calculate flux within each
    bkg_aps = CircularAperture(np.vstack([x,y]).T, rpsrc)
    flux_bkg = aperture_photometry(image, bkg_aps, mask=mask)
    flux_bkg = flux_bkg['aperture_sum']
    flux_bkg_adj = flux_bkg/bkg_aps.area() * src_ap_area
 	
	# Use sigma-clipping to determine the RMS of the background
	# Scale to the area of the source aperture
    me, md, sd = sigma_clipped_stats(flux_bkg_adj, sigma=3)


    bkg_rms = sd
    
    return bkg_rms, bkg_aps


# Function to calculate the circumference of an ellipse
def ellip_circumference(a, b):
    t = ((a-b)/(a+b))**2
    return np.pi*(a+b)*hyp2f1(-0.5, -0.5, 1, t)


# Function to determine x, y, position of an ellipse
def ellip_point(pos, a, b, theta, alpha):
    x = a*np.cos(alpha)*np.cos(theta) - b*np.sin(alpha)*np.sin(theta) + pos[0]
    y = a*np.cos(alpha)*np.sin(theta) + b*np.sin(alpha)*np.cos(theta) + pos[1]    
    return x, y


def calc_masked_aperture(ap, image, method='mmm', mask=None):

    positions = ap.positions
    extents = np.zeros((len(positions), 4), dtype=int)
    
    if isinstance(ap, EllipticalAnnulus):
        radius = ap.a_out
    elif isinstance(ap, CircularAnnulus):
        radius = ap.r_out
    elif isinstance(ap, CircularAperture):
        radius = ap.r
    elif isinstance(ap, EllipticalAperture):
        radius = ap.a
    
    extents[:, 0] = positions[:, 0] - radius + 0.5
    extents[:, 1] = positions[:, 0] + radius + 1.5
    extents[:, 2] = positions[:, 1] - radius + 0.5
    extents[:, 3] = positions[:, 1] + radius + 1.5
    
    ood_filter, extent, phot_extent = get_phot_extents(image, positions,
                                                       extents)
    
    x_min, x_max, y_min, y_max = extent
    x_pmin, x_pmax, y_pmin, y_pmax = phot_extent
    
    bkg = np.zeros(len(positions))
    area = np.zeros(len(positions))
    
    for i in range(len(bkg)):
        if isinstance(ap, EllipticalAnnulus):
            fraction = elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                               y_pmin[i], y_pmax[i],
                                               x_max[i] - x_min[i],
                                               y_max[i] - y_min[i],
                                               ap.a_out, ap.b_out, ap.theta, 0,
                                               1)
            b_in = ap.a_in * ap.b_out / ap.a_out
            fraction -= elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                                y_pmin[i], y_pmax[i],
                                                x_max[i] - x_min[i],
                                                y_max[i] - y_min[i],
                                                ap.a_in, b_in, ap.theta,
                                                0, 1)
        elif isinstance(ap, CircularAnnulus):
            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             ap.r_out, 0, 1)
            
            fraction -= circular_overlap_grid(x_pmin[i], x_pmax[i],
                                                y_pmin[i], y_pmax[i],
                                                x_max[i] - x_min[i],
                                                y_max[i] - y_min[i],
                                                ap.r_in, 0, 1)
        elif isinstance(ap, CircularAperture):
            fraction = circular_overlap_grid(x_pmin[i], x_pmax[i],
                                             y_pmin[i], y_pmax[i],
                                             x_max[i] - x_min[i],
                                             y_max[i] - y_min[i],
                                             ap.r, 0, 1) 
        
        elif isinstance(ap, EllipticalAperture):
            fraction = elliptical_overlap_grid(x_pmin[i], x_pmax[i],
                                               y_pmin[i], y_pmax[i],
                                               x_max[i] - x_min[i],
                                               y_max[i] - y_min[i],
                                               ap.a, ap.b, ap.theta, 0,
                                               1)
        
        pixel_data = image[y_min[i]:y_max[i], x_min[i]:x_max[i]] * fraction
        if mask is not None:
            pixel_data[mask[y_min[i]:y_max[i], x_min[i]:x_max[i]]] = 0.0
        
        good_pixels = pixel_data[pixel_data != 0.0].flatten()
        
        if method == 'mmm':
            skymod, skysigma, skew = mmm(good_pixels)
            bkg[i] = skymod
        elif method == 'sum':
            bkg[i] = np.sum(good_pixels)
        elif method == 'max':
            bkg[i] = np.nanmax(good_pixels)
        area[i] = len(good_pixels)

    return bkg, area
