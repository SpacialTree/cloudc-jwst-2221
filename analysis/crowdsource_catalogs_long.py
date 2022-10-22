import numpy as np
import crowdsource
import regions
import numpy as np
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import wcs
from astropy import table
from astropy import stats
from astropy import units as u
from astropy.io import fits
import requests
import urllib3
import urllib3.exceptions
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import DAOGroup, IntegratedGaussianPRF, extract_stars, IterativelySubtractedPSFPhotometry, BasicPSFPhotometry
from photutils.background import MMMBackground, MADStdBackgroundRMS

from crowdsource import crowdsource_base
from crowdsource.crowdsource_base import fit_im, psfmod

from astroquery.svo_fps import SvoFps

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'

import os
os.environ['WEBBPSF_PATH'] = '/orange/adamginsburg/jwst/webbpsf-data/'
with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
    os.environ['MAST_API_TOKEN'] = fh.read().strip()
import webbpsf


basepath = '/orange/adamginsburg/jwst/brick/'

for module in ('merged', 'nrca', 'nrcb', 'merged-reproject', ):
    for filtername in ('F405N', 'F410M', 'F466N'):
        fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
        row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
        fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
        fwhm_pix = float(row['PSF FWHM (pixel)'][0])

        try:
            pupil = 'clear'
            fh = fits.open(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits')
        except Exception:
            pupil = 'F444W'
            fh = fits.open(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_i2d.fits')

        im1 = fh
        data = im1[1].data
        wht = im1['WHT'].data
        err = im1['ERR'].data
        instrument = im1[0].header['INSTRUME']
        telescope = im1[0].header['TELESCOP']
        filt = im1[0].header['FILTER']

        wavelength_table = SvoFps.get_transmission_data(f'{telescope}/{instrument}.{filt}')
        obsdate = im1[0].header['DATE-OBS']

        has_downloaded = False
        while not has_downloaded:
            try:
                nrc = webbpsf.NIRCam()
                nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                nrc.filter = filt
                if module in ('nrca', 'nrcb'):
                    nrc.detector = f'{module.upper()}5' # I think NRCA5 must be the "long" detector?
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=False)
                else:
                    grid = nrc.psf_grid(num_psfs=16, all_detectors=True)
            except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                print(f"Failed to build PSF: {ex}")
            except Exception as ex:
                print(ex)
                continue

        yy, xx = np.indices([31,31], dtype=float)
        grid.x_0 = grid.y_0 = 15.5
        psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

        ww = wcs.WCS(im1[1].header)
        cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2) 
        reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
        preg = reg.to_pixel(ww)
        #mask = preg.to_mask()
        #cutout = mask.cutout(im1[1].data)
        #err = mask.cutout(im1[2].data)

        # crowdsource uses inverse-sigma, not inverse-variance
        weight = err**-1
        maxweight = np.percentile(weight[np.isfinite(weight)], 95)
        minweight = np.percentile(weight[np.isfinite(weight)], 5)
        weight[err < 1e-5] = 0
        weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
        weight[np.isnan(weight)] = 0

        weight[weight > maxweight] = maxweight
        weight[weight < minweight] = minweight



        filter_table = SvoFps.get_filter_list(facility=telescope, instrument=instrument)
        filter_table.add_index('filterID')
        instrument = 'NIRCam'
        eff_wavelength = filter_table.loc[f'{telescope}/{instrument}.{filt}']['WavelengthEff'] * u.AA

        #fwhm = (1.22 * eff_wavelength / (6.5*u.m)).to(u.arcsec, u.dimensionless_angles())

        pixscale = ww.proj_plane_pixel_area()**0.5
        #fwhm_pix = (fwhm / pixscale).decompose().value

        daogroup = DAOGroup(2 * fwhm_pix)
        mmm_bkg = MMMBackground()

        filtered_errest = stats.mad_std(data, ignore_nan=True)
        print(f'Error estimate for DAO: {filtered_errest}')

        daofind_fin = DAOStarFinder(threshold=10 * filtered_errest, fwhm=fwhm_pix, roundhi=1.0, roundlo=-1.0,
                                    sharplo=0.30, sharphi=1.40)
        finstars = daofind_fin(data)

        #grid.x_0 = 0
        #grid.y_0 = 0
        # not needed? def evaluate(x, y, flux, x_0, y_0):
        # not needed?     """
        # not needed?     Evaluate the `GriddedPSFModel` for the input parameters.
        # not needed?     """
        # not needed?     # Get the local PSF at the (x_0,y_0)
        # not needed?     psfmodel = grid._compute_local_model(x_0+slcs[1].start, y_0+slcs[0].start)

        # not needed?     # now evaluate the PSF at the (x_0, y_0) subpixel position on
        # not needed?     # the input (x, y) values
        # not needed?     return psfmodel.evaluate(x, y, flux, x_0, y_0)
        # not needed? grid.evaluate = evaluate

        phot = BasicPSFPhotometry(finder=daofind_fin,#finder_maker(),
                                    group_maker=daogroup,
                                    bkg_estimator=None, # must be none or it un-saturates pixels
                                    psf_model=grid,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(11, 11),
                                    aperture_radius=5*fwhm_pix)

        result = phot(data)
        coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}')
        result['skycoord_centroid'] = coords
        detector = "" # no detector #'s for long
        result.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_daophot_basic.fits", overwrite=True)

        if False:
            # iterative takes for-ev-er
            phot_ = IterativelySubtractedPSFPhotometry(finder=daofind_fin, group_maker=daogroup,
                                                        bkg_estimator=mmm_bkg,
                                                        psf_model=grid,
                                                        fitter=LevMarLSQFitter(),
                                                        niters=2, fitshape=(11, 11),
                                                        aperture_radius=2*fwhm_pix)

            result2 = phot_(data)
            coords2 = ww.pixel_to_world(result2['x_fit'], result2['y_fit'])
            result2['skycoord_centroid'] = coords2
            print(f'len(result2) = {len(result2)}, len(coords) = {len(coords)}')
            result2.write(f"{basepath}/{filtername}/{filtername.lower()}_{module}{detector}_daophot_iterative.fits", overwrite=True)



        results_unweighted  = fit_im(data, psf_model, weight=np.ones_like(data)*np.nanmedian(weight),
                                        #psfderiv=np.gradient(-psf_initial[0].data),
                                        nskyx=1, nskyy=1, refit_psf=False, verbose=True)
        stars, modsky, skymsky, psf = results_unweighted
        stars = Table(stars)
        # crowdsource explicitly inverts x & y from the numpy convention:
        # https://github.com/schlafly/crowdsource/issues/11
        coords = ww.pixel_to_world(stars['y'], stars['x'])
        stars['skycoord'] = coords
        stars['x'], stars['y'] = stars['y'], stars['x']

        tbl = fits.BinTableHDU(data=stars)
        hdu = fits.HDUList([fits.PrimaryHDU(header=im1[1].header), tbl])
        hdu.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_unweighted.fits", overwrite=True)
        fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_skymodel_unweighted.fits", overwrite=True)


        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
        pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=5, linewidth=0.5)
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_unweighted.png',
                   bbox_inches='tight')

        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data[:128,:128], norm=simple_norm(data[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky[:128,:128], norm=simple_norm(modsky[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow(skymsky[:128,:128], norm=simple_norm(skymsky[:128,:128], stretch='asinh'), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
        pl.subplot(2,2,4).imshow(data[:128,:128], norm=simple_norm(data[:128,:128], stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=8, linewidth=0.5)
        pl.axis([0,128,0,128])
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_zoom_unweighted.png',
                   bbox_inches='tight')

        # pl.figure(figsize=(10,5))
        # pl.subplot(1,2,1).imshow(psf_model(30,30), norm=simple_norm(psf_model(30,30), stretch='log'), cmap='cividis')
        # pl.title("Input model")
        # pl.subplot(1,2,2).imshow(psf(30,30), norm=simple_norm(psf(30,30), stretch='log'), cmap='cividis')
        # pl.title("Fitted model")



        yy, xx = np.indices([61, 61], dtype=float)
        grid.x_0 = preg.center.x+30
        grid.y_0 = preg.center.y+30
        gpsf2 = grid(xx+preg.center.x, yy+preg.center.y)
        psf_model = crowdsource.psf.SimplePSF(stamp=gpsf2)

        smoothing_scale = 0.55 # pixels
        gpsf3 = convolve(gpsf2, Gaussian2DKernel(smoothing_scale))
        psf_model_blur = crowdsource.psf.SimplePSF(stamp=gpsf3)

        fig = pl.figure(0, figsize=(10,10))
        fig.clf()
        ax = fig.gca()
        im = ax.imshow(weight, norm=simple_norm(weight, stretch='log')); pl.colorbar(mappable=im);
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_weights.png',
                   bbox_inches='tight')


        results_blur  = fit_im(data, psf_model_blur, weight=weight,
                            nskyx=1, nskyy=1, refit_psf=False, verbose=True)
        stars, modsky, skymsky, psf = results_blur
        stars = Table(stars)

        # crowdsource explicitly inverts x & y from the numpy convention:
        # https://github.com/schlafly/crowdsource/issues/11
        coords = ww.pixel_to_world(stars['y'], stars['x'])
        stars['skycoord'] = coords
        stars['x'], stars['y'] = stars['y'], stars['x']

        tbl = fits.BinTableHDU(data=stars)
        hdu = fits.HDUList([fits.PrimaryHDU(header=im1[1].header), tbl])
        hdu.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource.fits", overwrite=True)
        fits.PrimaryHDU(data=skymsky, header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_skymodel.fits", overwrite=True)
        fits.PrimaryHDU(data=data-modsky,
                        header=im1[1].header).writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}_crowdsource_data-modsky.fits", overwrite=True)

        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data[:128,:128], norm=simple_norm(data[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky[:128,:128], norm=simple_norm(modsky[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow((data-modsky)[:128,:128], norm=simple_norm((data-modsky)[:256,:256], stretch='asinh', max_percent=99.5, min_percent=0.5), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("data-modsky")
        pl.subplot(2,2,4).imshow(data[:128,:128], norm=simple_norm(data[:256,:256], stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=8, linewidth=0.5)
        pl.axis([0,128,0,128])
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.suptitle("Using WebbPSF model blurred a little")
        pl.tight_layout()
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics_zoom.png',
                   bbox_inches='tight')

        pl.figure(figsize=(12,12))
        pl.subplot(2,2,1).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("Data")
        pl.subplot(2,2,2).imshow(modsky, norm=simple_norm(modsky, stretch='log', max_percent=99.95), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
        pl.subplot(2,2,3).imshow(skymsky, norm=simple_norm(skymsky, stretch='asinh'), cmap='gray')
        pl.xticks([]); pl.yticks([]); pl.title("fit_im sky+skym")
        pl.subplot(2,2,4).imshow(data, norm=simple_norm(data, stretch='log', max_percent=99.95), cmap='gray')
        pl.subplot(2,2,4).scatter(stars['y'], stars['x'], marker='x', color='r', s=5, linewidth=0.5)
        pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw02221-o001_t001_nircam_{pupil}-{filtername.lower()}-{module}_catalog_diagnostics.png',
                   bbox_inches='tight')

