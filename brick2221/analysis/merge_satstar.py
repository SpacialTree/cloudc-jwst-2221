import numpy as np
import time
import datetime
import os
import sys
import warnings
from astropy.io import fits
import glob
from photutils.background import MMMBackground, MADStdBackgroundRMS
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder, IRAFStarFinder, find_peaks
from photutils.psf import (IntegratedGaussianPRF, extract_stars, EPSFBuilder)
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import stats
from astropy.table import Table, Column, MaskedColumn
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import coordinates
from astropy.visualization import simple_norm
from astropy import wcs
from astropy import table
from astropy import units as u
from astroquery.svo_fps import SvoFps
from astropy.stats import sigma_clip, mad_std
import dask
import dask.array

from tqdm.auto import tqdm
import pylab as pl

filternames = ['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N']

def combine_singleframe(tbls, max_offset=0.10 * u.arcsec, realign=False, nanaverage=nanaverage_dask,
                        min_offset=0.01*u.arcsec,
                        offsets_table=None,
                        verbose=True
                        ):
    """

    min_offset :
        The minimum allowed offset to declare a 'new' star.  Anything below this is assumed same star.

    offsets_table:
        A table to use to re-calculate sky coordinates from the WCS after
        shifting it.  This can be used because the catalogs are all
        intrinsically in pixel space, so changing the shift after the fact is OK.
        Using an offset table enables splitting out the re-alignment task from
        here; I want to be able to measure the alignment and be sure it's right
        before applying it.
    """
    if offsets_table is not None:
        tbls = [shift_individual_catalog(tbl, offsets_table, verbose=verbose) for tbl in tbls]

    dao = True
    qfcn = 'qfit'
    ffcn = 'cfit'
    flux_error_colname = 'flux_err'
    flux_colname = 'flux_fit'
    # skycoord comes in as skycoord_centroid but we want it to leave as skycoord
    skycoord_colname = 'skycoord_fit'
    column_names = (flux_colname, flux_error_colname, 'qfit', 'cfit', 'flux_init', 'flags', 'local_bkg', 'iter_detected', 'group_id', 'group_size', 'ra', 'dec', 'dra', 'ddec', )

    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]
        if ii == 0:
            basecrds = crds
        else:
            matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
            reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

            # mutual_reverse_matches = (matches[reverse_matches] == np.arange(len(reverse_matches)))
            # mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))
            # even if the match is not mutual, consider it the same star as an existing one because it's too close.
            # keep = (sep > max_offset) | ((~mutual_matches) & (sep  > min_offset))
            keep = sep > max_offset

            newcrds = crds[keep]
            basecrds = SkyCoord([basecrds, newcrds])
            print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={len(basecrds)}]")
            # f" ({mutual_matches.sum()} mutual matches ({(~mutual_matches).sum()} not), {(sep > max_offset).sum()} above {max_offset}, keeping {keep.sum()}), ", flush=True)
        print(f"Iteration {ii}: There are a total of {len(basecrds)} sources in the base coordinate list [method={'dao' if dao else 'crowdsource'}]")

    # do one loop of re-matching
    print("Starting re-matching", flush=True)
    for ii, tbl in enumerate(tbls):
        crds = tbl[skycoord_colname]

        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_reverse_matches = (match_inds[reverse_match_inds] == np.arange(len(reverse_match_inds)))
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))

        # do one iteration of bulk offset measurement
        radiff = (crds.ra[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].ra).to(u.arcsec)
        decdiff = (crds.dec[reverse_match_inds[mutual_reverse_matches]] - basecrds[mutual_reverse_matches].dec).to(u.arcsec)

        # don't allow sep=0, since that's self-reference.  Use stringent qf, fracflux
        # print(f"len(crds) = {len(crds)} len(basecrds) = {len(basecrds)} len(match_inds)={len(match_inds)} match_inds.max={match_inds.max()} len(reverse_match_inds)={len(reverse_match_inds)} reverse_match_inds.max={reverse_match_inds.max()} len(mutual_matches)={len(mutual_matches)}")
        if dao:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] < 0.40) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] < 0.40)
        else:
            oksep = (reverse_sep[mutual_reverse_matches] < max_offset) & (reverse_sep[mutual_reverse_matches] != 0) & (tbl[reverse_match_inds[mutual_reverse_matches]][qfcn] > 0.95) & (tbl[reverse_match_inds[mutual_reverse_matches]][ffcn] > 0.85)
        medsep_ra, medsep_dec = np.median(radiff[oksep]), np.median(decdiff[oksep])
        dmedsep_ra, dmedsep_dec = mad_std(radiff[oksep]), mad_std(decdiff[oksep])
        tbl.meta['ra_offset'] = medsep_ra
        tbl.meta['dec_offset'] = medsep_dec
        tbl.meta['dra_offset'] = dmedsep_ra
        tbl.meta['ddec_offset'] = dmedsep_dec

        with fits.open(tbl.meta['FILENAME']) as fh:
            dra_header = fh['SCI'].header['RAOFFSET']
            ddec_header = fh['SCI'].header['DEOFFSET']

        print(f"Exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']} was offset by {medsep_ra.to(u.marcsec):10.3f}+/-{dmedsep_ra.to(u.marcsec):7.3f},"
              f" {medsep_dec.to(u.marcsec):10.3f}+/-{dmedsep_dec.to(u.marcsec):7.3f} based on {oksep.sum()} matches.  dra={dra_header:7.5g} ddec={ddec_header:7.5g}")

        # for tbl0, should be nan (all self-match)
        if realign and not np.isnan(medsep_ra) and not np.isnan(medsep_dec):
            newcrds = SkyCoord(crds.ra - medsep_ra, crds.dec - medsep_dec, frame=crds.frame)
            tbl[skycoord_colname] = newcrds

    if realign:
        print("Realigning")
        # remake base coordinates after the rematching
        for ii, tbl in enumerate(tbls):
            crds = tbl[skycoord_colname]
            if ii == 0:
                basecrds = crds
            else:
                matches, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
                # reverse_matches, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)

                # mutual_reverse_matches = (matches[reverse_matches] == np.arange(len(reverse_matches)))
                # mutual_matches = (reverse_matches[matches] == np.arange(len(matches)))
                # keep = (sep > max_offset) | (~mutual_matches)
                keep = (sep > max_offset)

                newcrds = crds[keep]
                basecrds = SkyCoord([basecrds, newcrds])
                print(f"Added {len(newcrds)} new sources in exposure {tbl.meta['exposure']} {tbl.meta['MODULE' if 'MODULE' in tbl.meta else '']}")
                # f" ({mutual_matches.sum()} mutual matches ({(~mutual_matches).sum()} not), {(sep > max_offset).sum()} above {max_offset}, keeping {keep.sum()}), ", flush=True)

    print(f"There are a total of {len(basecrds)} sources in the base coordinate list after rematching")

    assert flux_error_colname in tbls[0].colnames
    assert flux_error_colname in column_names

    # this segment, from arrays = down to the end, uses a lot of memory

    arrays = {key: np.zeros([len(basecrds), len(tbls)], dtype='float') * np.nan
              for key in column_names if key in tbls[0].colnames or key in ('skycoord', 'ra', 'dec')}

    for ii, tbl in enumerate(tqdm(tbls, desc='Table Loop (stack)')):
        crds = tbl[skycoord_colname]

        # match_inds & mutual_matches have the shape of basecrds, i.e., they are set by the crossmatching above
        match_inds, sep, _ = crds.match_to_catalog_sky(basecrds, nthneighbor=1)
        reverse_match_inds, reverse_sep, _ = basecrds.match_to_catalog_sky(crds, nthneighbor=1)
        mutual_matches = (reverse_match_inds[match_inds] == np.arange(len(match_inds)))

        # only add sources to a row in basecrd if it is the closest star to that row
        keep = (sep < max_offset) & (mutual_matches)

        for key in arrays:
            if key not in ('skycoord', skycoord_colname, 'ra', 'dec'):
                arrays[key][match_inds[keep], ii] = tbl[key][keep]
        arrays['ra'][match_inds[keep], ii] = tbl[skycoord_colname].ra[keep]
        arrays['dec'][match_inds[keep], ii] = tbl[skycoord_colname].dec[keep]
        print(f"{ii}: Added {keep.sum()} of {len(keep)} sources from exposure {tbl.meta['exposure']} {tbl.meta['MODULE'] if 'MODULE' in tbl.meta else ''} [total={len(basecrds)}]", flush=True)

    print("Compiling arrays into table", flush=True)
    print(f"Column names are {arrays.keys()} and should be {column_names}", flush=True)
    arrays['skycoord'] = SkyCoord(ra=arrays['ra'], dec=arrays['dec'], frame='icrs', unit=(u.deg, u.deg))
    del arrays['ra']
    del arrays['dec']

    newtbl = Table(arrays)
    newtbl.meta = tbls[0].meta
    newtbl.meta['offsets'] = {tbl.meta['exposure']: (tbl.meta['ra_offset'], tbl.meta['dec_offset']) for tbl in tbls}

    newtbl['nmatch'] = np.isfinite(newtbl[flux_colname]).sum(axis=1)

    # note: mad_std must be in quotes b/c it's using _fast_sigma_clip
    clip_flux = sigma_clip(newtbl[flux_colname], stdfunc='mad_std', axis=1)
    clip_ra = sigma_clip(newtbl['skycoord'].ra.deg, stdfunc='mad_std', axis=1)
    clip_dec = sigma_clip(newtbl['skycoord'].dec.deg, stdfunc='mad_std', axis=1)
    to_mask = clip_flux.mask | clip_ra.mask | clip_dec.mask

    newtbl['mask'] = to_mask
    newtbl['nmatch_good'] = (~to_mask).sum(axis=1)

    keepmask = ~newtbl['mask']
    weights = 1 / newtbl[flux_error_colname]**2 * keepmask
    avgpos = SkyCoord(nanaverage(newtbl['skycoord'].ra.value, axis=1, weights=weights),
                      nanaverage(newtbl['skycoord'].dec.value, axis=1, weights=weights),
                      unit=(u.deg, u.deg),
                      frame='icrs')
    newtbl['skycoord_avg'] = avgpos
    newtbl['std_ra'] = nanaverage((newtbl['skycoord'].ra - avgpos.ra[:, None])**2, weights=weights, axis=1)**0.5
    newtbl['std_dec'] = nanaverage((newtbl['skycoord'].dec - avgpos.dec[:, None])**2, weights=weights, axis=1)**0.5

    print("Propagating flux error")
    newtbl[f'{flux_error_colname}_prop'] = (np.nansum(newtbl[flux_error_colname]**2 * weights, axis=1) / np.nansum(weights, axis=1))**0.5
    newtbl.meta[f'{flux_error_colname}_prop'] = 'propagated uncertainty on flux = 1/sum(weights)'

    for key in column_names:
        if key in newtbl.colnames:
            print(f"Propagating {key}")
            newtbl[f'{key}_avg'] = nanaverage(newtbl[f'{key}'], weights=weights, axis=1)
            newtbl[f'std_{key}_avg'] = nanaverage((newtbl[f'{key}'] - newtbl[f'{key}_avg'][:, None])**2, weights=weights, axis=1)**0.5
        else:
            print(f"Skipping {key}")

    return newtbl



def main():
    print("Starting main")
    import time
    t0 = time.time()

    



if __name__ == "__main__":
    main()