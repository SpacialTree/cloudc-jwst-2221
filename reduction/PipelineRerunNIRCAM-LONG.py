#!/usr/bin/env python
from glob import glob
from astroquery.mast import Mast, Observations
import os
import shutil
import numpy as np
import json
# import requests
import asdf
from astropy import log
from astropy.io import ascii, fits
from astropy.utils.data import download_file
from astropy.visualization import ImageNormalize, ManualInterval, LogStretch, LinearStretch
import matplotlib.pyplot as plt
import matplotlib as mpl

# do this before importing webb
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/brick/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

from jwst.pipeline import calwebb_image3
from jwst.pipeline import Detector1Pipeline, Image2Pipeline

# Individual steps that make up calwebb_image3
from jwst.tweakreg import TweakRegStep
from jwst.skymatch import SkyMatchStep
from jwst.outlier_detection import OutlierDetectionStep
from jwst.resample import ResampleStep
from jwst.source_catalog import SourceCatalogStep
from jwst import datamodels
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

from align_to_catalogs import realign_to_vvv, merge_a_plus_b
from saturated_star_finding import iteratively_remove_saturated_stars, remove_saturated_stars

from destreak import destreak

import crds

import datetime
import jwst

def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)


print(jwst.__version__)

# see 'destreak410.ipynb' for tests of this
medfilt_size = {'F410M': 15, 'F405N': 256, 'F466N': 55}

basepath = '/orange/adamginsburg/jwst/brick/'

def main(filtername, module, Observations=None):
    log.info(f"Processing filter {filtername} module {module}")


    basepath = '/orange/adamginsburg/jwst/brick/'
    os.environ["CRDS_PATH"] = f"{basepath}/crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80



    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/brick/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)

    # the files are one directory up
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except Exception as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id="2221",
                                            proposal_pi="Ginsburg*",
                                            #calib_level=3,
                                            )
    print("Obs table length:", len(obs_table))

    msk = ((np.char.find(obs_table['filters'], filtername.upper()) >= 0) |
           (np.char.find(obs_table['obs_id'], filtername.lower()) >= 0))
    data_products_by_obs = Observations.get_product_list(obs_table[msk])
    print("data prodcts by obs length: ", len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print("products_asn length:", len(products_asn))
    #valid_obsids = products_asn['obs_id'][np.char.find(np.unique(products_asn['obs_id']), 'jw02221-o001', ) == 0]
    #match = [x for x in valid_obsids if filtername.lower() in x][0]

    asn_mast_data = products_asn#[products_asn['obs_id'] == match]
    print("asn_mast_data:", asn_mast_data)

    manifest = Observations.download_products(asn_mast_data, download_dir=output_dir)
    print("manifest:", manifest)

    # MAST creates deep directory structures we don't want
    for row in manifest:
        try:
            shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
        except Exception as ex:
            print(f"Failed to move file with error {ex}")

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    uncal_mask = np.array([uri.endswith('_uncal.fits') for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
    uncal_mask &= ~already_downloaded
    print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)

        # MAST creates deep directory structures we don't want
        for row in manifest:
            try:
                shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
            except Exception as ex:
                print(f"Failed to move file with error {ex}")


    # all cases, except if you're just doing a merger?
    if module in ('nrca', 'nrcb', 'merged'):
        print(f"Searching for {os.path.join(output_dir, f'jw02221-*_image3_0[0-9][0-9]_asn.json')}")
        asn_file_search = glob(os.path.join(output_dir, f'jw02221-*_image3_0[0-9][0-9]_asn.json'))
        if len(asn_file_search) == 1:
            asn_file = asn_file_search[0]
        elif len(asn_file_search) > 1:
            asn_file = sorted(asn_file_search)[-1]
            print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
        else:
            raise ValueError("Mismatch")

        mapping = crds.rmap.load_mapping('/orange/adamginsburg/jwst/brick/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
        print(f"Mapping: {mapping.todict()['selections']}")
        print(f"Filtername: {filtername}")
        filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
        print(f"Filter_match: {filter_match} n={len(filter_match)}")
        tweakreg_asdf_filename = filter_match[0][4]
        tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
        tweakreg_parameters = tweakreg_asdf.tree['parameters']
        print(f'Filter {filtername}: {tweakreg_parameters}')


        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        print(f"In cwd={os.getcwd()}")
        # re-calibrate all uncal files -> cal files *without* suppressing first group
        for member in asn_data['products'][0]['members']:
            print(f"DETECTOR PIPELINE on {member['expname']}")
            print("Detector1Pipeline step")
            Detector1Pipeline.call(member['expname'].replace("_cal.fits",
                                                             "_uncal.fits"),
                                   save_results=True, output_dir=output_dir,
                                   steps={'ramp_fit': {'suppress_one_group':
                                                       False}})
            print(f"IMAGE2 PIPELINE on {member['expname']}")
            Image2Pipeline.call(member['expname'].replace("_cal.fits",
                                                          "_rate.fits"),
                                save_results=True, output_dir=output_dir,
                               )
    else:
        raise ValueError(f"Module is {module} - not allowed!")

    if module in ('nrca', 'nrcb'):
        print(f"Filter {filtername} module {module}")

        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)
        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}'
        asn_data['products'][0]['members'] = [row for row in asn_data['products'][0]['members']
                                                if f'{module}' in row['expname']]

        for member in asn_data['products'][0]['members']:
            hdr = fits.getheader(member['expname'])
            if filtername in (hdr['PUPIL'], hdr['FILTER']):
                outname = destreak(member['expname'],
                                   use_background_map=True,
                                   median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                member['expname'] = outname

        asn_file_each = asn_file.replace("_asn.json", f"_{module}_asn.json")
        with open(asn_file_each, 'w') as fh:
            json.dump(asn_data, fh)

        tweakreg_parameters.update({'fit_geometry': 'general',
                                    'brightest': 10000,
                                    'snr_threshold': 5,
                                    'nclip': 1,
                                    })

        calwebb_image3.Image3Pipeline.call(
            asn_file_each,
            steps={'tweakreg': tweakreg_parameters,},
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_each}")

        log.info(f"Realigning to VVV (module={module}")
        realign_to_vvv(filtername=filtername.lower(), module=module)

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits')

    if module == 'nrcb':
        # assume nrca is run before nrcb
        print("Merging already-combined nrca + nrcb modules")
        merge_a_plus_b(filtername)
        print("DONE Merging already-combined nrca + nrcb modules")

    if module == 'merged':
        # try merging all frames & modules
        log.info("Working on merged reduction (both modules)")

        # Load asn_data for both modules
        with open(asn_file) as f_obj:
            asn_data = json.load(f_obj)

        for member in asn_data['products'][0]['members']:
            hdr = fits.getheader(member['expname'])
            if filtername in (hdr['PUPIL'], hdr['FILTER']):
                outname = destreak(member['expname'],
                                    use_background_map=True,
                                    median_filter_size=2048)  # median_filter_size=medfilt_size[filtername])
                member['expname'] = outname

        asn_data['products'][0]['name'] = f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged'
        asn_file_merged = asn_file.replace("_asn.json", f"_merged_asn.json")
        with open(asn_file_merged, 'w') as fh:
            json.dump(asn_data, fh)

        vvvdr2fn = (f'{basepath}/{filtername.upper()}/pipeline/jw02221-o001_t001_nircam_clear-{filtername}-{module}_vvvcat.ecsv')
        print(f"Loaded VVV catalog {vvvdr2fn}")
        if os.path.exists(vvvdr2fn):
            tweakreg_parameters['abs_refcat'] = vvvdr2fn
            tweakreg_parameters['abs_searchrad'] = 1
        else:
            print(f"Did not find VVV catalog {vvvdr2fn}")

        tweakreg_parameters.update({'fit_geometry': 'general',
                                    'brightest': 10000,
                                    'snr_threshold': 5,
                                    'nclip': 1,

                                             })

        calwebb_image3.Image3Pipeline.call(
            asn_file,
            steps={'tweakreg': tweakreg_parameters,},
            output_dir=output_dir,
            save_results=True)
        print(f"DONE running {asn_file_merged}")

        log.info("Realigning to VVV (module=merged)")
        realign_to_vvv(filtername=filtername.lower(), module='merged')

        log.info("Removing saturated stars")
        remove_saturated_stars(f'jw02221-o001_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')

    globals().update(locals())
    return locals()

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                      default='F466N,F405N,F410M',
                      help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='merged,nrca,nrcb',
                    help="module list", metavar="modules")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    print(options)

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()
    Mast.login(api_token.strip())
    Observations.login(api_token)

    for filtername in filternames:
        for module in modules:
            print(f"Main Loop: {filtername} + {module}")
            results = main(filtername=filtername, module=module, Observations=Observations)


    print("Running notebooks")
    from run_notebook import run_notebook
    basepath = '/orange/adamginsburg/jwst/brick/'
    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')


"""
await app.openFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F410M/pipeline/jw02221-o001_t001_nircam_clear-f410m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F182M/pipeline/jw02221-o001_t001_nircam_clear-f182m-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F212N/pipeline/jw02221-o001_t001_nircam_clear-f212n-nrcb_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-merged_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrca_i2d.fits")
await app.appendFile("/jwst/brick/F466N/pipeline/jw02221-o001_t001_nircam_clear-f466n-nrcb_i2d.fits")
"""
