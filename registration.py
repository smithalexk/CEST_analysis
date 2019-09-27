#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from pathlib import Path
import shutil
from math import ceil, radians
import numpy as np
from nipype.interfaces import fsl
import nibabel as nib
import argparse
import sys
import json
import warnings

# Used in case CEST is a 2D slice (for use registering other datasets to Ref)
_slicenumber2d = list()


def register_cest(
    PathToData,
    CESTName,
    OffsetsName,
    OutFolder,
    B1Name,
    T1Name=None,
    RefName=None,
    b1FAname=None,
    RegDir="RegDir",
    phantom=False,
):

    """ Register CEST Data - Registers CEST Data to a Reference Dataset
    Registers CEST Data (CESTName) to a Reference Dataset (RefName). Also
    registers B1 Data (B1Name) and T1 Data (T1Name) if present in dataset
    
    Args:
        PathToData : pathlib Path Object 
            Path to the data to be processed
        CESTName : str
            File name prefix of the CEST data to be processed
            (utilizes wildcards at beginning/end of name)
        RefName : str
            Reference file name.  If not included will use
            the second dynamic of the 1st set CEST data as a
            reference.
        OffsetsName : pathlib Path object
            The filename for the Offsets file.
        B1Name : str
            The filename for the set of B1 files (needs an anatomical
            + the B1 map, or both anatomical images).  If B1 map not
            pre-calculated, will use the DREAM sequence.  Will also 
            divide the B1map by 600 in accordance with the DREAM 
            sequence output from the scanner.
        T1Name : str
            The filename of the T1 scans (optional).
    
    Returns:
        None
    """
    # Switch to Path Object for all Folders

    OutFolder = OutFolder
    regdir = PathToData / RegDir

    # Ensures enumerate will using entire string name instead of stepping through it
    if not isinstance(CESTName, (list, np.ndarray)):
        CESTName = [CESTName]

    if not regdir.is_dir():
        regdir.mkdir(parents=True)
    else:
        shutil.rmtree(str(regdir))
        regdir.mkdir(parents=True)

    # Generate identity matrix for use in registrations
    _genidentitymat(regdir)

    if RefName is None:

        # Ref Data Name
        RefPath = CESTName[0]

        # Splitting 2nd CEST image from rest for use as reference image
        Refdir = sorted(PathToData.glob("*{0}*nii.gz".format(RefPath)))
        fsl.ExtractROI(
            in_file=str(Refdir[0]),
            roi_file=str(regdir / "CEST_dyn2.nii.gz"),
            t_min=0,
            t_size=1,
        ).run()

        # Ref Data Name
        RefName = "CEST_dyn2"
        shutil.copyfile(
            str(regdir / "{0}.nii.gz".format(RefName)), str(regdir / "Ref_sROI.nii.gz")
        )
        shutil.copyfile(
            str(regdir / "{0}.nii.gz".format(RefName)),
            str(Path(PathToData) / "CEST_dyn2.nii.gz"),
        )

    _reg_ref(
        datapath=PathToData,
        refname=RefName,
        regdir=regdir,
        cest_name=CESTName[0],
        outfolder=OutFolder,
        phantom=phantom,
    )

    _b1resize(
        datapath=PathToData,
        b1name=B1Name,
        regdir=regdir,
        b1FAmap=b1FAname,
        outfolder=OutFolder,
        inrefname=RefName,
        phantom=phantom,
    )

    for idx, i in enumerate(CESTName):

        # Need to deal with cases where same offsets are used over multiple CEST/MT datasets
        if isinstance(OffsetsName, (list, np.ndarray)):
            offsetpath = PathToData / OffsetsName[idx]
        else:
            offsetpath = PathToData /OffsetsName 

        offsets = np.sort(np.loadtxt(offsetpath))

        if 0 in np.arange(np.floor(offsets[0]), np.ceil(offsets[-1]), 1):
            _cestreg(
                datapath=PathToData,
                cestprefix=i,
                offsetspath=offsetpath,
                regdir=regdir,
                outfolder=OutFolder,
                phantom=phantom,
            )
        else:
            _mtreg(
                datapath=PathToData,
                mtprefix=i,
                offsetspath=offsetpath,
                regdir=regdir,
                outfolder=OutFolder,
                phantom=phantom,
            )

    if T1Name is not None:
        _t1reg(
            datapath=PathToData,
            t1prefix=T1Name,
            regdir=regdir,
            outfolder=OutFolder,
            inrefname=RefName,
            phantom=phantom,
        )

    # Copy Offsets File into OutFolder
    shutil.copyfile(str(offsetpath), str(OutFolder / offsetpath.name))

    print("Finished!")

    return None


def register_mt(
    PathToData,
    mtName,
    OffsetsName,
    OutFolder,
    B1Name,
    T1Name,
    RefName=None,
    b1FAname=None,
    RegDir="RegDir",
    phantom=False,
):
    """ Wrapper to call register_cest. Left for legacy code
    """
    register_cest(
        PathToData=PathToData,
        CESTName=mtName,
        OffsetsName=OffsetsName,
        OutFolder=OutFolder,
        B1Name=B1Name,
        T1Name=T1Name,
        RefName=RefName,
        b1FAname=b1FAname,
        RegDir=RegDir,
        phantom=phantom,
    )

    return None


def _reg_ref(
    datapath,
    refname,
    regdir,
    cest_name,
    outfolder=None,
    phantom=False,
    refresize="Ref_CESTres.nii.gz",
):

    """_reg_ref_7T - Helper function to prep the reference volume for registration
    Resizes, BETs, creates a brain mask, and segments the Reference Volume
    for use in subsequent CEST Processing
    
    Parameters:
    -----------
        datapath : str
            The path to the datafolder
        refname : str
            The name of the high-resolution reference scan to register all data to.
        cest_name : str
            The name of the CEST data being analyzed.
        regdir : str
            The registration directory being used for all 
            of the registration computations.
        outfolder : str
            The output directory of the data being analyzed. 
            If blank, the datapath directory will be used.
        
    Returns:
    --------
        None

    """

    cestdir = sorted(datapath.glob("*{0}*.nii.gz".format(cest_name)))
    refdir = sorted(datapath.glob("*{0}*.nii.gz".format(refname)))

    try:
        cest_img = nib.load(str(cestdir[0]))
    except IndexError:
        raise NoCESTDataError(f"No CEST Data for {cest_name}")

    ref_img = nib.load(str(refdir[0]))
    try:
        dim3 = cest_img.shape[2]
    except IndexError:
        dim3 = 1

    # Perform Bias Field Correction on Reference image
    refbc = fsl.FAST()
    refbc.inputs.in_files = str(refdir[0])
    refbc.inputs.out_basename = str(regdir / "Ref_bc")
    refbc.inputs.output_biascorrected = True
    refbc.inputs.no_pve = True
    refbc.inputs.output_type = "NIFTI_GZ"
    refbc.run(ignore_exception=True)

    if dim3 == 1:
        sformref = ref_img.get_sform()
        sformcest = cest_img.get_sform()

        # Find center of CEST dataset
        cest_cent = [cest_img.shape[0] / 2 - 1, cest_img.shape[1] / 2 - 1, 0, 1]

        ref_coord = np.linalg.solve(sformref, sformcest @ cest_cent)

        Nslices = ceil(cest_img.header["pixdim"][3] / ref_img.header.get_zooms()[2])

        # Create ROI of the slices around the CEST slice
        if Nslices > 1:
            fsl.ExtractROI(
                in_file=str(regdir / "Ref_bc_restore.nii.gz"),
                roi_file=str(regdir / "Ref_sROI.nii.gz"),
                x_min=0,
                x_size=-1,
                y_min=0,
                y_size=-1,
                z_min=int(round(ref_coord[2]) - 1),
                z_size=Nslices,
            ).run()
        else:
            fsl.ExtractROI(
                in_file=str(regdir / "Ref_bc_restore.nii.gz"),
                roi_file=str(regdir / "Ref_sROI.nii.gz"),
                x_min=0,
                x_size=-1,
                y_min=0,
                y_size=-1,
                z_min=int(round(ref_coord[2])),
                z_size=Nslices,
            ).run()

        _slicenumber2d.append(int(round(ref_coord[2])))
        _slicenumber2d.append(Nslices)

        with (regdir / "Ref_SliceLocation.txt").open(mode="w") as FID:
            [FID.write(f"{slice}\n") for slice in _slicenumber2d]

    else:
        shutil.copyfile(
            str(regdir / "Ref_bc_restore.nii.gz"), str(regdir / "Ref_sROI.nii.gz")
        )

    cestsplt = fsl.ExtractROI()
    cestsplt.inputs.in_file = str(cestdir[0])
    cestsplt.inputs.roi_file = str(regdir / "CEST_ref.nii.gz")
    cestsplt.inputs.t_min = 0
    cestsplt.inputs.t_size = 1
    cestsplt.run()

    # Perform Bias Field Correction on CEST image
    if "ssfp" in cest_name:
        cestbet = fsl.BET()
        cestbet.inputs.in_file = str(regdir / "CEST_ref.nii.gz")
        cestbet.inputs.out_file = str(regdir / "CEST_brain.nii.gz")
        cestbet.inputs.padding = True
        cestbet.inputs.frac = 0.6
        cestbet.run()

        cestbc = fsl.FAST()
        cestbc.inputs.in_files = str(regdir / "CEST_brain.nii.gz")
        cestbc.inputs.out_basename = str(regdir / "CEST_bc")
        cestbc.inputs.output_biascorrected = True
        cestbc.inputs.output_biasfield = True
        cestbc.inputs.no_pve = True
        cestbc.inputs.output_type = "NIFTI_GZ"
        cestbc.run(ignore_exception=True)
    else:
        cestbc = fsl.FAST()
        cestbc.inputs.in_files = str(regdir / "CEST_ref.nii.gz")
        cestbc.inputs.out_basename = str(regdir / "CEST_bc")
        cestbc.inputs.output_biascorrected = True
        cestbc.inputs.output_biasfield = True
        cestbc.inputs.no_pve = True
        cestbc.inputs.output_type = "NIFTI_GZ"
        cestbc.run(ignore_exception=True)


    # Skull strip Reference and CEST Images
    if phantom:
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / "Ref_sROI.nii.gz")
        fmaths.inputs.out_file = str(regdir / "Ref_brain.nii.gz")
        fmaths.inputs.op_string = "-thrp 10"
        fmaths.run()

        fmaths.inputs.in_file = str(regdir / "Ref_brain.nii.gz")
        fmaths.inputs.out_file = str(regdir / "Ref_brain_mask.nii.gz")
        fmaths.inputs.op_string = "-bin"
        fmaths.run()

        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / "CEST_bc_restore.nii.gz")
        fmaths.inputs.out_file = str(regdir / "CEST_brain.nii.gz")
        fmaths.inputs.op_string = "-thrp 10"
        fmaths.run()
    else:
        btr = fsl.BET()
        btr.inputs.in_file = str(regdir / "Ref_sROI.nii.gz")
        btr.inputs.out_file = str(regdir / "Ref_brain.nii.gz")
        btr.inputs.frac = 0.4
        if dim3 == 1:
            btr.inputs.frac = 0.2
        btr.inputs.padding = True
        btr.inputs.mask = True
        btr.run()

        btr = fsl.BET()
        btr.inputs.in_file = str(regdir / "CEST_bc_restore.nii.gz")
        btr.inputs.out_file = str(regdir / "CEST_brain.nii.gz")
        if (
            cest_img.ndim <= 2
            or cest_img.shape[2] < 5
            or cest_img.header.get_zooms()[2] * cest_img.shape[2] < 25
        ):
            btr.inputs.padding = True
        btr.run()

    # Flirt BETted image into CEST Space
    flt = fsl.FLIRT()
    flt.inputs.in_file = str(regdir / "Ref_brain.nii.gz")
    flt.inputs.out_file = str(regdir / refresize)
    flt.inputs.reference = str(regdir / "CEST_brain.nii.gz")
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.out_matrix_file = str(regdir / "Ref_CESTres.txt")
    if dim3 == 1:
        flt.inputs.rigid2D = True
    flt.run()

    flt.inputs.in_file = str(regdir / "Ref_brain_mask.nii.gz")
    flt.inputs.out_file = str(regdir / "Ref_Mask.nii.gz")
    flt.inputs.in_matrix_file = str(regdir / "Ref_CESTres.txt")
    flt.inputs.apply_xfm = True
    flt.run()

    if outfolder is None:
        shutil.copyfile(str(regdir / refresize), refresize)
        shutil.copyfile(str(regdir / "Ref_Mask.nii.gz"), "Ref_Mask.nii.gz")
    else:
        shutil.copyfile(str(regdir / refresize), str(outfolder / refresize))
        shutil.copyfile(
            str(regdir / "Ref_Mask.nii.gz"), str(outfolder / "Ref_Mask.nii.gz")
        )

    return


def _b1resize(
    datapath,
    b1name,
    regdir,
    b1FAmap=None,
    outfolder=Path.cwd(),
    inrefname=None,
    outrefname="Ref_CESTres.nii.gz",
    phantom=False,
):
    """B1_RESIZE - Preprocesses B1 Data for Use with FABBER
    Takes the raw B1 data and processes it into a B1 map.  Also flirts it
    into the reference volume space.  Does not apply any actual registration,
    just resamples it so it is in the same resolution as the Reference data.

    Parameters:
    -----------
    datapath : pathlib Path object
        The path to the datafolder
    b1name : str
        The name of the B1+ scan to register.
    regdir : pathlib Path object
        The registration directory being used for all 
        of the registration computations.
    b1FAmap : str
        The name of the B1+ FA map to register. 
        If None, will assume this is in the b1name file.
    outfolder : pathlib Path object
        The output directory of the data being analyzed. 
        If blank, the datapath directory will be used.
    refname : str
        The name of the reference volume to register b1 to. 
        If blank, the file Ref_CESTres.nii.gz will be used.

    Returns:
    --------
        None

    Author:  asmith
    Version: 1.0
    Changelog:

    20181217 - initial creation
    """
    b1dir = sorted(datapath.glob("*{0}*.nii.gz".format(b1name)))

    if len(b1dir) > 1 and b1FAmap is None:
        fslmerge = fsl.Merge()
        fslmerge.inputs.in_files = [str(i) for v, i in enumerate(b1dir)]
        fslmerge.inputs.dimension = "t"
        fslmerge.inputs.merged_file = str(regdir / f"{b1name}_merged.nii.gz")
        fslmerge.run()

        b1dir = regdir / f"{b1name}_merged.nii.gz"

    try:
        b1vol = nib.load(str(b1dir[1]))
    except TypeError:
        b1vol = nib.load(str(b1dir))
    except IndexError:
        b1vol = nib.load(str(b1dir[0]))
    except:
        print(f"Unexpected Error: {sys.exc_info()[0]}")

    if b1FAmap is None and b1vol.ndim < 4:
        if len(b1dir) > 1:
            b1FAmap = Path(b1dir[-1].stem).stem
        else:
            raise NoFAMapError(
                "No FA Map specified for B1 Data!\nProvide a FA map to proceed!"
            )
    if b1FAmap is None and b1vol.shape[3] > 2:
        # Split Data
        fsplt = fsl.Split()
        fsplt.inputs.in_file = str(b1dir)
        fsplt.inputs.out_base_name = str(regdir / "DREAM_s")
        fsplt.inputs.dimension = "t"
        fsplt.run()

        # set variables so anatomical and FA map are defined
        b1dirs = sorted(regdir.glob("*DREAM_s*.nii.gz"))
        b1dir = str(b1dirs[1])
        b1FAmapdir = str(b1dirs[-1])

    elif b1FAmap is None and b1vol.shape[3] == 2:
        # Split Data
        fsplt = fsl.Split()
        fsplt.inputs.in_file = str(b1dir)
        fsplt.inputs.out_base_name = str(regdir / "DREAM_s")
        fsplt.inputs.dimension = "t"
        fsplt.run()

        # Split b1vol into component anatomicals for registration
        b1dirs = sorted(regdir.glob("*DREAM_s*.nii.gz"))
        b1dir = str(b1dirs[1])
        # Define b1FAmap for use in downstream registration
        b1FAmapdir = regdir / "B1map.nii.gz"

        # Build FA map from anatomical maps
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(b1dirs[0])
        fmaths.inputs.op_string = "-mul 2 -div"
        fmaths.inputs.in_file2 = str(b1dirs[1])
        fmaths.inputs.out_file = str(regdir / "tmp1.nii.gz")
        fmaths.run()
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / "tmp1.nii.gz")
        fmaths.inputs.op_string = f"-sqrt -atan -div {radians(60)} -mul 600"
        fmaths.inputs.out_file = str(b1FAmapdir)
        fmaths.run()
    else:
        b1FAmapdir = sorted(datapath.glob(f"*{b1FAmap}*.nii.gz"))[0]
        try:
            b1dir = str(b1dir[1])
        except IndexError:
            b1dir = str(b1dir[0])
    
    b1splt = fsl.ExtractROI()
    b1splt.inputs.in_file = b1dir
    b1splt.inputs.roi_file = str(regdir / "B1_1.nii.gz")
    b1splt.inputs.t_size = 1
    b1splt.inputs.t_min = 0
    b1splt.run()

    # Run FAST on B1 input data to get better registration
    b1FAST = fsl.FAST()
    b1FAST.inputs.in_files = str(regdir / "B1_1.nii.gz")
    b1FAST.inputs.out_basename = str(regdir / "B1_bc")
    b1FAST.inputs.output_biascorrected = True
    b1FAST.inputs.no_pve = True
    b1FAST.inputs.output_type = "NIFTI_GZ"
    b1FAST.run(ignore_exception=True)

    if phantom:
        # _FOVDiff(str(regdir / "B1_bc_restore.nii.gz"), str(inrefdir), "B1resred.txt", regdir=regdir)

        # Flirt B1 Image Data to Original Reference volume
        flt = fsl.FLIRT()
        flt.inputs.in_file = str(regdir / "B1_bc_restore.nii.gz")
        flt.inputs.reference = str(regdir / "Ref_bc_restore.nii.gz")
        flt.inputs.out_file = str(regdir / "B1_to_ref.nii.gz")
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.rigid2D = True
        # flt.inputs.in_matrix_file = str(regdir / "B1resred.txt")

        # flt.inputs.apply_xfm=True
        flt.inputs.out_matrix_file = str(regdir / "B1resred.txt")
        flt.run()
    else:
        # Flirt B1 Image Data to Original Reference volume
        flt = fsl.FLIRT()
        flt.inputs.in_file = str(regdir / "B1_bc_restore.nii.gz")
        flt.inputs.reference = str(regdir / "Ref_bc_restore.nii.gz")
        flt.inputs.out_file = str(regdir / "B1_to_ref.nii.gz")
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.out_matrix_file = str(regdir / "B1resred.txt")
        flt.run()

    if len(_slicenumber2d) > 0:
        # Flirt B1 Map Data to Original Reference volume
        flt = fsl.FLIRT()
        flt.inputs.in_file = str(b1FAmapdir)
        flt.inputs.reference = str(regdir / "Ref_bc_restore.nii.gz")
        flt.inputs.out_file = str(regdir / "B1map_to_ref.nii.gz")
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.in_matrix_file = str(regdir / "B1resred.txt")
        flt.inputs.apply_xfm = True
        flt.inputs.out_matrix_file = str(regdir / "B1resred.txt")
        flt.run()

        if _slicenumber2d[1] > 1:
            # Run for B1 Map
            fsl.ExtractROI(
                in_file=str(regdir / "B1map_to_ref.nii.gz"),
                roi_file=str(regdir / "B1map_sROI.nii.gz"),
                x_min=0,
                x_size=-1,
                y_min=0,
                y_size=-1,
                z_min=_slicenumber2d[0] - 1,
                z_size=_slicenumber2d[1],
            ).run()
        else:
            # Run for B1 Map
            fsl.ExtractROI(
                in_file=str(regdir / "B1map_to_ref.nii.gz"),
                roi_file=str(regdir / "B1map_sROI.nii.gz"),
                x_min=0,
                x_size=-1,
                y_min=0,
                y_size=-1,
                z_min=_slicenumber2d[0],
                z_size=_slicenumber2d[1],
            ).run()

        # Warp 2D B1map to CEST Space
        flt.inputs.in_file = str(regdir / "B1map_sROI.nii.gz")
        flt.inputs.out_file = str(regdir / "B1map_resred.nii.gz")
        flt.inputs.reference = str(regdir / outrefname)
        flt.inputs.in_matrix_file = str(regdir / "Ref_CESTres.txt")
        flt.inputs.apply_xfm = True
        flt.inputs.rigid2D = True
        flt.inputs.out_matrix_file = str(regdir / "B1toCEST.txt")
        flt.run()

    else:
        # Combine B1->Ref matrix with Ref->CEST matrix
        xfmcomb = fsl.ConvertXFM()
        xfmcomb.inputs.in_file = str(regdir / "B1resred.txt")
        xfmcomb.inputs.in_file2 = str(regdir / "Ref_CESTres.txt")
        xfmcomb.inputs.concat_xfm = True
        xfmcomb.inputs.out_file = str(regdir / "B1toCEST.txt")
        xfmcomb.run()

        # Use combine Ref->CEST matrix to register B1 FA map to CEST data
        flt.inputs.in_file = str(b1FAmapdir)
        flt.inputs.out_file = str(regdir / "B1map_resred.nii.gz")
        flt.inputs.reference = str(regdir / outrefname)
        flt.inputs.in_matrix_file = str(regdir / "B1toCEST.txt")
        flt.inputs.apply_xfm = True
        flt.inputs.out_matrix_file = str(regdir / "B1toCEST.txt")
        flt.run()

    # Convert Registered FA Map to Fraction of nominal angle and move to analysis folder
    fmaths = fsl.ImageMaths()
    fmaths.inputs.in_file = str(regdir / "B1map_resred.nii.gz")
    fmaths.inputs.op_string = "-div 600 -mul"
    fmaths.inputs.in_file2 = str(regdir / "Ref_Mask.nii.gz")
    fmaths.inputs.out_file = str(outfolder / "B1map_resize.nii.gz")
    fmaths.run()

    return None


def _cestreg(
    datapath,
    cestprefix,
    offsetspath,
    regdir,
    outfolder=Path.cwd(),
    CESTRefImage=0,
    phantom=False,
):
    """ _cestreg7T - Co-registers CEST data, and then registers that data
    to the high resolution reference measurement set up in _reg_ref_7T
    """
    cestdir = sorted(datapath.glob(f"*{cestprefix}*.nii.gz"))

    offsets = np.loadtxt(str(offsetspath))

    n_ones = np.where(abs(offsets) < 1)[0]

    cestoutname = cestprefix

    cestvol = nib.load(str(cestdir[0]))

    if cestvol.ndim == 4:
        fslsplit = fsl.Split()
        fslsplit.inputs.in_file = str(cestdir[0])
        fslsplit.inputs.out_base_name = str(regdir / f"{cestoutname}_Presplit")
        fslsplit.inputs.output_type = "NIFTI_GZ"
        fslsplit.inputs.dimension = "t"
        fslsplit.run()
        cestdir = sorted((regdir.glob(f"*{cestoutname}_Presplit*.nii.gz")))

    concatname1 = [str(i) for v, i in enumerate(cestdir) if v < n_ones[1]]
    concatname2 = [str(i) for v, i in enumerate(cestdir) if v > n_ones[-1]]
    concatname2.append(str(cestdir[CESTRefImage]))

    # Merge first half of files
    fslmerge = fsl.Merge()
    fslmerge.inputs.in_files = concatname1
    fslmerge.inputs.dimension = "t"
    fslmerge.inputs.merged_file = str(regdir / f"{cestoutname}_merged1.nii.gz")
    fslmerge.run()

    # Merge second half of files
    fslmerge.inputs.in_files = concatname2
    fslmerge.inputs.merged_file = str(regdir / f"{cestoutname}_merged2.nii.gz")
    fslmerge.run()

    # Motion-Correct Data
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(regdir / f"{cestoutname}_merged1.nii.gz")
    mcflirt.inputs.ref_vol = CESTRefImage
    mcflirt.inputs.out_file = str(regdir / f"{cestoutname}_merged1_mcf.nii.gz")
    if cestvol.ndim < 3:
        mcflirt.inputs.args = "-2d"
    mcflirt.run()

    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(regdir / f"{cestoutname}_merged2.nii.gz")
    mcflirt.inputs.ref_vol = len(concatname2) - 1
    mcflirt.inputs.out_file = str(regdir / f"{cestoutname}_merged2_mcf.nii.gz")
    if cestvol.ndim < 3:
        mcflirt.inputs.args = "-2d"
    mcflirt.run()

    # Extract one datapoint near middle of CEST
    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = str(regdir / f"{cestoutname}_merged1_mcf.nii.gz")
    fslroi.inputs.roi_file = str(regdir / f"{cestoutname}_mcfMin.nii.gz")
    fslroi.inputs.t_min = len(concatname1) - 1
    fslroi.inputs.t_size = 1
    fslroi.run()

    # Remove extra reference image in merged2
    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = str(regdir / f"{cestoutname}_merged2_mcf.nii.gz")
    fslroi.inputs.roi_file = str(regdir / f"{cestoutname}_merged2_mcf.nii.gz")
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = len(concatname2) - 1
    fslroi.run()

    concatname3 = [str(i) for v, i in enumerate(cestdir) if v in n_ones[1:]]
    concatname3.insert(0, str(regdir / f"{cestoutname}_mcfMin.nii.gz"))

    fslmerge.inputs.in_files = concatname3
    fslmerge.inputs.merged_file = str(regdir / f"{cestoutname}_merged3.nii.gz")
    fslmerge.run()

    # MCFLIRT middle range (e.g. where abs(offsets) < 1)
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(regdir / f"{cestoutname}_merged3.nii.gz")
    mcflirt.inputs.ref_vol = 0
    mcflirt.inputs.out_file = str(regdir / f"{cestoutname}_merged3_mcf.nii.gz")
    mcflirt.inputs.cost = "normmi"
    if cestvol.ndim < 3:
        mcflirt.inputs.args = "-2d"
    mcflirt.run()

    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = str(regdir / f"{cestoutname}_merged3_mcf.nii.gz")
    fslroi.inputs.roi_file = str(regdir / f"{cestoutname}_mcfMid.nii.gz")
    fslroi.inputs.t_min = 1
    fslroi.inputs.t_size = len(n_ones) - 1
    fslroi.run()

    # Merge all CEST images together
    fslmerge = fsl.Merge()
    fslmerge.inputs.in_files = [
        str(regdir / f"{cestoutname}_merged1_mcf.nii.gz"),
        str(regdir / f"{cestoutname}_mcfMid.nii.gz"),
        str(regdir / f"{cestoutname}_merged2_mcf.nii.gz"),
    ]
    fslmerge.inputs.dimension = "t"
    fslmerge.inputs.merged_file = str(regdir / f"{cestoutname}_mergedTot_mcf.nii.gz")
    fslmerge.run()

    # Separate CEST reference image to use as registration template for CEST data
    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = str(regdir / f"{cestoutname}_mergedTot_mcf.nii.gz")
    fslroi.inputs.roi_file = str(regdir / f"{cestoutname}_prereg.nii.gz")
    fslroi.inputs.t_min = CESTRefImage
    fslroi.inputs.t_size = 1
    fslroi.run()

    # Run FAST on Ref image
    cestfast = fsl.FAST()
    cestfast.inputs.in_files = str(regdir / f"{cestoutname}_prereg.nii.gz")
    cestfast.inputs.out_basename = str(regdir / f"{cestoutname}_bc")
    cestfast.inputs.output_biascorrected = True
    cestfast.inputs.output_biasfield = True
    cestfast.inputs.no_pve = True
    cestfast.run(ignore_exception=True)

    # Skull Strip CEST Image
    if phantom:
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / f"{cestoutname}_bc_restore.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{cestoutname}_prereg_brain.nii.gz")
        fmaths.inputs.op_string = "-thrp 10"
        fmaths.run()

        fmaths.inputs.in_file = str(regdir / f"{cestoutname}_prereg_brain.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{cestoutname}_prereg_brain_mask.nii.gz")
        fmaths.inputs.op_string = "-bin"
        fmaths.run()

    else:
        betcest0 = fsl.BET()
        betcest0.inputs.in_file = str(regdir / f"{cestoutname}_bc_restore.nii.gz")
        betcest0.inputs.out_file = str(regdir / f"{cestoutname}_prereg_brain.nii.gz")
        betcest0.inputs.mask = True
        if cestvol.ndim <= 2 or cestvol.shape[3] < 5 or cestvol.header.get_zooms()[2] * cestvol.shape[2] < 25:
            betcest0.inputs.padding = True
        betcest0.run()

    # BET rest of data
    fmaths = fsl.ImageMaths()
    fmaths.inputs.in_file = str(regdir / f"{cestoutname}_mergedTot_mcf.nii.gz")
    fmaths.inputs.op_string = "-mul"
    fmaths.inputs.in_file2 = str(regdir / f"{cestoutname}_prereg_brain_mask.nii.gz")
    fmaths.inputs.out_file = str(regdir / f"{cestoutname}_mergedTot_bc_brain.nii.gz")
    fmaths.run()

    # Move CEST data to analysis folder
    shutil.copyfile(
        str(regdir / f"{cestoutname}_mergedTot_bc_brain.nii.gz"),
        str(outfolder / f"{cestoutname}_reg.nii.gz"),
    )

    # Correct Mask so it is set at extent of CEST data
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = str(regdir / "Ref_Mask.nii.gz")
    fslmaths.inputs.op_string = "-thr 0.95 -mul"
    fslmaths.inputs.in_file2 = str(outfolder / f"{cestoutname}_reg.nii.gz")
    fslmaths.inputs.args = "-bin"
    fslmaths.inputs.out_file = str(regdir / "Ref_Mask.nii.gz")
    fslmaths.run()

    # Copy Corrected mask into Analysis folder
    shutil.copyfile(str(regdir / "Ref_Mask.nii.gz"), str(outfolder / "Ref_Mask.nii.gz"))

    # Change mask to only a single volume (instead of 4D volume from CEST data)
    maskroi = fsl.ExtractROI()
    maskroi.inputs.in_file = str(outfolder / "Ref_Mask.nii.gz")
    maskroi.inputs.roi_file = str(outfolder / "Ref_Mask.nii.gz")
    maskroi.inputs.t_min = 0
    maskroi.inputs.t_size = 1
    maskroi.run()

    return None


def _t1reg(
    datapath,
    t1prefix,
    regdir,
    outfolder=Path.cwd(),
    inrefname=None,
    outrefname="Ref_CESTres.nii.gz",
    phantom=False,
):
    """ _t1reg - Co-registers T1 data, and then registers that data
    to the high resolution reference measurement set up in _reg_ref_7T/_reg_ref
    """
    t1dir = sorted(datapath.glob(f"*{t1prefix}*.nii.gz"))
    if t1dir is None:
        print("No T1 Files Detected!")
        return None

    if inrefname is None:
        inrefdir = datapath / outrefname
    else:
        inrefdir = sorted(datapath.glob(f"*{inrefname}*.nii.gz"))[0]

    t1outname = t1prefix

    # Merge all T1 vols together
    if len(t1dir) > 1:
        concatname = [str(i) for i in t1dir]

        # Merge all files
        fslmerge = fsl.Merge()
        fslmerge.inputs.in_files = concatname
        fslmerge.inputs.dimension = "t"
        fslmerge.inputs.merged_file = str(regdir / f"{t1outname}_merged.nii.gz")
        fslmerge.run()
    else:
        shutil.copyfile(str(t1dir[0]), str(regdir / f"{t1outname}_merged.nii.gz"))

    #  MC Flirt Data
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(regdir / f"{t1outname}_merged.nii.gz")
    mcflirt.inputs.ref_vol = 0
    mcflirt.inputs.out_file = str(regdir / f"{t1outname}_merged_mcf.nii.gz")
    mcflirt.run()

    # Extract an ROI to use for registration to Ref
    fsl.ExtractROI(
        in_file=str(regdir / f"{t1outname}_merged_mcf.nii.gz"),
        roi_file=str(regdir / f"{t1outname}_vol1.nii.gz"),
        t_min=0,
        t_size=1,
    ).run()

    # Run FAST on Ref T1 image
    t1fast = fsl.FAST()
    t1fast.inputs.in_files = str(regdir / f"{t1outname}_vol1.nii.gz")
    t1fast.inputs.out_basename = str(regdir / f"{t1outname}_bc")
    t1fast.inputs.output_biascorrected = True
    t1fast.inputs.output_biasfield = True
    t1fast.inputs.no_pve = True
    t1fast.run(ignore_exception=True)

    t1vol = nib.load(str(t1dir[0]))

    # BET T1 Ref image
    if phantom:
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / f"{t1outname}_vol1.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{t1outname}_brain.nii.gz")
        fmaths.inputs.op_string = "-thrp 10"
        fmaths.run()

        fmaths.inputs.in_file = str(regdir / f"{t1outname}_brain.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{t1outname}_brain_mask.nii.gz")
        fmaths.inputs.op_string = "-bin"
        fmaths.run()

        if t1vol.ndim < 3 or t1vol.shape[2] == 1:
            # Flirt first T1 Image to Original Reference volume
            flt = fsl.FLIRT()
            flt.inputs.in_file = str(regdir / f"{t1outname}_vol1.nii.gz")
            flt.inputs.reference = str(regdir / "Ref_sROI.nii.gz")
            flt.inputs.out_file = str(regdir / "T1_to_ref.nii.gz")
            flt.inputs.rigid2D = True
            flt.inputs.output_type = "NIFTI_GZ"
            flt.inputs.out_matrix_file = str(regdir / "T1resred.txt")
            flt.run()
        else:
            # Flirt first T1 Image to Original Reference volume
            flt = fsl.FLIRT()
            flt.inputs.in_file = str(regdir / f"{t1outname}_vol1.nii.gz")
            flt.inputs.reference = str(inrefdir)
            flt.inputs.out_file = str(regdir / "T1_to_ref.nii.gz")
            flt.inputs.rigid2D = True
            flt.inputs.output_type = "NIFTI_GZ"
            flt.inputs.out_matrix_file = str(regdir / "T1resred.txt")
            flt.run()
    else:
        bett1 = fsl.BET()
        bett1.inputs.in_file = str(regdir / f"{t1outname}_vol1.nii.gz")
        bett1.inputs.out_file = str(regdir / f"{t1outname}_brain.nii.gz")
        bett1.inputs.mask = True
        bett1.inputs.padding = True
        bett1.run()

        # Flirt first T1 Image to Original Reference volume
        flt = fsl.FLIRT()
        flt.inputs.in_file = str(regdir / f"{t1outname}_vol1.nii.gz")
        flt.inputs.reference = str(inrefdir)
        flt.inputs.out_file = str(regdir / "T1_to_ref.nii.gz")
        flt.inputs.output_type = "NIFTI_GZ"
        flt.inputs.out_matrix_file = str(regdir / "T1resred.txt")
        flt.run()

    # BET rest of data
    fmaths = fsl.ImageMaths()
    fmaths.inputs.in_file = str(regdir / f"{t1outname}_merged_mcf.nii.gz")
    fmaths.inputs.op_string = "-mul"
    fmaths.inputs.in_file2 = str(regdir / f"{t1outname}_brain_mask.nii.gz")
    fmaths.inputs.out_file = str(regdir / f"{t1outname}_merged_brain.nii.gz")
    fmaths.run()

    if len(_slicenumber2d) > 0:
        # Flirt T1 Map Data to Original Reference volume
        if t1vol.ndim < 3 or t1vol.shape[2] == 1:
            flt = fsl.FLIRT()
            flt.inputs.in_file = str(regdir / f"{t1outname}_merged_brain.nii.gz")
            flt.inputs.reference = str(regdir / "Ref_sROI.nii.gz")
            flt.inputs.out_file = str(regdir / "T1Data_sROI.nii.gz")
            if phantom:
                flt.inputs.rigid2D = True
            flt.inputs.output_type = "NIFTI_GZ"
            flt.inputs.in_matrix_file = str(regdir / "T1resred.txt")
            flt.inputs.apply_xfm = True
            flt.inputs.out_matrix_file = str(regdir / "T1resred.txt")
            flt.run()
        else:
            flt = fsl.FLIRT()
            flt.inputs.in_file = str(regdir / f"{t1outname}_merged_brain.nii.gz")
            flt.inputs.reference = str(inrefdir)
            flt.inputs.out_file = str(regdir / "T1Data_to_ref.nii.gz")
            flt.inputs.output_type = "NIFTI_GZ"
            flt.inputs.in_matrix_file = str(regdir / "T1resred.txt")
            flt.inputs.apply_xfm = True
            flt.inputs.out_matrix_file = str(regdir / "T1resred.txt")
            flt.run()

            if _slicenumber2d[1] > 1:
                # Run for T1 Data
                fsl.ExtractROI(
                    in_file=str(regdir / "T1Data_to_ref.nii.gz"),
                    roi_file=str(regdir / "T1Data_sROI.nii.gz"),
                    x_min=0,
                    x_size=-1,
                    y_min=0,
                    y_size=-1,
                    z_min=_slicenumber2d[0] - 1,
                    z_size=_slicenumber2d[1],
                ).run()
            else:
                # Run for T1 Data
                fsl.ExtractROI(
                    in_file=str(regdir / "T1Data_to_ref.nii.gz"),
                    roi_file=str(regdir / "T1Data_sROI.nii.gz"),
                    x_min=0,
                    x_size=-1,
                    y_min=0,
                    y_size=-1,
                    z_min=_slicenumber2d[0],
                    z_size=_slicenumber2d[1],
                ).run()

        # Warp 2D T1 Data to CEST Space
        flt.inputs.in_file = str(regdir / "T1Data_sROI.nii.gz")
        flt.inputs.out_file = str(regdir / "T1Data_resred.nii.gz")
        flt.inputs.reference = str(regdir / outrefname)
        flt.inputs.in_matrix_file = str(regdir / "Ref_CESTres.txt")
        flt.inputs.apply_xfm = True
        flt.inputs.rigid2D = True
        flt.inputs.out_matrix_file = str(regdir / "T1toCEST.txt")
        flt.run()

    else:
        # Combine T1->Ref matrix with Ref->CEST matrix
        xfmcomb = fsl.ConvertXFM()
        xfmcomb.inputs.in_file = str(regdir / "T1resred.txt")
        xfmcomb.inputs.in_file2 = str(regdir / "Ref_CESTres.txt")
        xfmcomb.inputs.concat_xfm = True
        xfmcomb.inputs.out_file = str(regdir / "T1toCEST.txt")
        xfmcomb.run()

        # Use combine Ref->CEST matrix to register T1 data to CEST data
        flt.inputs.in_file = str(regdir / f"{t1outname}_merged_brain.nii.gz")
        flt.inputs.out_file = str(regdir / "T1Data_resred.nii.gz")
        flt.inputs.reference = str(regdir / outrefname)
        flt.inputs.in_matrix_file = str(regdir / "T1toCEST.txt")
        flt.inputs.apply_xfm = True
        flt.inputs.out_matrix_file = str(regdir / "T1toCEST.txt")
        flt.run()

    shutil.copyfile(
        str(regdir / "T1Data_resred.nii.gz"), str(outfolder / f"{t1outname}_reg.nii.gz")
    )

    return None


def _mtreg(
    datapath,
    mtprefix,
    offsetspath,
    regdir,
    outfolder=Path.cwd(),
    MTRefImage=-1,
    phantom=False,
):
    """ _mtreg - Co-registers qMT data, and then registers that data
    to the high resolution reference measurement set up in _reg_ref
    """
    mtdir = sorted(datapath.glob(f"*{mtprefix}*.nii.gz"))

    offsets = np.loadtxt(str(offsetspath))

    if MTRefImage == -1:
        MTRefImage = len(offsets) - 1

    mtoutname = mtprefix

    mtvol = nib.load(str(mtdir[0]))

    if mtvol.ndim != 4 or mtvol.shape[3] != offsets.shape[0]:
        concatname = [str(i) for i in mtdir]
        fslmerge = fsl.Merge()
        fslmerge.inputs.in_files = concatname
        fslmerge.inputs.dimension = "t"
        fslmerge.inputs.merged_file = str(regdir / f"{mtoutname}_merged.nii.gz")
        fslmerge.run()
        mtdir = regdir / f"{mtoutname}_merged.nii.gz"
    else:
        mtdir = mtdir[0]

    # Motion-Correct Data
    mcflirt = fsl.MCFLIRT()
    mcflirt.inputs.in_file = str(mtdir)
    mcflirt.inputs.mean_vol = True
    mcflirt.inputs.stages = 4
    mcflirt.inputs.out_file = str(regdir / f"{mtoutname}_merged_mcf.nii.gz")
    if mtvol.ndim < 3 or mtvol.shape[2] == 1:
        mcflirt.inputs.args = "-2d"
    mcflirt.run()

    # Separate MT reference image to use as registration template for MT data
    fslroi = fsl.ExtractROI()
    fslroi.inputs.in_file = str(regdir / f"{mtoutname}_merged_mcf.nii.gz")
    fslroi.inputs.roi_file = str(regdir / f"{mtoutname}_prereg.nii.gz")
    fslroi.inputs.t_min = MTRefImage
    fslroi.inputs.t_size = 1
    fslroi.run()

    # Run FAST on Ref image
    # cestfast = fsl.FAST()
    # cestfast.inputs.in_files = str(regdir / f"{mtoutname}_prereg.nii.gz")
    # cestfast.inputs.out_basename = str(regdir / f"{mtoutname}_bc")
    # cestfast.inputs.output_biascorrected = True
    # cestfast.inputs.output_biasfield = True
    # cestfast.inputs.no_pve = True
    # cestfast.run(ignore_exception=True)

    # Skull Strip MT Image
    if phantom:
        fmaths = fsl.ImageMaths()
        fmaths.inputs.in_file = str(regdir / f"{mtoutname}_bc_restore.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{mtoutname}_prereg_brain.nii.gz")
        fmaths.inputs.op_string = "-thrp 10"
        fmaths.run()

        fmaths.inputs.in_file = str(regdir / f"{mtoutname}_prereg_brain.nii.gz")
        fmaths.inputs.out_file = str(regdir / f"{mtoutname}_prereg_brain_mask.nii.gz")
        fmaths.inputs.op_string = "-bin"
        fmaths.run()

    else:
        betcest0 = fsl.BET()
        betcest0.inputs.in_file = str(regdir / f"{mtoutname}_prereg.nii.gz")
        # betcest0.inputs.in_file = str(regdir / f"{mtoutname}_bc_restore.nii.gz")
        betcest0.inputs.out_file = str(regdir / f"{mtoutname}_prereg_brain.nii.gz")
        betcest0.inputs.mask = True
        betcest0.inputs.padding = True
        betcest0.run()

    # Skull strip the rest of the data
    fmaths = fsl.ImageMaths()
    fmaths.inputs.in_file = str(regdir / f"{mtoutname}_merged_mcf.nii.gz")
    fmaths.inputs.op_string = "-mul"
    fmaths.inputs.in_file2 = str(regdir / f"{mtoutname}_prereg_brain_mask.nii.gz")
    fmaths.inputs.out_file = str(regdir / f"{mtoutname}_merged_bc_brain.nii.gz")
    fmaths.run()

    shutil.copyfile(
        str(regdir / f"{mtoutname}_merged_bc_brain.nii.gz"),
        str(outfolder / f"{mtoutname}_reg.nii.gz"),
    )

    # Correct Mask so it is set at extent of MT data
    fslmaths = fsl.ImageMaths()
    fslmaths.inputs.in_file = str(regdir / "Ref_Mask.nii.gz")
    fslmaths.inputs.op_string = "-thr 0.95 -mul"
    fslmaths.inputs.in_file2 = str(outfolder / f"{mtoutname}_reg.nii.gz")
    fslmaths.inputs.args = "-bin"
    fslmaths.inputs.out_file = str(regdir / "Ref_Mask.nii.gz")

    # Copy Corrected mask into Analysis folder
    shutil.copyfile(str(regdir / "Ref_Mask.nii.gz"), str(outfolder / "Ref_Mask.nii.gz"))

    return None


# Helper Functions:
def _FOVDiff(inputvol, refvol, outfile, regdir=Path.cwd()):
    """ _FOVDiff - Creates Resizing matrix for FLIRT to resolve small Z-FOV images
    Function will create a text file to use with FLIRT in the event that no full FOV
    image is collected to use as a reference with FLIRT.
    """

    input_img = nib.load(inputvol)
    ref_img = nib.load(refvol)

    # Find reference dimensions
    refdims = ref_img.header["dim"][1:4]
    refpixdims = ref_img.header["pixdim"][1:4]

    # Find input image dimensions
    inputdims = input_img.header["dim"][1:4]
    inputpixdims = input_img.header["pixdim"][1:4]

    refFOV = np.asarray(refdims) * np.asarray(refpixdims)
    inputFOV = np.asarray(inputdims) * np.asarray(inputpixdims)

    # Create resizing Identity Matrix
    with (Path(regdir) / outfile).open(mode="w") as FID:
        FID.write(
            "1 0 0 {0}\n0 1 0 {1}\n0 0 1 {2}\n0 0 0 1".format(
                (refFOV[0] - inputFOV[0]) / 2,
                (refFOV[1] - inputFOV[1]) / 2,
                (refFOV[2] - inputFOV[2]) / 2,
            )
        )

    return None


def _genidentitymat(regdir=Path.cwd()):
    """ _genidentitymat generates an identity matrix
    """
    with (Path(regdir) / "Identity.txt").open(mode="w") as FID:
        FID.write("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1")

    return None


class NoFAMapError(Exception):
    pass


class NoCESTDataError(Exception):
    pass


class JsonKeyError(Exception):
    pass


json_keys = [
    "Analysis Path",
    "Data Path",
    "Scan ID",
    "Reference Name",
    "CEST Names",
    "T1 Name",
    "B1 Anatomical Name",
    "B1 FA Name",
    "Offset File Names",
]


def create_json_example_file():
    json_dict = {i: "FileName" for i in json_keys}
    json_dict["Analysis Path"] = "/Path/To/Analysis/Folder"
    json_dict["Data Path"] = "/Path/To/Scans/Folder"
    json_dict["Scan ID"] = "/Name/of/Scan/Folder(s)"
    with open("json_file_example.json", "w") as FID:
        json.dump(json_dict, FID)
    return None


def json_file_parser(jsonfile):
    with open(jsonfile, "r") as FID:
        jsondata = json.load(FID)

    return json_to_dict(jsondata)


def json_to_dict(jsondata):
    if set(("CEST Names", "MT Names")).issubset(jsondata):
        raise JsonKeyError(
            "Cannot use both CEST and MT keys together!\nIf in doubt, place all names into 'CEST Names' key!"
        )
    elif not any(x in jsondata.keys() for x in ["CEST Names", "MT Names"]):
        raise JsonKeyError("Need to have either 'CEST Names' or 'MT Names' keys!")
    else:
        if "CEST Names" not in jsondata.keys():
            jsondata["CEST Names"] = jsondata.pop("MT Names")

    for key in json_keys:
        if key not in jsondata.keys():
            if key == "B1 FA Name":
                print(
                    "Assuming B1 Anatomical Images have sufficient data to calculate B1 map!"
                )
                jsondata[key] = None
                continue
            else:
                raise JsonKeyError(f"Missing: '{key}' from json file")
        
        

        if not isinstance(jsondata[key], list):
            jsondata[key] = jsondata[key].split(".")[0]

        if key in ["Analysis Path", "Data Path"]:
            jsondata[key] = Path(jsondata[key])
        elif key in ["CEST Names", "Scan ID", "Offset File Names"] and not isinstance(jsondata[key],list):
            jsondata[key] = [jsondata[key]]
        
        
        if key == "Offset File Names":
            for idx, offset in enumerate(jsondata[key]):
                if offset.split('.')[-1] != "txt":
                    jsondata[key][idx] = f"{offset}.txt"

    return jsondata


def copy_offsets(src, dest, filename):
    # Copy offsets file into DataFolder
    if not (dest / filename).exists():
        shutil.copyfile(src / filename, dest / filename)


def register_json_data(jsondata):
    for _, sn in enumerate(jsondata["Scan ID"]):

        print(f"Running registration for scan: {sn}")
        analysis_folder = jsondata["Analysis Path"] / sn

        for idx, cest_data in enumerate(jsondata["CEST Names"]):
            print(f"Registering {cest_data}")
            dir_test = list(
                (analysis_folder / cest_data).glob("*{0}*.nii.gz".format(cest_data))
            )

            copy_offsets(
                jsondata["Data Path"],
                jsondata["Data Path"] / sn,
                jsondata["Offset File Names"][idx],
            )
            # Create a new folder for Data analysis and run registration for that analysis
            if not dir_test:
                (analysis_folder / cest_data).mkdir(exist_ok=True, parents=True)

                register_cest(
                    PathToData=(jsondata["Data Path"] / sn),
                    RefName=jsondata["Reference Name"],
                    CESTName=cest_data,
                    OffsetsName=jsondata["Offset File Names"][idx],
                    OutFolder=(analysis_folder / cest_data),
                    B1Name=jsondata["B1 Anatomical Name"],
                    b1FAname=jsondata["B1 FA Name"],
                    T1Name=jsondata["T1 Name"],
                    RegDir=f"RegDir_{cest_data}",
                )
            else:
                warnings.warn(f"\nRegistration already peformed in directory:\n\t{analysis_folder / cest_data}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        dest="j_file",
        type=str,
        help="JSON file containing key-value pairs which point to files. See 'Example_json_file.json' for example",
        metavar="<JSON File>",
    )
    try:
        args = parser.parse_args()
    except SystemExit as err:
        create_json_example_file()
        if err.code == 2:
            parser.print_help()
        sys.exit(0)

    jsondata = json_file_parser(args.j_file)

    register_json_data(jsondata)


if __name__ == "__main__":
    main()
