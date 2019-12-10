#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Generate dataspec file
from pathlib import Path
import numpy as np
from CESTClass import CESTModel

import warnings

def create_fabber_ptrain_dataspec(
    offsetsfile,
    specfile=(Path.cwd() / "dataspec.txt"),
    ptrainfile=(Path.cwd() / "ptrain.txt"),
    nCESTp=[50],
    pw=[20e-3],
    dutycycle=[0.5],
    nomB1=[180],
    cwep=True,
    TR = 4,
    thetaEX = 90,
):
    """Generates the ptrain and dataspec files for use in the FABBER CEST module
    
    Args:
        offsetsfile (Path, str, list): Path/str (or list of Paths/str) to a file containing the offsets 
            for a given CEST sequence. There should be a list for each saturation power.
        specfile (Path, optional): Path to where the dataspect file should be generated. 
            Defaults to CWD / "dataspec.txt".
        ptrainfile (Path, optional): Path to where the ptrain file should be generated. 
            Defaults to CWD / "ptrain.txt".
        nCESTp (list, optional): List containing the number of pulses for each sequence. The list should 
            equal the number of offset files provided. Defaults to [50].
        pw (list, optional): List containing the pulse width for each pulse in the CEST pulse train for 
            each sequence. The list should equal the number of offset files provided. Defaults to [20e-3 seconds].
        dutycycle (list, optional): List containing the duty cycle for each sequence. The list should 
            equal the number of offset files provided. Defaults to [0.5].
        nomB1 (list, optional): List containing the nominal B1 for each CEST pulse train. The list should 
            equal the number of offset files provided. Defaults to [180˚].
        cwep (bool, optional): Flag for the Continuous Wave Equivalent Pulse approximation. Defaults to True.
        TR (int, optional): The TR (interval between each CEST train) for the sequences. Defaults to 4 seconds.
        thetaEX (int, optional): The Excitation pulse flip angle. Defaults to 90˚.
    
    """

    try:
        assert all(
            len(lst) == len(nCESTp) for lst in [pw, dutycycle, nomB1]
        ), "Length of input parameters are not the same!"
    except TypeError:
        raise TypeError("TR, nCESTp, pw, dutycycle and nomB1 need to by Type list")

    if isinstance(offsetsfile, list):
        assert len(offsetsfile) == len(
            nCESTp
        ), "Length of offsetsfile needs to match that of other inputs"


    pp = ''

    for ii, bb in enumerate(nomB1):
        if isinstance(offsetsfile, list):
            ppmoffsets = np.loadtxt(str(offsetsfile[ii]))
        elif ii == 0:
            ppmoffsets = np.loadtxt(str(offsetsfile))

        b1 = CESTModel(
            M0=[1, 0.2], T1=[1, 1], T2=[0.6, 0.6], B0=3, Rx=50, ChemShift=[0, 1]
        )
        b1.setsequence(
            ppmvec=ppmoffsets,
            TR=TR,
            thetaEX=thetaEX,
            B1amp=bb,
            DutyCycle=dutycycle[ii],
            nCESTp=nCESTp[ii],
            pwCEST=pw[ii],
            B1Shape="Gauss",
            InterSpoil=False,
        )
    
        if ii == 0:
            with specfile.open(mode="w") as FID:
                for offset in ppmoffsets:
                    if cwep:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}{3}\n".format(
                                offset, b1.B1eCEST[0] * 1e-6, 1.0,pp
                            )
                        )
                    else:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}{3}\n".format(
                                offset, b1.B1amp * 1e-6, nCESTp[ii], pp
                            )
                        )
        else:
            with specfile.open(mode="a") as FID:
                for offset in ppmoffsets:
                    if cwep:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}{3}\n".format(
                                offset, b1.B1eCEST[0] * 1e-6, 1.0, pp
                            )
                        )
                    else:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}{3}\n".format(
                                offset, b1.B1amp * 1e-6, nCESTp[ii], pp
                            )
                        )

    # Create pulse train timing
    with ptrainfile.open(mode="w") as FID:
        if cwep:
            # If CWEP, relative B1 Amplitude is 1, with full train as timing
            FID.write("{0:.6E}\t{1:.6E}\n".format(1.0, b1.B1eTiming))
        else:
            # Use B1 envelope to partition timing
            for idx, b1env in enumerate(b1.B1_Envelope):
                FID.write("{0:.6E}\t{1:.6E}\n".format(b1env, b1.B1_Timing[idx]))

            # Add in extra line for dutycycle
            FID.write("{0:.6E}\t{1:.6E}\n".format(0, b1.B1_Timing[-1] + b1.tMTDC))

    return None


def create_fabbercest_prompt(
    data_stem,
    dataspec_stem,
    ptrain_stem,
    poolmat_stem,
    OutFolder=Path.cwd(),
    maskname="Ref_Mask",
    LineShape=None,
    TR=None,
    thetaEX=None,
    B1Name=None,
    T1Name=None,
):
    """Generates the BASH file to run FABBER CEST for the given dataset.
    
    Args:
        data_stem (str): Name of the NIFTI file holding the CEST data (no extension).
        dataspec_stem (str): Name of the dataspec file (no extension).
        ptrain_stem (str): Name of the ptrain file (no extension).
        poolmat_stem (str): Name of the poolmat file (no extension).
        OutFolder (Path, str, optional): Path to the location where the analysis should be run. All input files should be located here as well. Defaults to Path.cwd().
        maskname (str, optional): Name of the NIFTI file containing the data mask (no extension). Defaults to "Ref_Mask".
        LineShape (str, optional): The lineshape used for the MT pool (superlorentzian, lorentzian, gaussian, Default: None).
        TR (float, optional): The TR (seconds) for the sequences. Defaults to None.
        thetaEX (float, optional): The excitation flip angle (degrees) for the sequences. Defaults to None.
        B1Name (str, optional): The NIFTI filename of the B1 Map (no extension). Defaults to None.
        T1Name (str, optional): The NIFTI filename of the T1 map (no extension). Defaults to None.

    """

    it = [x is None for x in [LineShape, TR, thetaEX]]
    if any(it) and not all(it):
        raise ParameterError("If one of (LineShape, TR, thetaEX) defined, all must be definied")
    
    # Creates Path object to directory where file will be created
    filename = OutFolder / (data_stem + ".sh")

    # Opens file and writes script
    with filename.open(mode="w") as FID:
        FID.write("#!/bin/sh\n\n")

        # If it will be on Jalapeno, use jalapeno flag to set proper executable directory
        FID.write("fabber_cest \\\n")

        # Build rest of script
        FID.write(f"--data={data_stem}.nii.gz \\\n")
        FID.write(f"--mask={maskname}.nii.gz \\\n")
        FID.write(
            f"--method=vb --noise=white --model=cest --data-order=singlefile \\\n"
        )
        FID.write(f"--max-iterations=20 --output={data_stem} --save-model-extras \\\n")
        FID.write(f"--spec={dataspec_stem}.txt --t12prior")
        FID.write(f" --pools={poolmat_stem}.txt --ptrain={ptrain_stem}.txt \\\n")
        FID.write(f"--save-model-fit --satspoil \\\n")

        # If using a lineshape, set it
        if LineShape is not None:
            FID.write(f"--lineshape={LineShape} ")

        # if using TR
        if TR is not None:
            FID.write(f"--TR={TR} ")

        # if using thetaEX
        if thetaEX is not None:
            FID.write(f"--EXFA={thetaEX} ")

        # psp counter is used to set the proper number of PSP_byname settings, will count from 1
        pspcounter = 1
        # Inputs B1 dataset
        if B1Name is not None:
            FID.write(
                f"\\\n--PSP_byname{pspcounter}=B1corr --PSP_byname{pspcounter}_type=I "
            )
            FID.write(f"--PSP_byname{pspcounter}_image={B1Name}.nii.gz ")
            pspcounter += 1

        # Inputs T1 dataset
        if T1Name is not None:
            FID.write(
                f"\\\n--PSP_byname{pspcounter}=T1a --PSP_byname{pspcounter}_type=I "
            )
            FID.write(f"--PSP_byname{pspcounter}_image={T1Name}.nii.gz ")
            FID.write(f"--PSP_byname{pspcounter}_prec=1e10 ")
            pspcounter += 1

    # Set executable
    filename.chmod(0o755)

    return None

def create_fabbert1_prompt(
    t1name, maskname, TR=None, OutFolder=Path.cwd(), fas=[25, 20, 15, 10, 5], B1Name=None, IR=False
):
        """Generates the BASH file to run FABBER T1 for the given dataset.
    
    Args:
        t1name (str): Name of the NIFTI file holding the T1 data (no extension).
        maskname (str): Name of the NIFTI file containing the data mask (no extension). 
        OutFolder (Path, str, optional): Path to the location where the analysis should be run. All input files should be located here as well. Defaults to Path.cwd().
        fas (list, optional): The flip angles [degrees] or IR values [seconds] for the sequences. Defaults to [25, 20, 15, 10, 5].
        B1Name (str, optional): The NIFTI filename of the B1 Map (no extension). Defaults to None.
        IR (bool, optional): Flag to switch between VFA and IR methods of T1 estimation. Defaults to False.

    """
    
    if not IR and TR is None:
        raise NoTRError(
                "No TR specified for VFA T1 Data!\nProvide a TR to proceed!"
            )
    
    filename = OutFolder / "VFA_FABBER.sh"

    with filename.open(mode="w") as FID:
        FID.write("#!/bin/sh\n\n")
        FID.write("fabber_t1 \\\n")

        FID.write("--output=VFA_Output \\\n")
        FID.write(f"--data={t1name}.nii.gz \\\n")

        FID.write(f"--mask={maskname}.nii.gz \\\n")

        FID.write("--method=vb ")
        if IR:
            with (OutFolder / "IR_TIs.txt").open(mode="w") as FID2:
                for fa in fas:
                    FID2.write(f"{fa}\n")
            FID.write("--model=ir --tis-file=IR_TIs.txt \\\n")
        else:
            with (OutFolder / "VFA_FAs.txt").open(mode="w") as FID2:
                for fa in fas:
                    FID2.write(f"{fa}\n")
            FID.write("--model=vfa --fas-file=VFA_FAs.txt \\\n")
        FID.write("--spatial --noise=white \\\n")
        FID.write("--data-order=singlefile --save-model-fit \\\n")
        
        if not IR:
            FID.write(f"--tr={TR:.6E} \\\n")

        if B1Name is not None:
            FID.write(f"--PSP_byname1=B1corr --PSP_byname1_type=I ")
            FID.write(f"--PSP_byname1_image={B1Name}.nii.gz")

    filename.chmod(0o755)

    return None

class NoTRError(Exception):
    pass

class ParameterError(Exception):
    pass
    
