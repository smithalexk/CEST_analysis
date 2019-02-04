# Generate dataspec file
from pathlib import Path
import numpy as np
from CEST.CESTClass import CESTModel


def create_fabber_ptrain_dataspec(
    offsetsfile,
    specfile=(Path.cwd() / "dataspec.txt"),
    ptrainfile=(Path.cwd() / "ptrain.txt"),
    nCESTp=[100],
    pw=[20e-3],
    dutycycle=[0.5],
    nomB1=[180],
    cwep=True,
):

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

    for ii, bb in enumerate(nomB1):
        if isinstance(offsetsfile, list):
            ppmoffsets = np.loadtxt(str(offsetsfile[ii]))
        elif ii == 0:
            ppmoffsets = np.loadtxt(str(offsetsfile))

        b1 = CESTModel(
            M0=[1, 0.2], T1=[1, 1], T2=[0.6, 0.6], B0=7, Rx=50, ChemShift=[0, 1]
        )
        b1.setsequence(
            ppmvec=ppmoffsets,
            TR=4,
            thetaEX=5,
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
                            "{0:.6E}\t{1:.6E}\t{2:.6E}\n".format(
                                offset, b1.B1eCEST[0] * 1e-6, 1.0
                            )
                        )
                    else:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}\n".format(
                                offset, b1.B1amp * 1e-6, nCESTp
                            )
                        )
        else:
            with specfile.open(mode="a") as FID:
                for offset in ppmoffsets:
                    if cwep:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}\n".format(
                                offset, b1.B1eCEST[0] * 1e-6, 1.0
                            )
                        )
                    else:
                        FID.write(
                            "{0:.6E}\t{1:.6E}\t{2:.6E}\n".format(
                                offset, b1.B1amp * 1e-6, nCESTp
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
    MaskName="Ref_Mask",
    LineShape=None,
    TR=None,
    thetaEX=None,
    B1Name=None,
    T1Name=None,
    jalapeno=False,
):

    # Creates Path object to directory where file will be created
    filename = OutFolder / (data_stem + ".sh")

    # Opens file and writes script
    with filename.open(mode="w") as FID:
        FID.write("#!/bin/sh\n\n")

        # If it will be on Jalapeno, use jalapeno flag to set proper executable directory
        if jalapeno:
            FID.write(
                "/home/fs0/asmith/scratch/Fabber/fabber_models_cest/fabber_cest \\\n"
            )
        else:
            FID.write(
                "/Users/asmith/Documents/Research/Oxford/Fabber/fabber_models_cest/build/fabber_cest \\\n"
            )

        # Build rest of script
        FID.write(f"--data={data_stem}.nii.gz \\\n")
        FID.write(f"--mask={MaskName}.nii.gz \\\n")
        FID.write(
            f"--method=vb --noise=white --model=cest --data-order=singlefile \\\n"
        )
        FID.write(f"--max-iterations=20 --output={data_stem} \\\n")
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
            FID.write(f"--PSP_byname{pspcounter}_prec=1e10 ")
            pspcounter += 1

        # Inputs T1 dataset
        if T1Name is not None:
            FID.write(
                f"\\\n--PSP_byname{pspcounter}=T1 --PSP_byname{pspcounter}_type=I "
            )
            FID.write(f"--PSP_byname{pspcounter}_image={T1Name}.nii.gz ")
            FID.write(f"--PSP_byname{pspcounter}_prec=1e10 ")
            pspcounter += 1

    # Set executable
    filename.chmod(0o755)

    return None


def create_fabbert1_prompt(
    t1name, maskname, TR, OutFolder=Path.cwd(), fas=[25, 20, 15, 5, 5], B1Name=None
):

    with (OutFolder / "VFA_FAs.txt").open(mode="w") as FID:
        for fa in fas:
            FID.write(f"{fa}\n")

    filename = OutFolder / "VFA_FABBER.sh"

    with filename.open(mode="w") as FID:
        FID.write("#!/bin/sh\n\n")
        FID.write(
            "/Users/asmith/Documents/Research/Oxford/Fabber/fabber_models_T1/build/fabber_t1 \\\n"
        )

        FID.write("--output=VFA_Output \\\n")
        FID.write(f"--data={t1name}.nii.gz \\\n")

        FID.write(f"--mask={maskname}.nii.gz \\\n")

        FID.write("--fas-file=VFA_FAs.txt \\\n")

        FID.write("--method=vb --model=vfa --spatial --noise=white \\\n")
        FID.write("--data-order=singlefile --save-model-fit \\\n")

        FID.write(f"--tr={TR:.6E} \\\n")

        if B1Name is not None:
            FID.write(f"--PSP_byname1=B1corr --PSP_byname1_type=I ")
            FID.write(f"--PSP_byname1_image={B1Name}.nii.gz")

    filename.chmod(0o755)

    return None
