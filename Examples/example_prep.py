#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# Std Lib
from pathlib import Path
import shutil

# Conda
import numpy as np
from nipype.interfaces import fsl

def main():
    
    with  open("poolmat.txt","w") as FID:
        FID.write(
                " 1.2774000e+08   0.0000000e+00   1.3000000e+00   7.0000000e-02\n"
            )
        FID.write(
                "3.5000000e+00   2.0000000e+01   0.7700000e+00   3.0000000e-04\n"
            )
        FID.write(" 0.0000000e+00   1.0000000e+01   1.0000000e+00   1.0000000e-05")

    #  Path To CEST Data
    cest_prefix = "registered_CEST_Data"

    B1_prefix = "B1map"

    t1_prefix = "VFA_Data"
    t1_tr = 20e-3
    t1_fas = [21, 6]

    # SSI of Pulse Sequence
    SSI = 2.375

    # Excitation Flip Angle
    thetaEX = 7

    nomB1 = 184

    # Load ppmoffsets
    cestoffsetsfile =  "ppm_offsets.txt"
    ppmoffsets = np.loadtxt(str(cestoffsetsfile))

    # Create temporary folder to perform splitting and reforming operations on
    TempFolder = Path("TempFolder")

    TempFolder.mkdir(exist_ok=True, parents=True)

    # Build VFA FABBER Input
    genfabber.create_fabbert1_prompt(
        t1name=t1_prefix,
        maskname="Ref_Mask",
        TR=t1_tr,
        fas=t1_fas,
        B1Name="B1map_resize",
    )

    if Path("VFA_Output").exists():
        shutil.rmtree("VFA_Output")
    

    cestoffsetsfile = TempFolder / "CEST_offsets.txt"

    np.savetxt(cestoffsetsfile, ppmoffsets, fmt="%3.4f")

    genfabber.create_fabber_ptrain_dataspec(
        offsetsfile=cestoffsetsfile,
        specfile="dataspec.txt",
        ptrainfile="ptrain.txt",
        nCESTp=50,
        pw=20e-3,
        dutycycle=0.5,
        nomB1=nomB1,
        cwep=True,
        thetaEX=thetaEX,
        TR=SSI,
    )

    # Create new Fabber input script for running Fabber CEST
    genfabber.create_fabbercest_prompt(
        data_stem=cest_prefix,
        dataspec_stem=f"dataspec",
        ptrain_stem="ptrain",
        poolmat_stem="poolmat",
        maskname="Ref_Mask",
        LineShape="superlorentzian",
        TR=SSI,
        thetaEX=thetaEX,
        B1Name=B1_prefix,
        T1Name="T1Map",
    )

    return None


if __name__ == "__main__":
    import sys
    from os import path
    sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
    import genfabber
    main()
