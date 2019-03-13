from math import pi, sqrt, exp, sin, cos
import numpy as np
from scipy import interpolate as interp
from pathlib import Path


class CESTModel:
    """ CESTModel - Class for Producing a CEST Spectrum
    This class will produce a CEST z-Spectrum from a set of input values.
    Supports N pools, with the 1st pool representing the water pool, and the
    Nth pool representing the macromolecular pool with a Gaussian,
    Lorentzian, Super-Lorentzian, or no lineshape.

    Constructor Syntax:  Obj = CESTClass(M0,T1,T2,Rx,B0,ChemShift,lineshape)
    
    Arguments:
    M0 - a 1xN vector listing the relative pool sizes, with the 1st pool
            representing water
    T1 - A 1xN vector listing the T1's (s) of each pool
    T2 - A 1xN vector listing the T2's (s) of each pool
    Rx - A 1x(N-1) vector listing the exchange rates (s) between each pool
            and water.
    B0 - The field strength in tesla.
    ChemShift - A 1xN vector listing the relative 

    Public Methods:
    Self.setSequence(ppmvec,TR,thetaEX,B1Shape,pwCEST,B1amp,DutyCycle,nCESTp,InterSpoil)
        -Sets the sequence parameters to be used by the class to create a
            Z-spectrum.
        Inputs:
            ppmvec - vector of saturation offsets in ppm.
            TR - The TR of the sequence (s).
            thetaEX - The flip angle of the excitation pulse in degrees.
            B1Shape - The shape of the saturation pulse 
                        ('Philips-SincGauss' or 'Siemens-Gauss').
            pwCEST - The pulse width of the saturation pulse (s).
            B1amp - The amplitude of the saturation pulse (degrees/uT/T).
            DutyCycle - The duty cycle of the pulse train (range: [0,1]).
            nCESTp - The number of pulses in the saturation train (>=1).
            InterSpoil - Flag to specify whether to include inter-pulse
                            spoiling.  Requirements: DutyCycle < 1
        
    Self.CEST_NPool(obj,CWEPFlag)
        -Creates a Z-Spectrum from the class members.
        Inputs:
            CWEPFlag - Flag to create a Continuous wave equivalent square 
                        pulse in place of a saturation pulse train.
    Public Member Variables:
            B0 - Scanner Field Strength
            B1_Envelope - MT Pulse Shape
            B1_Timing - Timing for B1_Envelope (s)
            B1eCEST - Square B1 Equivalent Pulse of pulse train (uT)
            pwCEST - CEST Pulse Duration (s)
            B1amp - Maximum B1 Amplitude (uT)        
            deltappm - Frequency Samples in ppm

    Other m-files required: none
    Private/Protected Subfunctions: CESTStruct(), absorptionLineShape(), 
                                    setPulseEnvelope(), fullMT(), LineMT()

    
    Author:  Alex Smith
    Date:    28-Nov-2018
    Version: 1.0
    Changelog:

    20170526 - initial creation
    """
    # Constants
    gamma = 42.58 * 2 * pi  # rad/s-T
    gammaH = 42.58  # MHz/T

    # Fixed Sequence parameters
    pwEX = np.array([1e-3])  # Excitation Pulse length
    tauSpoil = 3e-3  # Spoiling Time

    def __init__(self, **kwargs):
        # Convert all inputs to lowercase
        kwargs = {k.lower(): np.array(v) for k, v in kwargs.items()}

        # Add attributes to object
        # If no attribute, provide default
        self.M0 = kwargs.get("m0", np.array([1]))
        self.T1 = kwargs.get("t1", np.array([1]))
        self.T2 = kwargs.get("t2", np.array([0.1]))
        self.Rx = np.append(0,kwargs.get("rx", []))
        self.B0 = kwargs.get("b0", 3)
        self.ChemShift = kwargs.get("chemshift", np.zeros(1))

        # Set to avoid errors in later function
        self.mtLineshape = None

        # Check to ensure Pool properties are properly set
        if self.M0.size > 1:
            self.assertpoolspecifications()

    def addpool(self, M0, T1=1, T2=0.010, Rx=50, ChemShift=0):
        """ Adds a new pool to the CEST model

        Adds a new pool to the model. All new pool parameters must be included.
        """
        self.M0 = np.append(self.M0, M0)
        self.T1 = np.append(self.T1, T1)
        self.T2 = np.append(self.T2, T2)
        self.Rx = np.append(self.Rx, Rx)
        self.ChemShift = np.append(self.ChemShift, ChemShift)

    def addMTpool(
        self, M0, T1=1, T2=10e-6, Rx=20, ChemShift=-2.41, lineshape="SuperLorentzian"
    ):
        """ Adds/Replaces an MT pool to the CEST model """
        self.mtM0 = M0
        self.mtT1 = T1
        self.mtT2 = T2
        self.mtRx = Rx
        self.mtChemShift = ChemShift
        self.mtLineshape = lineshape.lower()

    def setsequence(
        self,
        ppmvec,
        TR,
        thetaEX,
        B1Shape,
        pwCEST,
        B1amp,
        DutyCycle=1,
        nCESTp=1,
        InterSpoil=False,
    ):
        """ SETSEQUENCE - Builds the Pulse Sequence objects for use in MT Model
        Creates the pulse sequence timing and power matrices, as well as the
        overall sequence timing
        
        CESTModel.setSequence(ppmvec,TR,thetaEX,B1Shape,pwCEST,B1amp,DutyCycle,nCESTp,InterSpoil)
        
        Inputs:
            ppmvec      - Vector of Offsets [ppm]
            TR          - Repetition Time of Sequence (Pulse Train + Readout) 
                        [ms]
            thetaEX     - Excitation pulse flip angle [deg]
            B1Shape     - The shape of the saturation pulse 
                        ('CW', 'SincGauss', or 'Gauss').
            pwCEST      - The pulse width of the saturation pulse [s].
            B1amp       - The amplitude of the saturation pulse [degrees/uT/T].
            DutyCycle   - The duty cycle of the pulse train (range: [0,1]).
            nCESTp      - The number of pulses in the saturation train (>=1).
            InterSpoil  - Flag to specify whether to include inter-pulse
                            spoiling.  Requirements: DutyCycle < 1
        """
        self.deltaHz = ppmvec * self.B0 * self.gammaH
        self.DutyCycle = DutyCycle
        self.nCESTp = nCESTp
        self.InterSpoil = InterSpoil

        # Set Pulse Width (s)
        self.pwCEST = pwCEST

        # Convert to Radians
        self.thetaEX = thetaEX * pi / 180

        # Set sequence timing parmaters
        tMTDC = (1 - self.DutyCycle) * pwCEST / self.DutyCycle
        self.tMTDC = tMTDC
        self._setsequence_tr(TR, tMTDC)

        # Find the Pulse Envelope
        PulseEnvelope = self._setpulse_envelope(pwCEST, B1Shape)

        self.B1_Timing = np.linspace(0, PulseEnvelope[-1, 0], 51)
        f = interp.interp1d(PulseEnvelope[:, 0], PulseEnvelope[:, 1])
        self.B1_Envelope = f(self.B1_Timing)

        # Find Correct B1 amplitude
        if B1amp > 25:
            print("Assuming B1amp was entered in Degrees, converting to uT\n")

            B1CEST = (
                self.B1_Envelope
                * B1amp
                * pi
                / 180
                / (np.trapz(y=self.B1_Envelope, x=self.B1_Timing) * self.gamma)
            )
            self.B1amp = max(B1CEST)
        elif B1amp < 0.01:
            print("Assuming B1amp was entered in T, converting to uT\n")
            self.B1amp = B1amp * 1e6
        else:
            self.B1amp = B1amp

        # Find the Continuous Wave equivalent pulse power
        B1e_Timing = self.B1_Timing
        B1e_amp = np.squeeze(
            self.B1_Envelope * self.B1amp
        )  # elementwise multiplication

        # Build Pulse Train
        B1e_amp = np.delete(np.tile(np.append(B1e_amp, 0), nCESTp), -1)
        B1e_Timing = np.append(B1e_Timing, tMTDC + B1e_Timing[-1])
        TmpTiming = np.append(self.B1_Timing, self.B1_Timing[-1] + tMTDC)
        for _ in range(nCESTp - 1):
            tmp = B1e_Timing[-1]
            B1e_Timing = np.append(B1e_Timing, TmpTiming + tmp)
        B1e_Timing = np.delete(B1e_Timing, -1)
        # Find Total Integrated Power
        w12int = np.unique(np.trapz(np.squeeze(B1e_amp) ** 2, B1e_Timing))

        # Divide by Timing for equivalent Square Pulse
        self.B1eCEST = np.sqrt(w12int / ((pwCEST + tMTDC) * nCESTp))
        self.B1eTiming = (pwCEST + tMTDC) * nCESTp

        self.deltappm = ppmvec

    def cest_npool(self, CWEPFlag=True):
        if self.mtLineshape == None:
            return self._fullMT(CWEPFlag)
        else:
            return self._lineMT(CWEPFlag)

    def cwe_pulse(self, CWEPFlag=True):
        pulseparams = {}
        if CWEPFlag:
            pulseparams.update(
                pwMT=np.array([self.pwCEST / self.DutyCycle * self.nCESTp])
            )
            pulseparams.update(B1amp=np.array([self.B1eCEST]))
            pulseparams.update(DutyCycle=1.0)
            pulseparams.update(nCESTp=1)
            pulseparams.update(InterSpoil=False)
        else:
            pulseparams.update(pwMT=self.B1_Timing)
            pulseparams.update(B1amp=self.B1_Envelope * self.B1amp)
            pulseparams.update(DutyCycle=self.DutyCycle)
            pulseparams.update(nCESTp=self.nCESTp)
            pulseparams.update(InterSpoil=self.InterSpoil)
        return pulseparams

    def _fullMT(self, CWEPFlag):
        PulseParams = self.cwe_pulse(CWEPFlag)

        # Determine Number of Pools
        npools = self.M0.size

        # Make sure arrays are proper dim
        MM = self.M0

        M0 = np.zeros(npools * 3)
        for ii in range(npools):
            M0[2 + ii * 3] = MM[ii]

        # Calculate relaxation/exchange matrices
        kjf = self.Rx[1:]
        kfj = kjf * MM[1:] / MM[0]
        ks = np.zeros((npools, npools))
        for ii in range(1, npools):
            ks[0, ii] = kfj[ii - 1]
            ks[ii, 0] = kjf[ii - 1]

        R2 = 1 / self.T2

        # Matrix is of the form (Mxf Myf Mzf, Mxs1, Mys1, Mzs1, ..., MzMT)
        RL = np.zeros((npools * 3, npools * 3))
        for ii in range(npools):
            for jj in range(npools):
                kLs = np.eye(3) * ks[jj, ii]
                RL[ii * 3 : 3 + ii * 3, jj * 3 : 3 + jj * 3] = kLs

            RLs = np.zeros((3, 3))
            RLs[0, 0] = -R2[ii] - ks[ii, :-1].sum()
            RLs[1, 1] = RLs[0, 0]
            RLs[2, 2] = -1 / self.T1[ii] - ks[ii, :].sum()
            RL[ii * 3 : 3 + ii * 3, ii * 3 : 3 + ii * 3] = RLs

        #  Determine Duty Cycle time
        if PulseParams["nCESTp"] > 1:
            tMTDC = (
                (1 - PulseParams["DutyCycle"])
                * PulseParams["pwMT"][-1]
                / PulseParams["DutyCycle"]
            )
        else:
            tMTDC = 0

        # Define approximate post MT spoiling time
        ts = 3e-3

        # Readout Length
        if isinstance(self.pwEX, np.ndarray):
            tr = self.TR - self.pwEX[-1]
            assert tr > 0, "Sequence Timing does not add to > 0!"
        else:
            tr = self.TR

        # Set Mz based on size of RL
        nr_RL = RL.shape[0]
        Mz = np.zeros((nr_RL, len(self.deltaHz)))

        # Exponential Matrices for the s and r phases of the pulse sequence
        De, V = np.linalg.eig(RL * ts)
        Es = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        De, V = np.linalg.eig(RL * tr)
        Er = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        De, V = np.linalg.eig(RL * tMTDC)
        Edc = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))

        D = np.diag(M0 > 0).astype(int)

        if PulseParams["InterSpoil"]:
            assert (
                PulseParams["DutyCycle"] != 1.0
            ), "Cannot Spoil between pulses if Duty Cycle == 1!"
            iD = D
        else:
            iD = np.eye(nr_RL)

        # Set omega1
        w1 = self.gamma * PulseParams["B1amp"]

        # Offsets with respect to Pulse sequence
        dw = (
            (self.deltaHz.reshape(-1, 1) - self.ChemShift * self.gammaH * self.B0)
            * 2
            * pi
        )
        dw = dw.T
        deltaMT = self.deltaHz * 2 * pi

        # Set Excitation Pulse
        C = np.diag(
            np.tile([sin(self.thetaEX), sin(self.thetaEX), cos(self.thetaEX)], npools)
        )

        if len(PulseParams["pwMT"]) > 1:
            ds = PulseParams["pwMT"] - np.append(0, PulseParams["pwMT"][0:-1])
        else:
            ds = PulseParams["pwMT"]

        # Loop for each RF power and offset
        for ii in range(len(deltaMT)):
            Ems = np.zeros((nr_RL, nr_RL))
            Emt = np.zeros((nr_RL, nr_RL))

            for jj in range(PulseParams["pwMT"].size):

                # Saturation Matrix
                # Matrix is of the form (Mxf Myf Mzf, Mxs1, Mys1, Mzs1, ..., MzMT)
                W = np.zeros((nr_RL, nr_RL))
                for kk in range(npools):
                    Ws = np.zeros((3, 3))
                    Ws[0, 1] = -dw[kk, ii]
                    Ws[1, 0] = dw[kk, ii]
                    Ws[1, 2] = -w1[jj]
                    Ws[2, 1] = w1[jj]
                    W[kk * 3 : 3 + kk * 3, kk * 3 : 3 + kk * 3] = Ws

                De, V = np.linalg.eig((RL + W) * ds[jj])
                Em = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))

                if jj == 0:
                    Emt = Em
                    Ems = (np.eye(nr_RL) - Em) @ np.linalg.inv((RL + W))
                else:
                    Emt = Em @ Emt
                    Ems = Em @ Ems + (np.eye(nr_RL) - Em) @ np.linalg.inv(RL + W)

            Emdc = np.linalg.matrix_power(Emt @ iD @ Edc, PulseParams["nCESTp"] - 1)
            Emm = np.eye(nr_RL)
            Emb = np.eye(nr_RL)
            for jj in range(PulseParams["nCESTp"] - 1):
                if jj == PulseParams["nCESTp"] - 2:
                    Emb = Emm
                Emm = Emm + np.linalg.matrix_power(Emt @ iD @ Edc, jj + 1)

            Mz[:, ii] = (
                np.linalg.inv(np.eye(nr_RL) - Es @ Emdc @ Emt @ D @ Er @ C @ D)
                @ (
                    Es @ Emdc @ Emt @ D @ (np.eye(nr_RL) - Er)
                    + Es @ Emb @ Emt @ iD @ (np.eye(nr_RL) - Edc)
                    + np.eye(nr_RL)
                    - Es
                    + Es @ Emm @ Ems @ RL
                )
                @ M0
            )

        # Normalize according to same sequence w/out MT prepulse
        De, V = np.linalg.eig(RL * self.TR)
        Er0 = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        Mz0 = np.linalg.inv(np.eye(nr_RL) - Er0 @ C @ D) @ ((np.eye(nr_RL) - Er0) @ M0)

        Mzn = np.abs(Mz[2, :] / Mz0[2])
        # Extract free pool
        Mz = np.abs(Mz[2, :])

        return (Mzn, Mz, Mz0)

    def _lineMT(self, CWEPFlag):
        PulseParams = self.cwe_pulse(CWEPFlag)

        # Determine Number of Pools
        npools = self.M0.size + 1

        # Make sure arrays are proper dim
        MM = self.M0

        M0 = np.zeros((npools - 1) * 3 + 1)
        for ii in range(npools - 1):
            M0[2 + ii * 3] = MM[ii]
        M0[-1] = self.mtM0

        # Calculate relaxation/exchange matrices
        kjf = np.append(self.Rx[1:], self.mtRx)
        kfj = kjf * np.append(MM[1:], self.mtM0) / MM[0]
        ks = np.zeros((npools, npools))
        for ii in range(1, npools):
            ks[0, ii] = kfj[ii - 1]
            ks[ii, 0] = kjf[ii - 1]

        R2 = np.append(1 / self.T2, 1 / self.mtT2)

        # Matrix is of the form (Mxf Myf Mzf, Mxs1, Mys1, Mzs1, ..., MzMT)
        RL = np.zeros((1 + (npools - 1) * 3, 1 + (npools - 1) * 3))
        for ii in range(npools - 1):
            for jj in range(npools - 1):
                kLs = np.eye(3) * ks[jj, ii]
                RL[ii * 3 : 3 + ii * 3, jj * 3 : 3 + jj * 3] = kLs

            RLs = np.zeros((3, 3))
            RLs[0, 0] = -R2[ii] - ks[ii, :-1].sum()
            RLs[1, 1] = RLs[0, 0]
            RLs[2, 2] = -1 / self.T1[ii] - ks[ii, :].sum()
            RL[ii * 3 : 3 + ii * 3, ii * 3 : 3 + ii * 3] = RLs

        RL[-1, -1] = -1 / self.mtT1 - ks[-1, :].sum()
        RL[2, -1] = ks[-1, 0]
        RL[-1, 2] = ks[0, -1]

        # Calc MT RF saturation rate
        gb = self._absorptionlineshape(
            self.mtT2,
            self.deltaHz - self.mtChemShift * self.gammaH * self.B0,
            self.mtLineshape,
        )

        Wb = np.zeros((len(PulseParams["B1amp"]), len(gb)))
        for ii in range(len(PulseParams["pwMT"])):
            Wb[ii, :] = pi * self.gamma ** 2 * gb * PulseParams["B1amp"][ii] ** 2
        #  Determine Duty Cycle time
        if PulseParams["nCESTp"] > 1:
            tMTDC = (
                (1 - PulseParams["DutyCycle"])
                * PulseParams["pwMT"][-1]
                / PulseParams["DutyCycle"]
            )
        else:
            tMTDC = 0

        # Define approximate post MT spoiling time
        ts = 3e-3

        # Readout Length
        if isinstance(self.pwEX, np.ndarray):
            tr = self.TR - self.pwEX[-1]
            assert tr > 0, "Sequence Timing does not add to > 0!"
        else:
            tr = self.TR

        # Set Mz based on size of RL
        nr_RL = RL.shape[0]
        Mz = np.zeros((nr_RL, len(self.deltaHz)))

        # Exponential Matrices for the s and r phases of the pulse sequence
        De, V = np.linalg.eig(RL * ts)
        Es = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        De, V = np.linalg.eig(RL * tr)
        Er = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        De, V = np.linalg.eig(RL * tMTDC)
        Edc = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))

        D = np.diag(M0 > 0).astype(int)

        if PulseParams["InterSpoil"]:
            assert (
                PulseParams["DutyCycle"] != 1.0
            ), "Cannot Spoil between pulses if Duty Cycle == 1!"
            iD = D
        else:
            iD = np.eye(nr_RL)

        # Set omega1
        w1 = self.gamma * PulseParams["B1amp"]

        # Offsets with respect to Pulse sequence
        dw = (
            (
                self.deltaHz.reshape(-1, 1)
                - np.append(self.ChemShift, self.mtChemShift) * self.gammaH * self.B0
            )
            * 2
            * pi
        )
        dw = dw.T
        deltaMT = self.deltaHz * 2 * pi

        # Set Excitation Pulse
        C = np.diag(
            np.append(
                np.tile(
                    [sin(self.thetaEX), sin(self.thetaEX), cos(self.thetaEX)],
                    npools - 1,
                ),
                cos(self.thetaEX),
            )
        )

        if len(PulseParams["pwMT"]) > 1:
            ds = PulseParams["pwMT"] - np.append(0, PulseParams["pwMT"][0:-1])
        else:
            ds = PulseParams["pwMT"]

        # Loop for each RF power and offset
        for ii in range(len(deltaMT)):
            Ems = np.zeros((nr_RL, nr_RL))
            Emt = np.zeros((nr_RL, nr_RL))

            for jj in range(PulseParams["pwMT"].size):

                # Saturation Matrix
                # Matrix is of the form (Mxf Myf Mzf, Mxs1, Mys1, Mzs1, ..., MzMT)
                W = np.zeros((nr_RL, nr_RL))
                for kk in range(npools - 1):
                    Ws = np.zeros((3, 3))
                    Ws[0, 1] = -dw[kk, ii]
                    Ws[1, 0] = dw[kk, ii]
                    Ws[1, 2] = -w1[jj]
                    Ws[2, 1] = w1[jj]
                    W[kk * 3 : 3 + kk * 3, kk * 3 : 3 + kk * 3] = Ws
                W[-1, -1] = -Wb[jj, ii]

                De, V = np.linalg.eig((RL + W) * ds[jj])
                Em = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))

                if jj == 0:
                    Emt = Em
                    Ems = (np.eye(nr_RL) - Em) @ np.linalg.inv((RL + W))
                else:
                    Emt = Em @ Emt
                    Ems = Em @ Ems + (np.eye(nr_RL) - Em) @ np.linalg.inv(RL + W)

            Emdc = np.linalg.matrix_power(Emt @ iD @ Edc, PulseParams["nCESTp"] - 1)
            Emm = np.eye(nr_RL)
            Emb = np.eye(nr_RL)
            for jj in range(PulseParams["nCESTp"] - 1):
                if jj == PulseParams["nCESTp"] - 2:
                    Emb = Emm
                Emm = Emm + np.linalg.matrix_power(Emt @ iD @ Edc, jj + 1)

            Mz[:, ii] = (
                np.linalg.inv(np.eye(nr_RL) - Es @ Emdc @ Emt @ D @ Er @ C @ D)
                @ (
                    Es @ Emdc @ Emt @ D @ (np.eye(nr_RL) - Er)
                    + Es @ Emb @ Emt @ iD @ (np.eye(nr_RL) - Edc)
                    + np.eye(nr_RL)
                    - Es
                    + Es @ Emm @ Ems @ RL
                )
                @ M0
            )

        # Normalize according to same sequence w/out MT prepulse
        De, V = np.linalg.eig(RL * self.TR)
        Er0 = np.real(V @ np.diag(np.exp(De)) @ np.linalg.inv(V))
        Mz0 = np.linalg.inv(np.eye(nr_RL) - Er0 @ C @ D) @ ((np.eye(nr_RL) - Er0) @ M0)

        Mzn = np.abs(Mz[2, :] / Mz0[2])
        # Extract free pool
        Mz = np.abs(Mz[2, :])

        return (Mzn, Mz, Mz0)

    def _setsequence_tr(self, TR, tMTDC):
        assert (
            TR
            - (
                self.pwCEST * self.nCESTp
                + tMTDC * (self.nCESTp - 1)
                + self.pwEX
                + self.tauSpoil
            )
            > 0
        ), "TR is not long enough by {0}".format(
            (
                self.pwCEST * self.nCESTp
                + tMTDC * (self.nCESTp - 1)
                + self.pwEX
                + self.tauSpoil
            )
            - TR
        )

        self.TR = TR

    # Static Methods:

    @staticmethod
    def _absorptionlineshape(T2, deltaMT, lineshape, cutoff=1000):
        """ absorptionLineShape - Creates Lineshapes for use with MT imaging
        
        Creates a lineshape based on the 'lineShape' input over a set of offsets
        for a given T2 value
        
        Syntax:  g = _absorptionLineShape(T2,delta,lineShape)
        
        Inputs:
            T2 - The T2 of the pool (s)
            delta - A vector containing the offsets in Hz for the lineshape.
            lineShape - A string which defines the shape used to create this
                lineshape.
                superlorentzian: Creates a super-Lorentzian lineshape.  
                    To overcome the discontinuity near 0 Hz, creates an interpolated
                    value from 1000 Hz away from 0 Hz.
                lorentzian: Creates a Lorentzian lineshape of the form 1/(1+x^2),
                    where x = 2*pi*delta*T2.
                gaussian: Creates a Gaussian lineshape.
                    
        
        Outputs:
            The lineshape vector over all deltas
        """

        #  Calculate g for specified lineshape
        if lineshape.lower() == "lorentzian":
            return (T2 / pi) * (1 / (1 + (2 * pi * deltaMT * T2) ** 2))
        elif lineshape.lower() == "gaussian":
            return (T2 / np.sqrt(2 * pi)) * np.exp(-(2 * pi * deltaMT * T2) ** 2 / 2)
        elif lineshape.lower() == "superlorentzian":

            #  Find abs(g) > 1000 Hz
            deltac = np.append(
                np.linspace(-1e5, -cutoff, 1e3), np.linspace(cutoff, 1e5, 1e3)
            )
            du = 2.0e-3
            u = np.arange(0, 1 + du, du)
            f = (sqrt(2 / pi) * (T2 / np.abs(3 * u ** 2 - 1))) * np.exp(
                -2
                * (
                    np.expand_dims((2 * pi * deltac * T2), axis=1)
                    @ np.expand_dims((1 / np.abs(3 * u ** 2 - 1)), axis=0)
                )
                ** 2
            )

            #  Interpolate to values of abs(g) < 1 kHz
            s = interp.InterpolatedUnivariateSpline(deltac, np.sum(f, 1) * du)
            g = s(deltaMT)
            g[g < 0] = 0
            return g

        else:
            return np.zeros(deltaMT.shape)

    @staticmethod
    def _setpulse_envelope(pw, B1Shape):
        # Get shape of RF pulse
        with (Path(__file__).parent / (B1Shape.lower() + ".txt")).open(mode='r') as FID:
            B1s = np.array([float(i) for i in FID.readlines()])

        # Calculate pulse based B1amp amd pw
        B1_Envelope = B1s / max(B1s)
        B1_Timing = np.linspace(0, pw, B1s.size)
        return np.column_stack((B1_Timing, B1_Envelope))

    # Error Handling:

    # Check Pool specifications utilised properly
    def assertpoolspecifications(self):
        assert self.M0.size == self.T1.size, "Pool Specifications not equal!"
        assert self.M0.size == self.T2.size, "Pool Specifications not equal!"
        assert self.M0.size == self.Rx.size, "Pool Specifications not equal!"
        assert self.M0.size == self.ChemShift.size, "Pool Specifications not equal!"
