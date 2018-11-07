#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Yuan Yao <yy682@cornell.edu>
#

'''
SHCI solver for CASCI and CASSCF using ARROW.
'''
import json
import ctypes
import os
import sys
import struct
import time
import tempfile
from subprocess import check_call, CalledProcessError
import numpy
import pyscf.tools
import pyscf.lib
from pyscf.lib import logger
from pyscf.lib import chkfile
from pyscf.lib import load_library
from pyscf import mcscf
from pyscf.dmrgscf import dmrg_sym
ndpointer = numpy.ctypeslib.ndpointer

# Settings
try:
   from pyscf.shciscf import settings
except ImportError:
    from pyscf import __config__
    settings = lambda: None
    settings.SHCIEXE = getattr(__config__, 'shci_SHCIEXE', None)
    settings.SHCISCRATCHDIR = getattr(__config__, 'shci_SHCISCRATCHDIR', None)
    settings.SHCIRUNTIMEDIR = getattr(__config__, 'shci_SHCIRUNTIMEDIR', None)
    settings.MPIPREFIX = getattr(__config__, 'shci_MPIPREFIX', None)
    if (settings.SHCIEXE is None or settings.SHCISCRATCHDIR is None):
        import sys
        sys.stderr.write('settings.py not found.  Please create %s\n'
                         % os.path.join(os.path.dirname(__file__), 'settings.py'))
        raise ImportError('settings.py not found')

# Libraries
libE3unpack = load_library('libicmpspt')
# TODO: Organize this better.
shciLib = load_library('libshciscf')

transformDinfh = shciLib.transformDinfh
transformDinfh.restyp = None
transformDinfh.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int32),
                      ndpointer(ctypes.c_int32), ndpointer(ctypes.c_double),
                      ndpointer(ctypes.c_double),ndpointer(ctypes.c_double)]

transformRDMDinfh = shciLib.transformRDMDinfh
transformRDMDinfh.restyp = None
transformRDMDinfh.argtypes = [ctypes.c_int, ndpointer(ctypes.c_int32),
                      ndpointer(ctypes.c_int32), ndpointer(ctypes.c_double),
                      ndpointer(ctypes.c_double),ndpointer(ctypes.c_double)]

writeIntNoSymm = shciLib.writeIntNoSymm
writeIntNoSymm.argtypes = [ctypes.c_int, ndpointer(ctypes.c_double),ndpointer(ctypes.c_double),
                           ctypes.c_double, ctypes.c_int, ndpointer(ctypes.c_int)]

fcidumpFromIntegral = shciLib.fcidumpFromIntegral
fcidumpFromIntegral.restype = None
fcidumpFromIntegral.argtypes = [ctypes.c_char_p,
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t,
                                ctypes.c_size_t,
                                ctypes.c_double,
                                ndpointer(ctypes.c_int32, flags="C_CONTIGUOUS"),
                                ctypes.c_size_t,
]

r2RDM = shciLib.r2RDM
r2RDM.restype = None
r2RDM.argtypes = [ ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                  ctypes.c_size_t,
                  ctypes.c_char_p]


class SHCI(pyscf.lib.StreamObject):
    r'''SHCI program interface and object to hold SHCI program input parameters.

    Attributes:
        davidsonTol: double
        epsilon2: double
        epsilon2Large: double
        targetError: double
        sampleN: int
        epsilon1: vector<double>
        onlyperturbative: bool
        restart: bool
        fullrestart: bool
        dE: double
        eps: double
        prefix: str
        stochastic: bool
        nblocks: int
        excitation: int
        nvirt: int
        singleList: bool
        io: bool
        nroots: int
        nPTiter: int
        DoRDM: bool
        sweep_iter: [int]
        sweep_epsilon: [float]
        initialStates: [[int]]
        groupname : str
            groupname, orbsym together can control whether to employ symmetry in
            the calculation.  "groupname = None and orbsym = []" requires the
            SHCI program using C1 symmetry.
        useExtraSymm : False
            if the symmetry of the molecule is Dooh or Cooh, then this keyword uses
            complex orbitals to make full use of this symmetry

    Examples:


    '''

    def __init__(self, mol=None, mo_coeff=None, mo_occ=None, maxM=None, tol=None, num_thrds=1, memory=None):
        self.mol = mol
        if mol is None:
            self.stdout = sys.stdout
            self.verbose = logger.NOTE
        else:
            self.stdout = mol.stdout
            self.verbose = mol.verbose
        self.outputlevel = 2

        self.executable = settings.SHCIEXE
        self.Lz_relabel = settings.LZ_RELABEL
        self.mpiprefix = settings.MPIPREFIX
        self.memory = memory

        self.integralFile = "FCIDUMP"
        self.configFile = "config.json"
        self.outputFile = "shci.out"
        if hasattr(settings, 'SHCIRUNTIMEDIR'):
            self.runtimeDir = settings.SHCIRUNTIMEDIR
        else:
            self.runtimeDir = '.'

        self.spin = None
        # TODO: Organize into pyscf and SHCI parameters
        # Standard SHCI Input parameters
        self.davidsonTol = 5.e-5
        self.epsilon2 = 1.e-7
        self.epsilon2Large = 1000.
        self.targetError = 1.e-4
        self.sampleN = 200
        self.epsilon1 = None
        self.onlyperturbative = False
        self.fullrestart = False
        self.dE = 1.e-8
        self.eps = None
        self.prefix = "."
        self.stochastic = True
        self.nblocks = 1
        self.excitation = 1000
        self.nvirt = 1e6
        self.singleList = True
        self.io = True
        self.nroots = 1
        self.nPTiter = 0
        self.DoRDM = True
        self.sweep_iter =[]
        self.sweep_epsilon = []
        self.maxIter = 6
        self.restart = False
        self.num_thrds = num_thrds
        self.orbsym = []
        self.onlywriteIntegral = False
        self.orbsym = None
        self.mo_coeff = mo_coeff # TODO:correct or not?
        self.mo_occ = mo_occ
        if mol is None:
           self.groupname = None
        else:
           if mol.symmetry:
              self.groupname = mol.groupname
           else:
              self.groupname = None
        self._restart = False
        self.returnInt = False
        self._keys = set(self.__dict__.keys())
        self.irrep_nelec = None
        self.TimeSymm = False
        self.useExtraSymm = False
        self.initialStates = None
        self.extraline = ""
        self.RDMformat = 'txt'

    @property
    def threads(self):
        return self.num_thrds
    @threads.setter
    def threads(self, x):
        self.num_thrds = x

    def dump_flags(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        log = logger.Logger(self.stdout, verbose)
        log.info( '******** SHCI (Arrow) flags ********' )
        log.info( 'executable = %s', self.executable)
        log.info( 'mpiprefix = %s', self.mpiprefix )
        log.info( 'integralFile = %s', os.path.join(self.runtimeDir, self.integralFile) )
        log.info( 'configFile = %s', os.path.join(self.runtimeDir, self.configFile) )
        log.info( 'outputFile = %s', os.path.join(self.runtimeDir, self.outputFile) )
        log.info( 'maxIter = %d', self.maxIter )
        log.info( 'nPTiter = %i', self.nPTiter )
        log.info( 'Stochastic = %r', self.stochastic )
        log.info( 'num_thrds = %d', self.num_thrds )
        log.info( 'memory = %s', self.memory )
        return self

    def make_rdm1(self, state, norb, nelec, link_index=None, **kwargs):
# Avoid calling self.make_rdm12 because it may be overloaded
        return SHCI.make_rdm12(self, state, norb, nelec, link_index, **kwargs)[0]

    def make_rdm12(self, state, norb, nelec, link_index=None, **kwargs):
        nelectrons = 0
        if isinstance(nelec, (int, numpy.integer)):
            nelectrons = nelec
        else:
            nelectrons = nelec[0]+nelec[1]

        twopdm = numpy.zeros( (norb, norb, norb, norb) )
        
        if self.RDMformat != 'csv':
            # use .txt format
            #file2pdm = "%s/spatialRDM.%d.%d.txt"%(self.prefix,state, state)
            file2pdm = "spatialRDM.txt"
            file2pdm = file2pdm.encode()  # .encode for python3 compatibility
            r2RDM( twopdm, norb, file2pdm )
        else:
            # use .csv format
            dm_file = os.path.join(self.runtimeDir, '2rdm.csv')
            i, j, k, l, val = numpy.loadtxt(dm_file, dtype = numpy.dtype('i,i,i,i,d'),
                                            delimiter=',', skiprows=1, unpack=True)
            twopdm[i,j,k,l] = twopdm[j,i,l,k] = val
     
            twopdm = numpy.transpose(twopdm, (0,3,1,2))

        if (self.groupname == 'Dooh' or self.groupname == 'Coov') and self.useExtraSymm:
            nRows, rowIndex, rowCoeffs = DinfhtoD2h(self, norb, nelec)
            twopdmcopy = 1.*twopdm
            twopdm = 0.*twopdm

            twopdmcopy = numpy.ascontiguousarray(twopdmcopy, numpy.float64)
            twopdm = numpy.ascontiguousarray(twopdm, numpy.float64)
            transformRDMDinfh(norb, numpy.ascontiguousarray(nRows, numpy.int32),
                              numpy.ascontiguousarray(rowIndex, numpy.int32),
                              numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                              twopdmcopy,
                              twopdm)
            twopdmcopy = None

        onepdm = numpy.einsum('ikjj->ki', twopdm)
        onepdm /= (nelectrons-1)
        return onepdm, twopdm

    def kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        if self.nroots == 1:
            roots = 0
        else:
            roots = range(self.nroots)
        if fciRestart is None:
            fciRestart = self.restart or self._restart

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeSHCIConfFile(self, nelec, fciRestart)
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        if self.onlywriteIntegral:
            logger.info(self, 'Only write integral')
            try:
                calc_e = readEnergy(self)
            except IOError:
                if self.nroots == 1:
                    calc_e = 0.0
                else :
                    calc_e = [0.0] * self.nroots
            return calc_e, roots
        if self.returnInt:
            return h1e, eri

        executeSHCI(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)
        #exit(0)

        return calc_e, roots

    def approx_kernel(self, h1e, eri, norb, nelec, fciRestart=None, ecore=0, **kwargs):
        fciRestart = True

        if 'orbsym' in kwargs:
            self.orbsym = kwargs['orbsym']
        writeIntegralFile(self, h1e, eri, norb, nelec, ecore)
        writeSHCIConfFile(self, nelec, fciRestart )
        if self.verbose >= logger.DEBUG1:
            inFile = os.path.join(self.runtimeDir, self.configFile)
            logger.debug1(self, 'SHCI Input conf')
            logger.debug1(self, open(inFile, 'r').read())
        executeSHCI(self)
        if self.verbose >= logger.DEBUG1:
            outFile = os.path.join(self.runtimeDir, self.outputFile)
            logger.debug1(self, open(outFile).read())
        calc_e = readEnergy(self)

        if self.nroots==1:
            roots = 0
        else:
            roots = range(self.nroots)
        return calc_e, roots

    def restart_scheduler_(self):
        def callback(envs):
            if (envs['norm_gorb'] < self.shci_switch_tol or
                ('norm_ddm' in envs and envs['norm_ddm'] < self.shci_switch_tol*10)):
                self._restart = True
            else :
                self._restart = False
        return callback

    def spin_square(self, civec, norb, nelec):
        if isinstance(nelec, (int, numpy.integer)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else :
            neleca, nelecb = nelec
        s = (neleca - nelecb) * .5
        ss = s * (s+1)
        if isinstance(civec, int):
            return ss, s*2+1
        else:
            return [ss]*len(civec), [s*2+1]*len(civec)


def print1Int(h1,name):
 with open('%s.X'%(name), 'w') as fout:
    fout.write('%d\n'%h1[0].shape[0])
    for i in range(h1[0].shape[0]):
        for j in range(h1[0].shape[0]):
           if (abs(h1[0,i,j]) > 1.e-8):
            fout.write('%16.10g %4d %4d\n'%(h1[0,i,j], i+1, j+1))

 with open('%s.Y'%(name), 'w') as fout:
    fout.write('%d\n'%h1[1].shape[0])
    for i in range(h1[1].shape[0]):
        for j in range(h1[1].shape[0]):
           if (abs(h1[1,i,j]) > 1.e-8):
            fout.write('%16.10g %4d %4d\n'%(h1[1,i,j], i+1, j+1))

 with open('%s.Z'%(name), 'w') as fout:
    fout.write('%d\n'%h1[2].shape[0])
    for i in range(h1[2].shape[0]):
        for j in range(h1[2].shape[0]):
           if (abs(h1[2,i,j]) > 1.e-8):
            fout.write('%16.10g %4d %4d\n'%(h1[2,i,j], i+1, j+1))

 with open('%sZ'%(name), 'w') as fout:
    fout.write('%d\n'%h1[2].shape[0])
    for i in range(h1[2].shape[0]):
        for j in range(h1[2].shape[0]):
           if (abs(h1[2,i,j]) > 1.e-8):
            fout.write('%16.10g %4d %4d\n'%(h1[2,i,j], i+1, j+1))


def make_sched( SHCI ):

    nIter = len( SHCI.sweep_iter )
    # Check that the number of different epsilons match the number of iter steps.
    assert( nIter == len( SHCI.sweep_epsilon ) )

    if( nIter == 0 ):
        SHCI.sweep_iter = [0]
        SHCI.sweep_epsilon = [1.e-3]
    
#    schedStr = 'schedule '
#    for it, eps in zip( SHCI.sweep_iter, SHCI.sweep_epsilon ):
#        schedStr += '\n' + str( it ) + '\t' + str( eps )
#
#    schedStr += '\nend\n'

    #TODO: remove sweep_iter from SHCI when using Junhao's program
    schedlist = SHCI.sweep_epsilon

    return schedlist


def writeSHCIConfFile( SHCI, nelec, Restart ):
    input_vars = {'system':'chem'}
    input_vars['n_up'] = nelec[0]
    input_vars['n_dn'] = nelec[1]
    
    input_vars['eps_vars'] = [SHCI.sweep_epsilon[-1]]
    if (not Restart):
        input_vars['eps_vars_schedule'] = make_sched(SHCI)
    else:
        input_vars['eps_vars_schedule'] = [SHCI.sweep_epsilon[-1]]
    
    input_vars['var_only'] = True

    if SHCI.RDMformat != 'csv':    
        input_vars['2rdm'] = True
    else:
        input_vars['get_2rdm_csv'] = True

    # Note: For Arrow, infinite groups are used only when useExtraSymm is true
    if SHCI.groupname == 'Coov' and not SHCI.useExtraSymm:
        input_vars['chem'] = {'point_group': 'C2v'}
    elif SHCI.groupname == 'Dooh' and  not SHCI.useExtraSymm:
        input_vars['chem'] = {'point_group': 'D2h'}
    else:
        input_vars['chem'] = {'point_group': SHCI.groupname}
    
    if SHCI.TimeSymm:
        input_vars['time_sym'] = True

    if SHCI.groupname !='Coov' and SHCI.groupname != 'Dooh': 
        occ_up = []
        occ_dn = []
        for i_up in range(nelec[0]):
            occ_up.append(SHCI.orbsym[i_up])
        for i_dn in range(nelec[1]):
            occ_dn.append(SHCI.orbsym[i_dn])
    
        irrep_map = dmrg_sym.IRREP_MAP[SHCI.groupname]
        irrep_occs_up = [0]*len(irrep_map)
        irrep_occs_dn = [0]*len(irrep_map)
        for i in range(len(irrep_map)):
            irrep_occs_up[irrep_map[i]-1] = occ_up.count(i)
            irrep_occs_dn[irrep_map[i]-1] = occ_dn.count(i)
    
        input_vars['chem']['irreps'] = range(1,len(irrep_map)+1)
        input_vars['chem']['irrep_occs_up'] = irrep_occs_up
        input_vars['chem']['irrep_occs_dn'] = irrep_occs_dn

    with open('config.json','w') as outfile:
        json.dump(input_vars, outfile, indent=2) 

    try:
        check_call('rm wf_eps1* integrals_cache.dat ', shell=True)
    except:
        pass
#    if not Restart: # recalculate wavefunction
#        try:
#            check_call('rm wf_eps1* integrals_cache.dat ', shell=True)
#        except:
#            pass
#    else: # read in existing wavefunction
#        try:
#            check_call('rm integrals_cache.dat', shell=True)
#        except:
#            pass


def D2htoDinfh(SHCI, norb, nelec):
    coeffs = numpy.zeros(shape=(norb, norb)).astype(complex)
    nRows = numpy.zeros(shape=(norb,), dtype=int)
    rowIndex = numpy.zeros(shape=(2*norb,), dtype=int)
    rowCoeffs = numpy.zeros(shape=(2*norb,), dtype=float)
    orbsym1 = numpy.zeros(shape=(norb,), dtype=int)

    orbsym = numpy.asarray(SHCI.orbsym)
    A_irrep_ids = set([0,1,4,5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)

    # A1g/A2g/A1u/A2u for Dooh or A1/A2 for Coov
    for ir in A_irrep_ids:
        is_gerade = ir in (0, 1)
        for i in numpy.where(orbsym == ir)[0]:
            coeffs[i,i] = 1.0
            nRows[i] = 1
            rowIndex[2*i] = i
            rowCoeffs[2*i] = 1.
            if is_gerade:  # A1g/A2g for Dooh or A1/A2 for Coov
                orbsym1[i] = 1
            else:  # A1u/A2u for Dooh
                orbsym1[i] = 2

    # See L146 of pyscf/symm/basis.py
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir%10) in (0,2,5,7)]
    for ir in Ex_irrep_ids:
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir+1)[0]
        else:
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir-1)[0]

        if ir % 10 in (0, 5):
            l = (ir // 10) * 2
        else:
            l = (ir // 10) * 2 + 1

        for ix, iy in zip(Ex, Ey):
            nRows[ix] = nRows[iy] = 2
            if is_gerade:
                orbsym1[ix], orbsym1[iy] = 2*l+3, -(2*l+3)
            else:
                orbsym1[ix], orbsym1[iy] = 2*l+4, -(2*l+4)

            rowIndex[2*ix], rowIndex[2*ix+1] = ix, iy
            rowIndex[2*iy], rowIndex[2*iy+1] = ix, iy

            coeffs[ix,ix], coeffs[ix,iy] = ((-1)**l)*1.0/(2.0**0.5), ((-1)**l)*1.0j/(2.0**0.5)
            coeffs[iy,ix], coeffs[iy,iy] = 1.0/(2.0**0.5), -1.0j/(2.0**0.5)
            rowCoeffs[2*ix], rowCoeffs[2*ix+1] = ((-1)**l)*1.0/(2.0**0.5), ((-1)**l)*1.0/(2.0**0.5)
            rowCoeffs[2*iy], rowCoeffs[2*iy+1] = 1.0/(2.0**0.5), -1.0/(2.0**0.5)

    return coeffs, nRows, rowIndex, rowCoeffs, orbsym1


def DinfhtoD2h(SHCI, norb, nelec):
    nRows = numpy.zeros(shape=(norb,), dtype=int)
    rowIndex = numpy.zeros(shape=(2*norb,), dtype=int)
    rowCoeffs = numpy.zeros(shape=(4*norb,), dtype=float)

    orbsym = numpy.asarray(SHCI.orbsym)
    A_irrep_ids = set([0,1,4,5])
    E_irrep_ids = set(orbsym).difference(A_irrep_ids)

    for ir in A_irrep_ids:
        for i in numpy.where(orbsym == ir)[0]:
            nRows[i] = 1
            rowIndex[2*i] = i
            rowCoeffs[4*i] = 1.

    # See L146 of pyscf/symm/basis.py
    Ex_irrep_ids = [ir for ir in E_irrep_ids if (ir%10) in (0,2,5,7)]
    for ir in Ex_irrep_ids:
        is_gerade = (ir % 10) in (0, 2)
        if is_gerade:
            # See L146 of basis.py
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir+1)[0]
        else:
            Ex = numpy.where(orbsym == ir)[0]
            Ey = numpy.where(orbsym == ir-1)[0]

        if ir % 10 in (0, 5):
            l = (ir // 10) * 2
        else:
            l = (ir // 10) * 2 + 1

        for ix, iy in zip(Ex, Ey):
            nRows[ix] = nRows[iy] = 2

            rowIndex[2*ix], rowIndex[2*ix+1] = ix, iy
            rowIndex[2*iy], rowIndex[2*iy+1] = ix, iy

            rowCoeffs[4*ix], rowCoeffs[4*ix+2] = ((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)
            rowCoeffs[4*iy+1], rowCoeffs[4*iy+3] = -((-1)**l)*1.0/(2.0**0.5), 1.0/(2.0**0.5)

    return nRows, rowIndex, rowCoeffs

def writeIntegralFile(SHCI, h1eff, eri_cas, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        neleca = nelec//2 + nelec%2
        nelecb = nelec - neleca
    else :
        neleca, nelecb = nelec

    # The name of the FCIDUMP file, default is "FCIDUMP".
    integralFile = os.path.join(SHCI.runtimeDir, SHCI.integralFile)


    from pyscf import symm
    from pyscf.dmrgscf import dmrg_sym

    if (SHCI.groupname == 'Dooh' or SHCI.groupname == 'Coov') and SHCI.useExtraSymm:
       coeffs, nRows, rowIndex, rowCoeffs, orbsym = D2htoDinfh(SHCI, norb, nelec)

       newintt = numpy.tensordot(coeffs.conj(), h1eff, axes=([1],[0]))
       newint1 = numpy.tensordot(newintt, coeffs, axes=([1],[1]))
       newint1r = numpy.zeros(shape=(norb, norb), order='C')
       for i in range(norb):
          for j in range(norb):
             newint1r[i,j] = newint1[i,j].real
       int2 = pyscf.ao2mo.restore(1, eri_cas, norb)
       eri_cas = numpy.zeros_like(int2)

       transformDinfh(norb, numpy.ascontiguousarray(nRows, numpy.int32),
                      numpy.ascontiguousarray(rowIndex, numpy.int32),
                      numpy.ascontiguousarray(rowCoeffs, numpy.float64),
                      numpy.ascontiguousarray(int2, numpy.float64),
                      numpy.ascontiguousarray(eri_cas, numpy.float64))


       writeIntNoSymm(norb, numpy.ascontiguousarray(newint1r, numpy.float64),
                      numpy.ascontiguousarray(eri_cas, numpy.float64), ecore, neleca+nelecb,
                      numpy.asarray(orbsym, dtype=numpy.int32))

    else:
       if SHCI.groupname is not None and SHCI.orbsym is not []:
          orbsym = dmrg_sym.convert_orbsym(SHCI.groupname, SHCI.orbsym)
       else:
          orbsym = [1]*norb

       eri_cas = pyscf.ao2mo.restore(8, eri_cas, norb)
       # Writes the FCIDUMP file using functions in SHCI_tools.cpp.
       integralFile = integralFile.encode()  # .encode for python3 compatibility
       fcidumpFromIntegral( integralFile, h1eff, eri_cas, norb, neleca+nelecb,
                            ecore, numpy.asarray(orbsym, dtype=numpy.int32),
                            abs(neleca-nelecb) )
   
    # Fix possible errors in symmetry lables 
    if SHCI.groupname == 'Dooh' and SHCI.useExtraSymm:
#       # these lines call the python program 
#       if os.path.isfile("FCIDUMP_new"):
#          check_call('rm FCIDUMP_new', shell = True)
#       check_call('python checkLz_fcidump.py FCIDUMP', shell=True)
#       while os.path.isfile("FCIDUMP_new"):
#           check_call('mv FCIDUMP_new FCIDUMP', shell = True)
#           check_call('python checkLz_fcidump.py FCIDUMP', shell = True) 

       # these lines call the fortran program
       check_call(SHCI.Lz_relabel +' '+ integralFile, shell = True)


def executeSHCI(SHCI):
    outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
    try:
        cmd = ' '.join((SHCI.mpiprefix, SHCI.executable))
        cmd = "%s >> %s" % (cmd, outFile)
        check_call(cmd, shell=True)
    except CalledProcessError as err:
        logger.error(SHCI, cmd)
        raise err


def readEnergy(SHCI):
    with open('result.json') as result_file:
	result = json.load(result_file)
    e_var = min(result['energy_var'].values())
    return e_var

def SHCISCF(mf, norb, nelec, maxM=1000, tol=1.e-8, *args, **kwargs):
    '''Shortcut function to setup CASSCF using the SHCI solver.  The SHCI


    Examples:

    >>> mol = gto.M(atom='C 0 0 0; C 0 0 1')
    >>> mf = scf.RHF(mol).run()
    >>> mc = SHCISCF(mf, 4, 4)
    >>> mc.kernel()
    -74.414908818611522
    '''

    mc = mcscf.CASSCF(mf, norb, nelec, *args, **kwargs)
    mc.fcisolver = SHCI(mf.mol, mf.mo_coeff, mf.mo_occ, maxM, tol=tol)

    return mc


def get_hso1e(wso,x,rp):
   nb = x.shape[0]
   hso1e = numpy.zeros((3,nb,nb))
   for ic in range(3):
      hso1e[ic] = reduce(numpy.dot,(rp.T,x.T,wso[ic],x,rp))
   return hso1e

def get_wso(mol):
   nb = mol.nao_nr()
   wso = numpy.zeros((3,nb,nb))
   for iatom in range(mol.natm):
      zA  = mol.atom_charge(iatom)
      xyz = mol.atom_coord(iatom)
      mol.set_rinv_orig(xyz)
      wso += zA*mol.intor('cint1e_prinvxp_sph', 3) # sign due to integration by part                                                                            
   return wso

def get_p(dm,x,rp):
   pLL = rp.dot(dm.dot(rp.T))
   pLS = pLL.dot(x.T)
   pSS = x.dot(pLL.dot(x.T))
   return pLL,pLS,pSS

def get_fso2e_withkint(kint,x,rp,pLL,pLS,pSS):
   nb = x.shape[0]
   fso2e = numpy.zeros((3,nb,nb))
   for ic in range(3):
      gsoLL = -2.0*numpy.einsum('lmkn,lk->mn',kint[ic],pSS)
      gsoLS = -numpy.einsum('mlkn,lk->mn',kint[ic],pLS) \
              -numpy.einsum('lmkn,lk->mn',kint[ic],pLS)
      gsoSS = -2.0*numpy.einsum('mnkl,lk',kint[ic],pLL) \
              -2.0*numpy.einsum('mnlk,lk',kint[ic],pLL) \
              +2.0*numpy.einsum('mlnk,lk',kint[ic],pLL)
      fso2e[ic] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
             + x.T.dot(gsoSS.dot(x))
      fso2e[ic] = reduce(numpy.dot,(rp.T,fso2e[ic],rp))
   return fso2e

def get_kint2(mol):
   nb = mol.nao_nr()
   kint= mol.intor('int2e_spv1spv2_spinor', comp=3)
   return kint.reshape(3,nb, nb, nb, nb)

def get_fso2e(mol, x, rp, pLL, pLS, pSS):
   nb = mol.nao_nr()
   np = nb*nb
   nq = np*np
   ddint = mol.intor('int2e_ip1ip2_sph',9).reshape(3,3,nq)
   fso2e = numpy.zeros((3,nb,nb))

   ddint[0,0] = ddint[1,2]-ddint[2,1]
   kint = ddint[0,0].reshape(nb,nb,nb,nb)
   gsoLL = -2.0*numpy.einsum('lmkn,lk->mn',kint,pSS)
   gsoLS = -numpy.einsum('mlkn,lk->mn',kint,pLS) \
           -numpy.einsum('lmkn,lk->mn',kint,pLS)
   gsoSS = -2.0*numpy.einsum('mnkl,lk',kint,pLL) \
           -2.0*numpy.einsum('mnlk,lk',kint,pLL) \
           +2.0*numpy.einsum('mlnk,lk',kint,pLL)
   fso2e[0] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
               + x.T.dot(gsoSS.dot(x))
   fso2e[0] = reduce(numpy.dot,(rp.T,fso2e[0],rp))


   ddint[0,0] = ddint[2,0]-ddint[2,1]
   kint = ddint[0,0].reshape(nb,nb,nb,nb)
   gsoLL = -2.0*numpy.einsum('lmkn,lk->mn',kint,pSS)
   gsoLS = -numpy.einsum('mlkn,lk->mn',kint,pLS) \
           -numpy.einsum('lmkn,lk->mn',kint,pLS)
   gsoSS = -2.0*numpy.einsum('mnkl,lk',kint,pLL) \
           -2.0*numpy.einsum('mnlk,lk',kint,pLL) \
           +2.0*numpy.einsum('mlnk,lk',kint,pLL)
   fso2e[1] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
               + x.T.dot(gsoSS.dot(x))
   fso2e[1] = reduce(numpy.dot,(rp.T,fso2e[1],rp))

   ddint[0,0] = ddint[0,1]-ddint[1,0]
   kint = ddint[0,0].reshape(nb,nb,nb,nb)
   gsoLL = -2.0*numpy.einsum('lmkn,lk->mn',kint,pSS)
   gsoLS = -numpy.einsum('mlkn,lk->mn',kint,pLS) \
           -numpy.einsum('lmkn,lk->mn',kint,pLS)
   gsoSS = -2.0*numpy.einsum('mnkl,lk',kint,pLL) \
           -2.0*numpy.einsum('mnlk,lk',kint,pLL) \
           +2.0*numpy.einsum('mlnk,lk',kint,pLL)
   fso2e[2] = gsoLL + gsoLS.dot(x) + x.T.dot(-gsoLS.T) \
               + x.T.dot(gsoSS.dot(x))
   fso2e[2] = reduce(numpy.dot,(rp.T,fso2e[2],rp))
   return fso2e

def get_kint(mol):
   nb = mol.nao_nr()
   np = nb*nb
   nq = np*np
   ddint = mol.intor('int2e_ip1ip2_sph',9).reshape(3,3,nq)

   kint = numpy.zeros((3,nq))
   kint[0] = ddint[1,2]-ddint[2,1]# x = yz - zy                                                                                                                 
   kint[1] = ddint[2,0]-ddint[0,2]# y = zx - xz                                                                                                                 
   kint[2] = ddint[0,1]-ddint[1,0]# z = xy - yx                                                                                                              
   return kint.reshape(3,nb,nb,nb,nb)

def writeSOCIntegrals(mc, ncasorbs=None, rdm1=None, pictureChange1e="bp", pictureChange2e="bp", uncontract=True):
       from pyscf.x2c import x2c, sfx2c1e
       from pyscf.lib.parameters import LIGHT_SPEED
       LIGHT_SPEED = 137.0359895000
       alpha = 1.0/LIGHT_SPEED

       if (uncontract):
          xmol, contr_coeff = x2c.X2C().get_xmol(mc.mol)
       else:
          xmol, contr_coeff = mc.mol, numpy.eye(mc.mo_coeff.shape[0])


       rdm1ao = rdm1
       if (rdm1 is None):
          rdm1ao = 1.*mc.make_rdm1()
       if len(rdm1ao.shape) > 2 : rdm1ao = (rdm1ao[0]+rdm1ao[1])

       if (uncontract):
          dm = reduce(numpy.dot, (contr_coeff, rdm1ao, contr_coeff.T))
       else:
          dm = 1.*rdm1ao
       np, nc = contr_coeff.shape[0], contr_coeff.shape[1]

       hso1e = numpy.zeros((3, np, np))
       h1e_1c, x, rp = sfx2c1e.SpinFreeX2C(mc.mol).get_hxr(mc.mol, uncontract=uncontract)

       #two electron terms       
       if (pictureChange2e == "bp") :
          h2ao = -(alpha)**2*0.5*xmol.intor('cint2e_p1vxp1_sph', comp=3, aosym='s1')
          h2ao = h2ao.reshape(3, np, np, np, np)
          hso1e += 1.*(numpy.einsum('ijklm,lm->ijk', h2ao, dm) 
                      - 1.5*(numpy.einsum('ijklm, kl->ijm', h2ao, dm)
                             +numpy.einsum('ijklm,mj->ilk', h2ao, dm) ) )
       elif (pictureChange2e == "x2c"):
          dm1 = dm/2.
          pLL,pLS,pSS = get_p(dm1,x,rp)
          #kint = get_kint(xmol)
          #hso1e += -(alpha)**2*0.5*get_fso2e_withkint(kint,x,rp,pLL,pLS,pSS)
          hso1e += -(alpha)**2*0.5*get_fso2e(xmol,x,rp,pLL,pLS,pSS)
       elif (pictureChange2e == "none"):
          hso1e *= 0.0
       else:
          print(pictureChane2e, "not a valid option")
          exit(0)

       #MF 1 electron term
       if (pictureChange1e == "bp") :
          hso1e += (alpha)**2*0.5*get_wso(xmol)
       elif (pictureChange1e == "x2c1"):
          dm /= 2.
          pLL,pLS,pSS = get_p(dm,x,rp)
          wso = (alpha)**2*0.5*get_wso(xmol)
          hso1e += get_hso1e(wso, x, rp)
       elif (pictureChange1e == "x2cn") :
          h1e_2c = x2c.get_hcore(xmol)

          for i in range(np):
             for j in range(np):
                if ( abs(h1e_2c[2*i, 2*j+1].imag) > 1.e-8) :
                   hso1e[0][i,j] -= h1e_2c[2*i, 2*j+1].imag*2.
                if ( abs(h1e_2c[2*i, 2*j+1].real) > 1.e-8) :
                   hso1e[1][i,j] -= h1e_2c[2*i, 2*j+1].real*2.
                if ( abs(h1e_2c[2*i, 2*j].imag) > 1.e-8) :
                   hso1e[2][i,j] -= h1e_2c[2*i, 2*j].imag*2.
       else:
          print(pictureChane1e, "not a valid option")
          exit(0)

       h1ao = numpy.zeros((3, nc, nc))
       if (uncontract):
          for ic in range(3):
             h1ao[ic] = reduce(numpy.dot,(contr_coeff.T,hso1e[ic],contr_coeff))
       else:
          h1ao = 1.* hso1e

       ncore, ncas = mc.ncore, mc.ncas
       if (ncasorbs is not None):
          ncas = ncasorbs
       mo_coeff = mc.mo_coeff
       h1 = numpy.einsum('xpq,pi,qj->xij', h1ao,
                         mo_coeff, mo_coeff)[:, ncore:ncore+ncas, ncore:ncore+ncas]
       print1Int(h1, 'SOC')

def dryrun(mc, mo_coeff=None):
    '''Generate FCIDUMP and SHCI config file'''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    #bak, mc.fcisolver.onlywriteIntegral = mc.fcisolver.onlywriteIntegral, True
    mc.casci(mo_coeff)
    #mc.fcisolver.onlywriteIntegral = bak

#mc is the CASSCF object
#nroots is the number of roots that will be used to calcualte the SOC matrix
def runQDPT(mc, gtensor):
   if mc.fcisolver.__class__.__name__ == 'FakeCISolver':
       SHCI = mc.fcisolver.fcisolvers[0]
       outFile = os.path.join(SHCI.runtimeDir, SHCI.outputFile)
       writeSHCIConfFile(SHCI, mc.nelecas, False)
       confFile = os.path.join(SHCI.runtimeDir, SHCI.configFile)
       f = open(confFile, 'a')
       for SHCI2 in mc.fcisolver.fcisolvers[1:] :
          if (SHCI2.prefix != "") :
             f.write( 'prefix %s\n' %(SHCI2.prefix))
       if (gtensor):
          f.write("dogtensor\n")
       f.close()
       try:
          cmd = ' '.join((SHCI.mpiprefix, SHCI.QDPTexecutable, confFile))
          cmd = "%s > %s 2>&1" % (cmd, outFile)
          check_call(cmd, shell=True)
          check_call('cat %s|grep -v "#"'%(outFile), shell=True)
       except CalledProcessError as err:
          logger.error(mc.fcisolver, cmd)
          raise err

   else:
       writeSHCIConfFile(mc.fcisolver, mc.nelecas, False)
       confFile = os.path.join(mc.fcisolver.runtimeDir, mc.fcisolver.configFile)
       outFile = os.path.join(mc.fcisolver.runtimeDir, mc.fcisolver.outputFile)
       if (gtensor):
          f = open(confFile, 'a')
          f.write("dogtensor\n")
          f.close()
       try:
          cmd = ' '.join((mc.fcisolver.mpiprefix, mc.fcisolver.QDPTexecutable, confFile))
          cmd = "%s > %s 2>&1" % (cmd, outFile)
          check_call(cmd, shell=True)
          check_call('cat %s|grep -v "#"'%(outFile), shell=True)
       except CalledProcessError as err:
          logger.error(mc.fcisolver, cmd)
          raise err

def doSOC(mc, gtensor=False, pictureChange="bp"):
   writeSOCIntegrals(mc, pictureChange=pictureChange)
   dryrun(mc)
   if (gtensor or True):
      ncore, ncas = mc.ncore, mc.ncas
      charge_center = numpy.einsum('z,zx->x',mc.mol.atom_charges(),mc.mol.atom_coords())
      h1ao = mc.mol.intor('cint1e_cg_irxp_sph', comp=3)
      h1 = numpy.einsum('xpq,pi,qj->xij', h1ao,
                        mc.mo_coeff, mc.mo_coeff)[:, ncore:ncore+ncas, ncore:ncore+ncas]
      print1Int(h1, 'GTensor')

   runQDPT(mc, gtensor)

if __name__ == '__main__':
    from pyscf import gto, scf, mcscf, dmrgscf
    from pyscf.shciscf import shci

    # Initialize N2 molecule
    b = 1.098
    mol = gto.Mole()
    mol.build(
    verbose = 5,
    output = None,
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    )

    # Create HF molecule
    mf = scf.RHF( mol )
    mf.conv_tol = 1e-9
    mf.scf()

    # Number of orbital and electrons
    norb = 8
    nelec = 10
    dimer_atom = 'N'

    mc = mcscf.CASSCF( mf, norb, nelec )
    e_CASSCF = mc.mc2step()[0]

    # Create SHCI molecule for just variational opt.
    # Active spaces chosen to reflect valence active space.
    mch = shci.SHCISCF( mf, norb, nelec )
    mch.fcisolver.nPTiter = 0 # Turn off perturbative calc.
    mch.fcisolver.outputFile = 'no_PT.dat'
    mch.fcisolver.sweep_iter = [ 0, 3 ]
    mch.fcisolver.DoRDM = True
    # Setting large epsilon1 thresholds highlights improvement from perturbation.
    mch.fcisolver.sweep_epsilon = [ 1e-2, 1e-2 ]
    e_noPT = mch.mc1step()[0]

    # Run a single SHCI iteration with perturbative correction.
    mch.fcisolver.stochastic = False # Turns on deterministic PT calc.
    mch.fcisolver.outputFile = 'PT.dat' # Save output under different name.
    shci.writeSHCIConfFile( mch.fcisolver, [nelec/2,nelec/2] , True )
    shci.executeSHCI( mch.fcisolver )

    # Open and get the energy from the binary energy file hci.e.
    # Open and get the energy from the
    with open( mch.fcisolver.outputFile, 'r' ) as f:
    	lines = f.readlines()

    e_PT = float( lines[ len(lines) - 1 ].split()[2] )

    # e_PT = shci.readEnergy( mch.fcisolver )

    # Comparison Calculations
    del_PT = e_PT - e_noPT
    del_shci = e_CASSCF - e_PT

    print( '\n\nEnergies for %s2 give in E_h.' % dimer_atom )
    print( '=====================================' )
    print( 'SHCI Variational: %6.12f' %e_noPT )
    # Prints the total energy including the perturbative component.
    print( 'SHCI Perturbative: %6.12f' %e_PT )
    print( 'Perturbative Change: %6.12f' %del_PT )
    print( 'CASSCF Total Energy: %6.12f' %e_CASSCF )
    print( 'E(CASSCF) - E(SHCI): %6.12f' %del_shci )
