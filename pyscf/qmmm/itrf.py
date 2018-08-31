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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
QM part interface
'''

import numpy
import pyscf
from pyscf import lib
from pyscf import gto
from pyscf import df


def mm_charge(scf_method, coords, charges, unit=None):
    '''Modify the QM method using the (non-relativistic) potential generated by MM charges.

    Args:
        scf_method : a HF or DFT object

        coords : 2D array, shape (N,3)
            MM particle coordinates
        charges : 1D array
            MM particle charges
    Kwargs:
        unit : str
            Bohr, AU, Ang (case insensitive). Default is the same to mol.unit

    Returns:
        Same method object as the input scf_method with modified 1e Hamiltonian

    Note:
        1. if MM charge and X2C correction are used together, function mm_charge
        needs to be applied after X2C decoration (.x2c method), eg
        mf = mm_charge(scf.RHF(mol).x2c()), [(0.5,0.6,0.8)], [-0.5]).
        2. Once mm_charge function is applied on the SCF object, it
        affects all the post-HF calculations eg MP2, CCSD, MCSCF etc

    Examples:

    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = mm_charge(dft.RKS(mol), [(0.5,0.6,0.8)], [-0.3])
    >>> mf.kernel()
    -101.940495711284
    '''
    if unit is None:
        unit = scf_method.mol.unit
    if unit.startswith(('B','b','au','AU')):
        coords = numpy.asarray(coords, order='C')
    elif unit.startswith(('A','a')):
        coords = numpy.asarray(coords, order='C') / lib.parameters.BOHR
    else:
        coords = numpy.asarray(coords, order='C') / unit
    charges = numpy.asarray(charges)
    method_class = scf_method.__class__

    class QMMM(method_class, _QMMM):
        def __init__(self):
            self.__dict__.update(scf_method.__dict__)

        def get_hcore(self, mol=None):
            if mol is None: mol = self.mol
            if hasattr(scf_method, 'get_hcore'):
                h1e = method_class.get_hcore(self, mol)
            else:  # DO NOT modify post-HF objects to avoid the MM charges applied twice
                raise RuntimeError('mm_charge function cannot be applied on post-HF methods')

            if pyscf.DEBUG:
                v = 0
                for i,q in enumerate(charges):
                    mol.set_rinv_origin(coords[i])
                    v += mol.intor('int1e_rinv') * -q
            else:
                fakemol = gto.fakemol_for_charges(coords)
                if mol.cart:
                    intor = 'int3c2e_cart'
                else:
                    intor = 'int3c2e_sph'
                j3c = df.incore.aux_e2(mol, fakemol, intor=intor, aosym='s2ij')
                v = lib.unpack_tril(numpy.einsum('xk,k->x', j3c, -charges))
            return h1e + v

        def energy_nuc(self):
# nuclei lattice interaction
            nuc = self.mol.energy_nuc()
            for j in range(self.mol.natm):
                q2, r2 = self.mol.atom_charge(j), self.mol.atom_coord(j)
                r = lib.norm(r2-coords, axis=1)
                nuc += q2*(charges/r).sum()
            return nuc

        def nuc_grad_method(self):
            scf_grad = method_class.nuc_grad_method(self)
            return mm_charge_grad(scf_grad, coords, charges, 'Bohr')

    return QMMM()

def mm_charge_grad(scf_grad, coords, charges, unit=None):
    '''Apply the MM charges in the QM gradients' method.  It affects both the
    electronic and nuclear parts of the QM fragment.

    Args:
        scf_grad : a HF or DFT gradient object (grad.HF or grad.RKS etc)
            Once mm_charge_grad function is applied on the SCF object,
            it affects all post-HF calculations eg MP2, CCSD, MCSCF etc
        coords : 2D array, shape (N,3)
            MM particle coordinates
        charges : 1D array
            MM particle charges
    Kwargs:
        unit : str
            Bohr, AU, Ang (case insensitive). Default is the same to mol.unit

    Returns:
        Same gradeints method object as the input scf_grad method

    Examples:

    >>> from pyscf import gto, scf, grad
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1', basis='ccpvdz', verbose=0)
    >>> mf = mm_charge(scf.RHF(mol), [(0.5,0.6,0.8)], [-0.3])
    >>> mf.kernel()
    -101.940495711284
    >>> hfg = mm_charge_grad(grad.hf.RHF(mf), coords, charges)
    >>> hfg.kernel()
    [[-0.25912357 -0.29235976 -0.38245077]
     [-1.70497052 -1.89423883  1.2794798 ]]
    '''
    if unit is None:
        unit = scf_grad.mol.unit
    if unit.startswith(('B','b','au','AU')):
        coords = numpy.asarray(coords, order='C')
    elif unit.startswith(('A','a')):
        coords = numpy.asarray(coords, order='C') / lib.parameters.BOHR
    else:
        coords = numpy.asarray(coords, order='C') / unit
    charges = numpy.asarray(charges)

    class QMMM(scf_grad.__class__, _QMMMGrad):
        def __init__(self):
            self.__dict__.update(scf_grad.__dict__)

        def get_hcore(self, mol=None):
            ''' (QM 1e grad) + <-d/dX i|q_mm/r_mm|j>'''
            if mol is None: mol = self.mol
            g_qm = scf_grad.get_hcore(mol)
            nao = g_qm.shape[1]
            if pyscf.DEBUG:
                v = 0
                for i,q in enumerate(charges):
                    mol.set_rinv_origin(coords[i])
                    v += mol.intor('int1e_iprinv', comp=3) * q
            else:
                fakemol = gto.fakemol_for_charges(coords)
                j3c = df.incore.aux_e2(mol, fakemol, intor='int3c2e_ip1',
                                       aosym='s1', comp=3)
                v = numpy.einsum('ipqk,k->ipq', j3c, charges)
            return scf_grad.get_hcore(mol) + v

        def grad_nuc(self, mol=None, atmlst=None):
            if mol is None: mol = scf_grad.mol
            g_qm = scf_grad.grad_nuc(mol, atmlst)
# nuclei lattice interaction
            g_mm = numpy.empty((mol.natm,3))
            for i in range(mol.natm):
                q1 = mol.atom_charge(i)
                r1 = mol.atom_coord(i)
                r = lib.norm(r1-coords, axis=1)
                g_mm[i] = -q1 * numpy.einsum('i,ix,i->x', charges, r1-coords, 1/r**3)
            if atmlst is not None:
                g_mm = g_mm[atmlst]
            return g_qm + g_mm
    return QMMM()

# A tag to label the derived class
class _QMMM:
    pass
class _QMMMGrad:
    pass

if __name__ == '__main__':
    from pyscf import scf, cc, grad
    mol = gto.Mole()
    mol.atom = ''' O                  0.00000000    0.00000000   -0.11081188
                   H                 -0.00000000   -0.84695236    0.59109389
                   H                 -0.00000000    0.89830571    0.52404783 '''
    mol.basis = 'cc-pvdz'
    mol.build()

    coords = [(0.5,0.6,0.8)]
    #coords = [(0.0,0.0,0.0)]
    charges = [-0.5]
    mf = mm_charge(scf.RHF(mol), coords, charges)
    print(mf.kernel()) # -76.3206550372
    mycc = cc.ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    ecc, t1, t2 = mycc.kernel() # ecc = -0.228939687075
    l1, l2 = mycc.solve_lambda()[1:]

    hfg = mm_charge_grad(grad.RHF(mf), coords, charges)
    g1 = grad.ccsd.kernel(mycc, t1, t2, l1, l2, mf_grad=hfg)
    print(g1)
# [[-0.50179182 -0.61489747 -0.96396085]
#  [-0.077348   -0.23457358  0.02839861]
#  [-0.44182244  0.14559838 -0.22348068]]

