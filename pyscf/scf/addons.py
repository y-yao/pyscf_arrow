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
# Authors: Qiming Sun <osirpt.sun@gmail.com>
#          Junzi Liu <latrix1247@gmail.com>
#

import copy
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.gto import mole
from pyscf.lib import logger
from pyscf.scf import hf


def frac_occ_(mf, tol=1e-3):
    from pyscf.scf import uhf, rohf
    old_get_occ = mf.get_occ
    mol = mf.mol

    def guess_occ(mo_energy, nocc):
        sorted_idx = numpy.argsort(mo_energy)
        homo = mo_energy[sorted_idx[nocc-1]]
        lumo = mo_energy[sorted_idx[nocc]]
        frac_occ_lst = abs(mo_energy - homo) < tol
        integer_occ_lst = (mo_energy <= homo) & (~frac_occ_lst)
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[integer_occ_lst] = 1
        degen = numpy.count_nonzero(frac_occ_lst)
        frac = nocc - numpy.count_nonzero(integer_occ_lst)
        mo_occ[frac_occ_lst] = float(frac) / degen
        return mo_occ, numpy.where(frac_occ_lst)[0], homo, lumo

    get_grad = None

    if isinstance(mf, uhf.UHF):
        def get_occ(mo_energy, mo_coeff=None):
            nocca, noccb = mol.nelec
            mo_occa, frac_lsta, homoa, lumoa = guess_occ(mo_energy[0], nocca)
            mo_occb, frac_lstb, homob, lumob = guess_occ(mo_energy[1], noccb)

            if abs(homoa - lumoa) < tol or abs(homob - lumob) < tol:
                mo_occ = numpy.array([mo_occa, mo_occb])
                logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  '
                            '%6g for beta orbitals %s',
                            mo_occa[frac_lsta[0]], frac_lsta,
                            mo_occb[frac_lstb[0]], frac_lstb)
                logger.info(mf, '  alpha HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                logger.info(mf, '  beta  HOMO = %.12g  LUMO = %.12g', homob, lumob)
                logger.debug(mf, '  alpha mo_energy = %s', mo_energy[0])
                logger.debug(mf, '  beta  mo_energy = %s', mo_energy[1])
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

    elif isinstance(mf, rohf.ROHF):
        def get_occ(mo_energy, mo_coeff=None):
            nocca, noccb = mol.nelec
            mo_occa, frac_lsta, homoa, lumoa = guess_occ(mo_energy, nocca)
            mo_occb, frac_lstb, homob, lumob = guess_occ(mo_energy, noccb)

            if abs(homoa - lumoa) < tol or abs(homob - lumob) < tol:
                mo_occ = mo_occa + mo_occb
                logger.warn(mf, 'fraction occ = %6g for alpha orbitals %s  '
                            '%6g for beta orbitals %s',
                            mo_occa[frac_lsta[0]], frac_lsta,
                            mo_occb[frac_lstb[0]], frac_lstb)
                logger.info(mf, '  HOMO = %.12g  LUMO = %.12g', homoa, lumoa)
                logger.debug(mf, '  mo_energy = %s', mo_energy)
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

        def get_grad(mo_coeff, mo_occ, fock):
            occidxa = mo_occ > 0
            occidxb = mo_occ > 1
            viridxa = ~occidxa
            viridxb = ~occidxb
            uniq_var_a = viridxa.reshape(-1,1) & occidxa
            uniq_var_b = viridxb.reshape(-1,1) & occidxb

            if hasattr(fock, 'focka'):
                focka = fock.focka
                fockb = fock.fockb
            elif getattr(fock, 'ndim', None) == 3:
                focka, fockb = fock
            else:
                focka = fockb = fock
            focka = reduce(numpy.dot, (mo_coeff.T.conj(), focka, mo_coeff))
            fockb = reduce(numpy.dot, (mo_coeff.T.conj(), fockb, mo_coeff))

            g = numpy.zeros_like(focka)
            g[uniq_var_a]  = focka[uniq_var_a]
            g[uniq_var_b] += fockb[uniq_var_b]
            return g[uniq_var_a | uniq_var_b]

    else:  # RHF
        def get_occ(mo_energy, mo_coeff=None):
            nocc = (mol.nelectron+1) // 2  # n_docc + n_socc
            mo_occ, frac_lst, homo, lumo = guess_occ(mo_energy, nocc)
            n_docc = mol.nelectron // 2
            n_socc = nocc - n_docc
            if abs(homo - lumo) < tol or n_socc:
                mo_occ *= 2
                degen = len(frac_lst)
                mo_occ[frac_lst] -= float(n_socc) / degen
                logger.warn(mf, 'fraction occ = %6g  for orbitals %s',
                            mo_occ[frac_lst[0]], frac_lst)
                logger.info(mf, 'HOMO = %.12g  LUMO = %.12g', homo, lumo)
                logger.debug(mf, '  mo_energy = %s', mo_energy)
            else:
                mo_occ = old_get_occ(mo_energy, mo_coeff)
            return mo_occ

    mf.get_occ = get_occ
    if get_grad is not None:
        mf.get_grad = get_grad
    return mf
frac_occ = frac_occ_

def dynamic_occ_(mf, tol=1e-3):
    assert(isinstance(mf, hf.RHF))
    old_get_occ = mf.get_occ
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        nocc = mol.nelectron // 2
        sort_mo_energy = numpy.sort(mo_energy)
        lumo = sort_mo_energy[nocc]
        if abs(sort_mo_energy[nocc-1] - lumo) < tol:
            mo_occ = numpy.zeros_like(mo_energy)
            mo_occ[mo_energy<lumo] = 2
            lst = abs(mo_energy - lumo) < tol
            mo_occ[lst] = 0
            logger.warn(mf, 'set charge = %d', mol.charge+int(lst.sum())*2)
            logger.info(mf, 'HOMO = %.12g  LUMO = %.12g',
                        sort_mo_energy[nocc-1], sort_mo_energy[nocc])
            logger.debug(mf, '  mo_energy = %s', sort_mo_energy)
        else:
            mo_occ = old_get_occ(mo_energy, mo_coeff)
        return mo_occ
    mf.get_occ = get_occ
    return mf
dynamic_occ = dynamic_occ_

def dynamic_level_shift_(mf, factor=1.):
    '''Dynamically change the level shift in each SCF cycle.  The level shift
    value is set to (HF energy change * factor)
    '''
    old_get_fock = mf.get_fock
    last_e = [None]
    def get_fock(h1e, s1e, vhf, dm, cycle=-1, diis=None,
                 diis_start_cycle=None, level_shift_factor=None, damp_factor=None):
        if cycle >= 0 or diis is not None:
            ehf =(numpy.einsum('ij,ji', h1e, dm) +
                  numpy.einsum('ij,ji', vhf, dm) * .5)
            if last_e[0] is not None:
                level_shift_factor = abs(ehf-last_e[0]) * factor
                logger.info(mf, 'Set level shift to %g', level_shift_factor)
            last_e[0] = ehf
        return old_get_fock(h1e, s1e, vhf, dm, cycle, diis, diis_start_cycle,
                            level_shift_factor, damp_factor)
    mf.get_fock = get_fock
    return mf
dynamic_level_shift = dynamic_level_shift_

def float_occ_(mf):
    '''
    For UHF, allowing the Sz value being changed during SCF iteration.
    Determine occupation of alpha and beta electrons based on energy spectrum
    '''
    from pyscf.scf import uhf
    assert(isinstance(mf, uhf.UHF))
    def get_occ(mo_energy, mo_coeff=None):
        mol = mf.mol
        ee = numpy.sort(numpy.hstack(mo_energy))
        n_a = numpy.count_nonzero(mo_energy[0]<(ee[mol.nelectron-1]+1e-3))
        n_b = mol.nelectron - n_a
        if mf.nelec is None:
            nelec = mf.mol.nelec
        else:
            nelec = mf.nelec
        if n_a != nelec[0]:
            logger.info(mf, 'change num. alpha/beta electrons '
                        ' %d / %d -> %d / %d',
                        nelec[0], nelec[1], n_a, n_b)
            mf.nelec = (n_a, n_b)
        return uhf.UHF.get_occ(mf, mo_energy, mo_coeff)
    mf.get_occ = get_occ
    return mf
dynamic_sz_ = float_occ = float_occ_

def follow_state_(mf, occorb=None):
    occstat = [occorb]
    old_get_occ = mf.get_occ
    def get_occ(mo_energy, mo_coeff=None):
        if occstat[0] is None:
            mo_occ = old_get_occ(mo_energy, mo_coeff)
        else:
            mo_occ = numpy.zeros_like(mo_energy)
            s = reduce(numpy.dot, (occstat[0].T, mf.get_ovlp(), mo_coeff))
            nocc = mf.mol.nelectron // 2
            #choose a subset of mo_coeff, which maximizes <old|now>
            idx = numpy.argsort(numpy.einsum('ij,ij->j', s, s))
            mo_occ[idx[-nocc:]] = 2
            logger.debug(mf, '  mo_occ = %s', mo_occ)
            logger.debug(mf, '  mo_energy = %s', mo_energy)
        occstat[0] = mo_coeff[:,mo_occ>0]
        return mo_occ
    mf.get_occ = get_occ
    return mf
follow_state = follow_state_

def mom_occ_(mf, occorb, setocc):
    '''Use maximum overlap method to determine occupation number for each orbital in every
    iteration. It can be applied to unrestricted HF/KS and restricted open-shell
    HF/KS.'''
    from pyscf.scf import uhf, rohf
    if isinstance(mf, uhf.UHF):
        coef_occ_a = occorb[0][:, setocc[0]>0]
        coef_occ_b = occorb[1][:, setocc[1]>0]
    elif isinstance(mf, rohf.ROHF):
        if mf.mol.spin != (numpy.sum(setocc[0]) - numpy.sum(setocc[1])):
            raise ValueError('Wrong occupation setting for restricted open-shell calculation.')
        coef_occ_a = occorb[:, setocc[0]>0]
        coef_occ_b = occorb[:, setocc[1]>0]
    else:
        raise RuntimeError('Cannot support this class of instance %s' % mf)
    log = logger.Logger(mf.stdout, mf.verbose)
    def get_occ(mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_coeff is None: mo_coeff = mf.mo_coeff
        if isinstance(mf, rohf.ROHF): mo_coeff = numpy.array([mo_coeff, mo_coeff])
        mo_occ = numpy.zeros_like(setocc)
        nocc_a = int(numpy.sum(setocc[0]))
        nocc_b = int(numpy.sum(setocc[1]))
        s_a = reduce(numpy.dot, (coef_occ_a.T, mf.get_ovlp(), mo_coeff[0]))
        s_b = reduce(numpy.dot, (coef_occ_b.T, mf.get_ovlp(), mo_coeff[1]))
        #choose a subset of mo_coeff, which maximizes <old|now>
        idx_a = numpy.argsort(numpy.einsum('ij,ij->j', s_a, s_a))[::-1]
        idx_b = numpy.argsort(numpy.einsum('ij,ij->j', s_b, s_b))[::-1]
        mo_occ[0][idx_a[:nocc_a]] = 1.
        mo_occ[1][idx_b[:nocc_b]] = 1.

        log.debug(' New alpha occ pattern: %s', mo_occ[0])
        log.debug(' New beta occ pattern: %s', mo_occ[1])
        if isinstance(mf.mo_energy, numpy.ndarray) and mf.mo_energy.ndim == 1:
            log.debug1(' Current mo_energy(sorted) = %s', mo_energy)
        else:
            log.debug1(' Current alpha mo_energy(sorted) = %s', mo_energy[0])
            log.debug1(' Current beta mo_energy(sorted) = %s', mo_energy[1])

        if (int(numpy.sum(mo_occ[0])) != nocc_a):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                      nocc_a, int(numpy.sum(mo_occ[0])))
        if (int(numpy.sum(mo_occ[1])) != nocc_b):
            log.error('mom alpha electron occupation numbers do not match: %d, %d',
                      nocc_b, int(numpy.sum(mo_occ[1])))

        #output 1-dimension occupation number for restricted open-shell
        if isinstance(mf, rohf.ROHF): mo_occ = mo_occ[0, :] + mo_occ[1, :]
        return mo_occ
    mf.get_occ = get_occ
    return mf
mom_occ = mom_occ_

def project_mo_nr2nr(mol1, mo1, mol2):
    r''' Project orbital coefficients from basis set 1 (C1 for mol1) to basis
    set 2 (C2 for mol2).

    .. math::

        |\psi1\rangle = |AO1\rangle C1

        |\psi2\rangle = P |\psi1\rangle = |AO2\rangle S^{-1}\langle AO2| AO1\rangle> C1 = |AO2\rangle> C2

        C2 = S^{-1}\langle AO2|AO1\rangle C1

    There are three relevant functions:
    :func:`project_mo_nr2nr` is the projection for non-relativistic (scalar) basis.
    :func:`project_mo_nr2r` projects from non-relativistic to relativistic basis.
    :func:`project_mo_r2r`  is the projection between relativistic (spinor) basis.
    '''
    s22 = mol2.intor_symmetric('int1e_ovlp')
    s21 = mole.intor_cross('int1e_ovlp', mol2, mol1)
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        return lib.cho_solve(s22, numpy.dot(s21, mo1))
    else:
        return [lib.cho_solve(s22, numpy.dot(s21, x)) for x in mo1]

def project_mo_nr2r(mol1, mo1, mol2):
    __doc__ = project_mo_nr2nr.__doc__

    assert(not mol1.cart)
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_sph', mol2, mol1)

    ua, ub = mol2.sph2spinor_coeff()
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        mo2 = numpy.dot(s21, mo1)
        return lib.cho_solve(s22, mo2)
    else:
        return [lib.cho_solve(s22, numpy.dot(s21, x)) for x in mo1]

def project_mo_r2r(mol1, mo1, mol2):
    __doc__ = project_mo_nr2nr.__doc__

    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    t22 = mol2.intor_symmetric('int1e_spsp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_spinor', mol2, mol1)
    t21 = mole.intor_cross('int1e_spsp_spinor', mol2, mol1)
    n2c = s21.shape[1]
    pl = lib.cho_solve(s22, s21)
    ps = lib.cho_solve(t22, t21)
    if isinstance(mo1, numpy.ndarray) and mo1.ndim == 2:
        return numpy.vstack((numpy.dot(pl, mo1[:n2c]),
                             numpy.dot(ps, mo1[n2c:])))
    else:
        return [numpy.vstack((numpy.dot(pl, x[:n2c]),
                              numpy.dot(ps, x[n2c:]))) for x in mo1]

def project_dm_nr2nr(mol1, dm1, mol2):
    r''' Project density matrix representation from basis set 1 (mol1) to basis
    set 2 (mol2).

    .. math::

        |AO2\rangle DM_AO2 \langle AO2|

        = |AO2\rangle P DM_AO1 P \langle AO2|

        DM_AO2 = P DM_AO1 P

        P = S_{AO2}^{-1}\langle AO2|AO1\rangle

    There are three relevant functions:
    :func:`project_dm_nr2nr` is the projection for non-relativistic (scalar) basis.
    :func:`project_dm_nr2r` projects from non-relativistic to relativistic basis.
    :func:`project_dm_r2r`  is the projection between relativistic (spinor) basis.
    '''
    s22 = mol2.intor_symmetric('int1e_ovlp')
    s21 = mole.intor_cross('int1e_ovlp', mol2, mol1)
    p21 = lib.cho_solve(s22, s21)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())

def project_dm_nr2r(mol1, dm1, mol2):
    __doc__ = project_dm_nr2nr.__doc__

    assert(not mol1.cart)
    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_sph', mol2, mol1)

    ua, ub = mol2.sph2spinor_coeff()
    s21 = numpy.dot(ua.T.conj(), s21) + numpy.dot(ub.T.conj(), s21) # (*)
    # mo2: alpha, beta have been summed in Eq. (*)
    # so DM = mo2[:,:nocc] * 1 * mo2[:,:nocc].H
    p21 = lib.cho_solve(s22, s21)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())

def project_dm_r2r(mol1, dm1, mol2):
    __doc__ = project_dm_nr2nr.__doc__

    s22 = mol2.intor_symmetric('int1e_ovlp_spinor')
    t22 = mol2.intor_symmetric('int1e_spsp_spinor')
    s21 = mole.intor_cross('int1e_ovlp_spinor', mol2, mol1)
    t21 = mole.intor_cross('int1e_spsp_spinor', mol2, mol1)
    n2c = s21.shape[1]
    pl = lib.cho_solve(s22, s21)
    ps = lib.cho_solve(t22, t21)
    p21 = scipy.linalg.block_diag(pl, ps)
    if isinstance(dm1, numpy.ndarray) and dm1.ndim == 2:
        return reduce(numpy.dot, (p21, dm1, p21.conj().T))
    else:
        return lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())


def remove_linear_dep_(mf, threshold=1e-8, lindep=1e-10):
    '''
    Args:
        threshold : float
            The threshold under which the eigenvalues of the overlap matrix are
            discarded to avoid numerical instability.
        lindep : float
            The threshold that triggers the special treatment of the linear
            dependence issue.
    '''
    s = mf.get_ovlp()
    cond = numpy.max(lib.cond(s))
    if cond < 1./lindep:
        return mf

    logger.info(mf, 'Applying remove_linear_dep_ on SCF obejct.')
    logger.debug(mf, 'Overlap condition number %g', cond)
    def eigh(h, s):
        d, t = numpy.linalg.eigh(s)
        x = t[:,d>threshold] / numpy.sqrt(d[d>threshold])
        xhx = reduce(numpy.dot, (x.T.conj(), h, x))
        e, c = numpy.linalg.eigh(xhx)
        c = numpy.dot(x, c)
        return e, c
    mf._eigh = eigh
    return mf
remove_linear_dep = remove_linear_dep_

def convert_to_uhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the unrestricted HF/KS object

    Note if mf is an second order SCF object, the second order object will not
    be converted (in other words, only the underlying SCF object will be
    converted)

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An unrestricted SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, hf.SCF))

    logger.debug(mf, 'Converting %s to UHF', mf.__class__)

    def update_mo_(mf, mf1):
        if mf.mo_energy is not None:
            if isinstance(mf, scf.uhf.UHF):
                mf1.mo_occ = mf.mo_occ
                mf1.mo_coeff = mf.mo_coeff
                mf1.mo_energy = mf.mo_energy
            elif not hasattr(mf, 'kpts'):  # UHF
                mf1.mo_occ = numpy.array((mf.mo_occ>0, mf.mo_occ==2), dtype=numpy.double)
                mf1.mo_energy = (mf.mo_energy, mf.mo_energy)
                mf1.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
            else:  # This to handle KRHF object
                mf1.mo_occ = ([numpy.asarray(occ> 0, dtype=numpy.double)
                               for occ in mf.mo_occ],
                              [numpy.asarray(occ==2, dtype=numpy.double)
                               for occ in mf.mo_occ])
                mf1.mo_energy = (mf.mo_energy, mf.mo_energy)
                mf1.mo_coeff = (mf.mo_coeff, mf.mo_coeff)
        return mf1

    if isinstance(mf, scf.ghf.GHF):
        raise NotImplementedError

    elif out is not None:
        assert(isinstance(out, scf.uhf.UHF))
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif isinstance(mf, scf.uhf.UHF):
# Remove with_df for SOSCF method because the post-HF code checks the
# attribute .with_df to identify whether an SCF object is DF-SCF method.
# with_df in SOSCF is used in orbital hessian approximation only.  For the
# returned SCF object, whehter with_df exists in SOSCF has no effects on the
# mean-field energy and other properties.
        if hasattr(mf, '_scf'):
            return _update_mf_without_soscf(mf, copy.copy(mf._scf), remove_df)
        else:
            return copy.copy(mf)

    else:
        known_cls = {scf.hf.RHF        : scf.uhf.UHF,
                     scf.rohf.ROHF     : scf.uhf.UHF,
                     scf.hf_symm.RHF   : scf.uhf_symm.UHF,
                     scf.hf_symm.ROHF  : scf.uhf_symm.UHF,
                     dft.rks.RKS       : dft.uks.UKS,
                     dft.roks.ROKS     : dft.uks.UKS,
                     dft.rks_symm.RKS  : dft.uks_symm.UKS,
                     dft.rks_symm.ROKS : dft.uks_symm.UKS}
        out = _object_without_soscf(mf, known_cls, remove_df)

    return update_mo_(mf, out)

def _object_without_soscf(mf, known_class, remove_df=False):
    sub_classes = []
    obj = None
    for i, cls in enumerate(mf.__class__.__mro__):
        if cls in known_class:
            obj = known_class[cls](mf.mol)
            break
        else:
            sub_classes.append(cls)

    if obj is None:
        raise NotImplementedError(
            "Incompatible object types. Mean-field `mf` class not found in "
            "`known_class` type.\n\nmf = '%s'\n\nknown_class = '%s'" %
            (mf.__class__.__mro__, known_class))

# Mimic the initialization procedure to restore the Hamiltonian
    for cls in reversed(sub_classes):
        if (not remove_df) and 'DFHF' in cls.__name__:
            obj = obj.density_fit()
        elif 'SecondOrder' in cls.__name__:
# SOSCF is not a necessary part
            # obj = obj.newton()
            remove_df = remove_df or (not hasattr(mf._scf, 'with_df'))
        elif 'SFX2C1E' in cls.__name__:
            obj = obj.sfx2c1e()
    return _update_mf_without_soscf(mf, obj, remove_df)

def _update_mf_without_soscf(mf, out, remove_df=False):
    mf_dic = dict(mf.__dict__)

    # For SOSCF, avoid the old _scf to be copied to the new object
    if '_scf' in mf_dic:
        mf_dic.pop('_scf')
        mf_dic.pop('with_df', None)

    out.__dict__.update(mf_dic)

    if remove_df and hasattr(out, 'with_df'):
        delattr(out, 'with_df')
    return out

def convert_to_rhf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the restricted HF/KS object

    Note if mf is an second order SCF object, the second order object will not
    be converted (in other words, only the underlying SCF object will be
    converted)

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An unrestricted SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, hf.SCF))

    logger.debug(mf, 'Converting %s to RHF', mf.__class__)

    def update_mo_(mf, mf1):
        if mf.mo_energy is not None:
            if isinstance(mf, scf.hf.RHF): # RHF/ROHF/KRHF/KROHF
                mf1.mo_occ = mf.mo_occ
                mf1.mo_coeff = mf.mo_coeff
                mf1.mo_energy = mf.mo_energy
            elif not hasattr(mf, 'kpts'):  # UHF
                mf1.mo_occ = mf.mo_occ[0] + mf.mo_occ[1]
                mf1.mo_energy = mf.mo_energy[0]
                mf1.mo_coeff =  mf.mo_coeff[0]
                if hasattr(mf.mo_coeff[0], 'orbsym'):
                    mf1.mo_coeff = lib.tag_array(mf1.mo_coeff, orbsym=mf.mo_coeff[0].orbsym)
            else:  # KUHF
                mf1.mo_occ = [occa+occb for occa, occb in zip(*mf.mo_occ)]
                mf1.mo_energy = mf.mo_energy[0]
                mf1.mo_coeff =  mf.mo_coeff[0]
        return mf1

    if getattr(mf, 'nelec', None) is None:
        nelec = mf.mol.nelec
    else:
        nelec = mf.nelec

    if isinstance(mf, scf.ghf.GHF):
        raise NotImplementedError

    elif out is not None:
        assert(isinstance(out, scf.hf.RHF))
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif (isinstance(mf, scf.hf.RHF) or
          (nelec[0] != nelec[1] and isinstance(mf, scf.rohf.ROHF))):
        if hasattr(mf, '_scf'):
            return _update_mf_without_soscf(mf, copy.copy(mf._scf), remove_df)
        else:
            return copy.copy(mf)

    else:
        if nelec[0] == nelec[1]:
            known_cls = {scf.uhf.UHF      : scf.hf.RHF      ,
                         scf.uhf_symm.UHF : scf.hf_symm.RHF ,
                         dft.uks.UKS      : dft.rks.RKS     ,
                         dft.uks_symm.UKS : dft.rks_symm.RKS,
                         scf.rohf.ROHF    : scf.hf.RHF      ,
                         scf.hf_symm.ROHF : scf.hf_symm.RHF ,
                         dft.roks.ROKS    : dft.rks.RKS     ,
                         dft.rks_symm.ROKS: dft.rks_symm.RKS}
        else:
            known_cls = {scf.uhf.UHF      : scf.rohf.ROHF    ,
                         scf.uhf_symm.UHF : scf.hf_symm.ROHF ,
                         dft.uks.UKS      : dft.roks.ROKS    ,
                         dft.uks_symm.UKS : dft.rks_symm.ROKS}
        out = _object_without_soscf(mf, known_cls, remove_df)

    return update_mo_(mf, out)

def convert_to_ghf(mf, out=None, remove_df=False):
    '''Convert the given mean-field object to the generalized HF/KS object

    Note if mf is an second order SCF object, the second order object will not
    be converted (in other words, only the underlying SCF object will be
    converted)

    Args:
        mf : SCF object

    Kwargs
        remove_df : bool
            Whether to convert the DF-SCF object to the normal SCF object.
            This conversion is not applied by default.

    Returns:
        An generalized SCF object
    '''
    from pyscf import scf
    from pyscf import dft
    from pyscf.soscf import newton_ah
    assert(isinstance(mf, hf.SCF))

    logger.debug(mf, 'Converting %s to GHF', mf.__class__)

    def update_mo_(mf, mf1):
        if mf.mo_energy is not None:
            if isinstance(mf, scf.hf.RHF): # RHF
                nao, nmo = mf.mo_coeff.shape
                orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, True)

                mf1.mo_energy = numpy.empty(nmo*2)
                mf1.mo_energy[orbspin==0] = mf.mo_energy
                mf1.mo_energy[orbspin==1] = mf.mo_energy
                mf1.mo_occ = numpy.empty(nmo*2)
                mf1.mo_occ[orbspin==0] = mf.mo_occ > 0
                mf1.mo_occ[orbspin==1] = mf.mo_occ == 2

                mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mf.mo_coeff.dtype)
                mo_coeff[:nao,orbspin==0] = mf.mo_coeff
                mo_coeff[nao:,orbspin==1] = mf.mo_coeff
                if hasattr(mf.mo_coeff, 'orbsym'):
                    orbsym = numpy.zeros_like(orbspin)
                    orbsym[orbspin==0] = mf.mo_coeff.orbsym
                    orbsym[orbspin==1] = mf.mo_coeff.orbsym
                    mo_coeff = lib.tag_array(mo_coeff, orbsym=orbsym)
                mf1.mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)

            else: # UHF
                nao, nmo = mf.mo_coeff[0].shape
                orbspin = get_ghf_orbspin(mf.mo_energy, mf.mo_occ, False)

                mf1.mo_energy = numpy.empty(nmo*2)
                mf1.mo_energy[orbspin==0] = mf.mo_energy[0]
                mf1.mo_energy[orbspin==1] = mf.mo_energy[1]
                mf1.mo_occ = numpy.empty(nmo*2)
                mf1.mo_occ[orbspin==0] = mf.mo_occ[0]
                mf1.mo_occ[orbspin==1] = mf.mo_occ[1]

                mo_coeff = numpy.zeros((nao*2,nmo*2), dtype=mf.mo_coeff[0].dtype)
                mo_coeff[:nao,orbspin==0] = mf.mo_coeff[0]
                mo_coeff[nao:,orbspin==1] = mf.mo_coeff[1]
                if hasattr(mf.mo_coeff[0], 'orbsym'):
                    orbsym = numpy.zeros_like(orbspin)
                    orbsym[orbspin==0] = mf.mo_coeff[0].orbsym
                    orbsym[orbspin==1] = mf.mo_coeff[1].orbsym
                    mo_coeff = lib.tag_array(mo_coeff, orbsym=orbsym)
                mf1.mo_coeff = lib.tag_array(mo_coeff, orbspin=orbspin)
        return mf1

    if out is not None:
        assert(isinstance(out, scf.ghf.GHF))
        out = _update_mf_without_soscf(mf, out, remove_df)

    elif isinstance(mf, scf.ghf.GHF):
        if hasattr(mf, '_scf'):
            return _update_mf_without_soscf(mf, copy.copy(mf._scf), remove_df)
        else:
            return copy.copy(mf)

    else:
        known_cls = {scf.hf.RHF        : scf.ghf.GHF,
                     scf.rohf.ROHF     : scf.ghf.GHF,
                     scf.uhf.UHF       : scf.ghf.GHF,
                     scf.hf_symm.RHF   : scf.ghf_symm.GHF,
                     scf.hf_symm.ROHF  : scf.ghf_symm.GHF,
                     scf.uhf_symm.UHF  : scf.ghf_symm.GHF,
                     dft.rks.RKS       : None,
                     dft.roks.ROKS     : None,
                     dft.uks.UKS       : None,
                     dft.rks_symm.RKS  : None,
                     dft.rks_symm.ROKS : None,
                     dft.uks_symm.UKS  : None}
        out = _object_without_soscf(mf, known_cls, remove_df)

    return update_mo_(mf, out)

def get_ghf_orbspin(mo_energy, mo_occ, is_rhf=None):
    '''Spin of each GHF orbital when the GHF orbitals are converted from
    RHF/UHF orbitals

    For RHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.  In the occupied orbital space, if degenerated, first
    alpha then beta, last the (open-shell) singly occupied (alpha) orbitals. In
    the unoccupied orbital space, first the (open-shell) unoccupied (beta)
    orbitals if applicable, then alpha and beta orbitals

    For UHF orbitals, the orbspin corresponds to first occupied orbitals then
    unoccupied orbitals.
    '''
    if is_rhf is None:  # guess whether the orbitals are RHF orbitals
        is_rhf = mo_energy[0].ndim == 0

    if is_rhf:
        nmo = mo_energy.size
        nocc = numpy.count_nonzero(mo_occ >0)
        nvir = nmo - nocc
        ndocc = numpy.count_nonzero(mo_occ==2)
        nsocc = nocc - ndocc
        orbspin = numpy.array([0,1]*ndocc + [0]*nsocc + [1]*nsocc + [0,1]*nvir)
    else:
        nmo = mo_energy[0].size
        nocca = numpy.count_nonzero(mo_occ[0]>0)
        nvira = nmo - nocca
        noccb = numpy.count_nonzero(mo_occ[1]>0)
        nvirb = nmo - noccb
        # round(6) to avoid numerical uncertainty in degeneracy
        es = numpy.append(mo_energy[0][mo_occ[0] >0],
                          mo_energy[1][mo_occ[1] >0])
        oidx = numpy.argsort(es.round(6))
        es = numpy.append(mo_energy[0][mo_occ[0]==0],
                          mo_energy[1][mo_occ[1]==0])
        vidx = numpy.argsort(es.round(6))
        orbspin = numpy.append(numpy.array([0]*nocca+[1]*noccb)[oidx],
                               numpy.array([0]*nvira+[1]*nvirb)[vidx])
    return orbspin
