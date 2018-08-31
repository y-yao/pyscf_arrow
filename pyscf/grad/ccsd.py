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
CCSD analytical nuclear gradients
'''

import time
import ctypes
import numpy
from pyscf import lib
from functools import reduce
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import _ccsd
from pyscf.cc import ccsd_rdm
from pyscf.scf import cphf
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange, _index_frozen_active


#
# Note: only works with canonical orbitals
# Non-canonical formula refers to JCP, 95, 2639
#
def kernel(mycc, t1=None, t2=None, l1=None, l2=None, eris=None, atmlst=None,
           mf_grad=None, d1=None, d2=None, verbose=logger.INFO):
    if eris is not None:
        if abs(eris.fock - numpy.diag(eris.fock.diagonal())).max() > 1e-3:
            raise RuntimeError('CCSD gradients does not support NHF (non-canonical HF)')

    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if mf_grad is None: mf_grad = mycc._scf.nuc_grad_method()

    log = logger.new_logger(mycc, verbose)
    time0 = time.clock(), time.time()

    log.debug('Build ccsd rdm1 intermediates')
    if d1 is None:
        d1 = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    time1 = log.timer_debug1('rdm1 intermediates', *time0)
    log.debug('Build ccsd rdm2 intermediates')
    fdm2 = lib.H5TmpFile()
    if d2 is None:
        d2 = ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, fdm2, True)
    time1 = log.timer_debug1('rdm2 intermediates', *time1)

    mol = mycc.mol
    mo_coeff = mycc.mo_coeff
    mo_energy = mycc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(mycc.mo_occ > 0)
    with_frozen = not (mycc.frozen is None or mycc.frozen is 0)
    OA, VA, OF, VF = _index_frozen_active(mycc.get_frozen_mask(), mycc.mo_occ)

    log.debug('symmetrized rdm2 and MO->AO transformation')
# Roughly, dm2*2 is computed in _rdm2_mo2ao
    mo_active = mo_coeff[:,numpy.hstack((OA,VA))]
    _rdm2_mo2ao(mycc, d2, mo_active, fdm2)  # transform the active orbitals
    time1 = log.timer_debug1('MO->AO transformation', *time1)
    hf_dm1 = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol.offset_nr_by_atom()
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    de = numpy.zeros((len(atmlst),3))
    Imat = numpy.zeros((nao,nao))
    vhf1 = fdm2.create_dataset('vhf1', (len(atmlst),3,nao,nao), 'f8')

# 2e AO integrals dot 2pdm
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory*.9e6/8/(nao**3*2.5)))

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
        ip1 = p0
        vhf = numpy.zeros((3,nao,nao))
        for b0, b1, nf in _shell_prange(mol, shl0, shl1, blksize):
            ip0, ip1 = ip1, ip1 + nf
            dm2buf = _load_block_tril(fdm2['dm2'], ip0, ip1, nao)
            dm2buf[:,:,diagidx] *= .5
            shls_slice = (b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)
            eri0 = mol.intor('int2e', aosym='s2kl', shls_slice=shls_slice)
            Imat += lib.einsum('ipx,iqx->pq', eri0.reshape(nf,nao,-1), dm2buf)
            eri0 = None

            eri1 = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                             shls_slice=shls_slice).reshape(3,nf,nao,-1)
            de[k] -= numpy.einsum('xijk,ijk->x', eri1, dm2buf) * 2
            dm2buf = None
# HF part
            for i in range(3):
                eri1tmp = lib.unpack_tril(eri1[i].reshape(nf*nao,-1))
                eri1tmp = eri1tmp.reshape(nf,nao,nao,nao)
                vhf[i] += numpy.einsum('ijkl,ij->kl', eri1tmp, hf_dm1[ip0:ip1])
                vhf[i] -= numpy.einsum('ijkl,il->kj', eri1tmp, hf_dm1[ip0:ip1]) * .5
                vhf[i,ip0:ip1] += numpy.einsum('ijkl,kl->ij', eri1tmp, hf_dm1)
                vhf[i,ip0:ip1] -= numpy.einsum('ijkl,jk->il', eri1tmp, hf_dm1) * .5
            eri1 = eri1tmp = None
        vhf1[k] = vhf
        log.debug('2e-part grad of atom %d %s = %s', ia, mol.atom_symbol(ia), de[k])
        time1 = log.timer_debug1('2e-part grad of atom %d'%ia, *time1)

    Imat = reduce(numpy.dot, (mo_coeff.T, Imat, mycc._scf.get_ovlp(), mo_coeff)) * -1

    dm1mo = numpy.zeros((nmo,nmo))
    if with_frozen:
        dco = Imat[OF[:,None],OA] / (mo_energy[OF,None] - mo_energy[OA])
        dfv = Imat[VF[:,None],VA] / (mo_energy[VF,None] - mo_energy[VA])
        dm1mo[OA[:,None],OA] = doo + doo.T
        dm1mo[OF[:,None],OA] = dco
        dm1mo[OA[:,None],OF] = dco.T
        dm1mo[VA[:,None],VA] = dvv + dvv.T
        dm1mo[VF[:,None],VA] = dfv
        dm1mo[VA[:,None],VF] = dfv.T
    else:
        dm1mo[:nocc,:nocc] = doo + doo.T
        dm1mo[nocc:,nocc:] = dvv + dvv.T

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    vhf = mycc._scf.get_veff(mycc.mol, dm1) * 2
    Xvo = reduce(numpy.dot, (mo_coeff[:,nocc:].T, vhf, mo_coeff[:,:nocc]))
    Xvo+= Imat[:nocc,nocc:].T - Imat[nocc:,:nocc]

    dm1mo += _response_dm1(mycc, Xvo, eris)
    time1 = log.timer_debug1('response_rdm1 intermediates', *time1)

    Imat[nocc:,:nocc] = Imat[:nocc,nocc:].T
    im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))
    time1 = log.timer_debug1('response_rdm1', *time1)

    log.debug('h1 and JK1')
    hcore_deriv = mf_grad.hcore_generator(mol)
    s1 = mf_grad.get_ovlp(mol)
    zeta = lib.direct_sum('i+j->ij', mo_energy, mo_energy) * .5
    zeta[nocc:,:nocc] = mo_energy[:nocc]
    zeta[:nocc,nocc:] = mo_energy[:nocc].reshape(-1,1)
    zeta = reduce(numpy.dot, (mo_coeff, zeta*dm1mo, mo_coeff.T))

    dm1 = reduce(numpy.dot, (mo_coeff, dm1mo, mo_coeff.T))
    p1 = numpy.dot(mo_coeff[:,:nocc], mo_coeff[:,:nocc].T)
    vhf_s1occ = reduce(numpy.dot, (p1, mycc._scf.get_veff(mol, dm1+dm1.T), p1))
    time1 = log.timer_debug1('h1 and JK1', *time1)

    # Hartree-Fock part contribution
    dm1p = hf_dm1 + dm1*2
    dm1 += hf_dm1
    zeta += mf_grad.make_rdm1e(mo_energy, mo_coeff, mycc.mo_occ)

    for k, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ia]
# s[1] dot I, note matrix im1 is not hermitian
        de[k] += numpy.einsum('xij,ij->x', s1[:,p0:p1], im1[p0:p1])
        de[k] += numpy.einsum('xji,ij->x', s1[:,p0:p1], im1[:,p0:p1])
# h[1] \dot DM, contribute to f1
        h1ao = hcore_deriv(ia)
        de[k] += numpy.einsum('xij,ji->x', h1ao, dm1)
# -s[1]*e \dot DM,  contribute to f1
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], zeta[p0:p1]  )
        de[k] -= numpy.einsum('xji,ij->x', s1[:,p0:p1], zeta[:,p0:p1])
# -vhf[s_ij[1]],  contribute to f1, *2 for s1+s1.T
        de[k] -= numpy.einsum('xij,ij->x', s1[:,p0:p1], vhf_s1occ[p0:p1]) * 2
        de[k] -= numpy.einsum('xij,ij->x', vhf1[k], dm1p)

    de += rhf_grad.grad_nuc(mol, atmlst)
    log.timer('%s gradients' % mycc.__class__.__name__, *time0)
    return de


def as_scanner(grad_cc):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns total CCSD energy.

    The solver will automatically use the results of last calculation as the
    initial guess of the new calculation.  All parameters assigned in the
    CCSD and the underlying SCF objects (conv_tol, max_memory etc) are
    automatically applied in the solver.

    Note scanner has side effects.  It may change many underlying objects
    (_scf, with_df, with_x2c, ...) during calculation.

    Examples::

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom='H 0 0 0; F 0 0 1')
    >>> cc_scanner = cc.CCSD(scf.RHF(mol)).nuc_grad_method().as_scanner()
    >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.1'))
    >>> e_tot, grad = cc_scanner(gto.M(atom='H 0 0 0; F 0 0 1.5'))
    '''
    from pyscf import gto
    if isinstance(grad_cc, lib.GradScanner):
        return grad_cc

    logger.info(grad_cc, 'Create scanner for %s', grad_cc.__class__)

    class CCSD_GradScanner(grad_cc.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, gto.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            # The following simple version also works.  But eris object is
            # recomputed in cc_scanner and solve_lambda.
            # cc = self.base
            # cc(mol)
            # eris = cc.ao2mo()
            # cc.solve_lambda(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris)
            # mf_grad = cc._scf.nuc_grad_method()
            # de = self.kernel(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris, mf_grad=mf_grad)

            cc = self.base
            mf_scanner = cc._scf
            mf_scanner(mol)
            cc.mol = mol
            cc.mo_coeff = mf_scanner.mo_coeff
            cc.mo_occ = mf_scanner.mo_occ
            eris = cc.ao2mo(cc.mo_coeff)
            cc.kernel(cc.t1, cc.t2, eris=eris)
            cc.solve_lambda(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris)
            mf_grad = mf_scanner.nuc_grad_method()
            de = self.kernel(cc.t1, cc.t2, cc.l1, cc.l2, eris=eris,
                             mf_grad=mf_grad, **kwargs)
            return cc.e_tot, de
        @property
        def converged(self):
            cc = self.base
            return all((cc._scf.converged, cc.converged, cc.converged_lambda))
    return CCSD_GradScanner(grad_cc)

def _response_dm1(mycc, Xvo, eris=None):
    nvir, nocc = Xvo.shape
    nmo = nocc + nvir
    with_frozen = not (mycc.frozen is None or mycc.frozen is 0)
    if eris is None or with_frozen:
        mo_energy = mycc._scf.mo_energy
        mo_occ = mycc.mo_occ
        mo_coeff = mycc.mo_coeff
        def fvind(x):
            x = x.reshape(Xvo.shape)
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            v = mycc._scf.get_veff(mycc.mol, dm + dm.T)
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, v, mo_coeff[:,:nocc]))
            return v * 2
    else:
        mo_energy = eris.fock.diagonal()
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        ovvo = numpy.empty((nocc,nvir,nvir,nocc))
        for i in range(nocc):
            ovvo[i] = eris.ovvo[i]
            ovvo[i] = ovvo[i] * 4 - ovvo[i].transpose(1,0,2)
            ovvo[i]-= eris.oovv[i].transpose(2,1,0)
        def fvind(x):
            return numpy.einsum('iabj,bj->ai', ovvo, x.reshape(Xvo.shape))
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1

def _rdm2_mo2ao(mycc, d2, mo_coeff, fsave=None):
# dm2 = ccsd_rdm._make_rdm2(mycc, None, d2, with_dm1=False)
# dm2 = numpy.einsum('pi,ijkl->pjkl', mo_coeff, dm2)
# dm2 = numpy.einsum('pj,ijkl->ipkl', mo_coeff, dm2)
# dm2 = numpy.einsum('pk,ijkl->ijpl', mo_coeff, dm2)
# dm2 = numpy.einsum('pl,ijkl->ijkp', mo_coeff, dm2)
# dm2 = dm2 + dm2.transpose(1,0,2,3)
# dm2 = dm2 + dm2.transpose(0,1,3,2)
# return ao2mo.restore(4, dm2*.5, nmo)
    log = logger.Logger(mycc.stdout, mycc.verbose)
    time1 = time.clock(), time.time()
    if fsave is None:
        incore = True
        fsave = lib.H5TmpFile()
    else:
        incore = False
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2

    nocc, nvir = dovov.shape[:2]
    mo_coeff = numpy.asarray(mo_coeff, order='F')
    nao, nmo = mo_coeff.shape
    nao_pair = nao * (nao+1) // 2
    nvir_pair = nvir * (nvir+1) //2

    fdrv = getattr(_ccsd.libcc, 'AO2MOnr_e2_drv')
    ftrans = _ccsd.libcc.AO2MOtranse2_nr_s1
    fmm = _ccsd.libcc.CCmmm_transpose_sum
    pao_loc = ctypes.POINTER(ctypes.c_void_p)()
    def _trans(vin, orbs_slice, out=None):
        nrow = vin.shape[0]
        if out is None:
            out = numpy.empty((nrow,nao_pair))
        fdrv(ftrans, fmm,
             out.ctypes.data_as(ctypes.c_void_p),
             vin.ctypes.data_as(ctypes.c_void_p),
             mo_coeff.ctypes.data_as(ctypes.c_void_p),
             ctypes.c_int(nrow), ctypes.c_int(nao),
             (ctypes.c_int*4)(*orbs_slice), pao_loc, ctypes.c_int(0))
        return out

    fswap = lib.H5TmpFile()
    max_memory = mycc.max_memory - lib.current_memory()[0]
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nvir_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('v', (nao_pair,nvir_pair), 'f8', chunks=(nao_pair,blksize))
    for p0, p1 in lib.prange(0, nvir_pair, blksize):
        fswap['v'][:,p0:p1] = _trans(lib.unpack_tril(_cp(dvvvv[p0:p1])),
                                     (nocc,nmo,nocc,nmo)).T
    time1 = log.timer_debug1('_rdm2_mo2ao pass 1', *time1)

# transform dm2_ij to get lower triangular (dm2+dm2.transpose(0,1,3,2))
    blksize = int(max_memory*1e6/8/(nao_pair+nmo**2))
    blksize = min(nao_pair, max(ccsd.BLKMIN, blksize))
    fswap.create_dataset('o', (nmo,nocc,nao_pair), 'f8', chunks=(nmo,nocc,blksize))
    buf1 = numpy.zeros((nocc,nocc,nmo,nmo))
    buf1[:,:,:nocc,:nocc] = doooo
    buf1[:,:,nocc:,nocc:] = _cp(doovv)
    buf1 = _trans(buf1.reshape(nocc**2,-1), (0,nmo,0,nmo))
    fswap['o'][:nocc] = buf1.reshape(nocc,nocc,nao_pair)
    dovoo = numpy.asarray(dooov).transpose(2,3,0,1)
    for p0, p1 in lib.prange(nocc, nmo, nocc):
        buf1 = numpy.zeros((nocc,p1-p0,nmo,nmo))
        buf1[:,:,:nocc,:nocc] = dovoo[:,p0-nocc:p1-nocc]
        buf1[:,:,nocc:,:nocc] = dovvo[:,p0-nocc:p1-nocc]
        buf1[:,:,:nocc,nocc:] = dovov[:,p0-nocc:p1-nocc]
        buf1[:,:,nocc:,nocc:] = dovvv[:,p0-nocc:p1-nocc]
        buf1 = buf1.transpose(1,0,3,2).reshape((p1-p0)*nocc,-1)
        buf1 = _trans(buf1, (0,nmo,0,nmo))
        fswap['o'][p0:p1] = buf1.reshape(p1-p0,nocc,nao_pair)
    time1 = log.timer_debug1('_rdm2_mo2ao pass 2', *time1)
    dovoo = buf1 = None

# transform dm2_kl then dm2 + dm2.transpose(2,3,0,1)
    gsave = fsave.create_dataset('dm2', (nao_pair,nao_pair), 'f8', chunks=(nao_pair,blksize))
    for p0, p1 in lib.prange(0, nao_pair, blksize):
        buf1 = numpy.zeros((p1-p0,nmo,nmo))
        buf1[:,nocc:,nocc:] = lib.unpack_tril(_cp(fswap['v'][p0:p1]))
        buf1[:,:,:nocc] = fswap['o'][:,:,p0:p1].transpose(2,0,1)
        buf2 = _trans(buf1, (0,nmo,0,nmo))
        if p0 > 0:
            buf1 = _cp(gsave[:p0,p0:p1])
            buf1[:p0,:p1-p0] += buf2[:p1-p0,:p0].T
            buf2[:p1-p0,:p0] = buf1[:p0,:p1-p0].T
            gsave[:p0,p0:p1] = buf1
        lib.transpose_sum(buf2[:,p0:p1], inplace=True)
        gsave[p0:p1] = buf2
    time1 = log.timer_debug1('_rdm2_mo2ao pass 3', *time1)
    if incore:
        return fsave['dm2'].value
    else:
        return fsave

#
# .
# . .
# ----+             -----------
# ----|-+       =>  -----------
# . . | | .
# . . | | . .
#
def _load_block_tril(h5dat, row0, row1, nao, out=None):
    nao_pair = nao * (nao+1) // 2
    if out is None:
        out = numpy.ndarray((row1-row0,nao,nao_pair))
    dat = h5dat[row0*(row0+1)//2:row1*(row1+1)//2]
    p1 = 0
    for i in range(row0, row1):
        p0, p1 = p1, p1 + i+1
        out[i-row0,:i+1] = dat[p0:p1]
        for j in range(row0, i):
            out[j-row0,i] = out[i-row0,j]
    for i in range(row1, nao):
        i2 = i*(i+1)//2
        out[:,i] = h5dat[i2+row0:i2+row1]
    return out

def _cp(a):
    return numpy.array(a, copy=False, order='C')

class Gradients(lib.StreamObject):
    def __init__(self, mycc):
        self.base = mycc
        self.mol = mycc.mol
        self.stdout = mycc.stdout
        self.verbose = mycc.verbose
        self.atmlst = None
        self.de = None
        self._keys = set(self.__dict__.keys())

    def kernel(self, t1=None, t2=None, l1=None, l2=None, eris=None,
               atmlst=None, mf_grad=None, verbose=None, _kern=kernel):
        log = logger.new_logger(self, verbose)
        mycc = self.base
        if t1 is None: t1 = mycc.t1
        if t2 is None: t2 = mycc.t2
        if l1 is None: l1 = mycc.l1
        if l2 is None: l2 = mycc.l2
        if eris is None:
            eris = mycc.ao2mo()
        if t1 is None or t2 is None:
            t1, t2 = mycc.kernel(eris=eris)[1:]
        if l1 is None or l2 is None:
            l1, l2 = mycc.solve_lambda(eris=eris)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        self.de = _kern(mycc, t1, t2, l1, l2, eris, atmlst,
                        mf_grad, verbose=log)
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            rhf_grad._write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    as_scanner = as_scanner


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf).run()
    g1 = Gradients(mycc).kernel()
#[[ 0   0                1.00950925e-02]
# [ 0   2.28063426e-02  -5.04754623e-03]
# [ 0  -2.28063426e-02  -5.04754623e-03]]
    print(lib.finger(g1) - -0.036999389889460096)

    mcs = mycc.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2] - (e1-e2)/0.002*lib.param.BOHR)

    print('-----------------------------------')
    mol = gto.M(
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf)
    mycc.frozen = [0,1,10,11,12]
    mycc.max_memory = 1
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
#[[ -7.81105940e-17   3.81840540e-15   1.20415540e-02]
# [  1.73095055e-16  -7.94568837e-02  -6.02077699e-03]
# [ -9.49844615e-17   7.94568837e-02  -6.02077699e-03]]
    print(lib.finger(g1) - 0.10599632044533455)

    mcs = mycc.as_scanner()
    mol.set_geom_([
            ["O" , (0. , 0.     , 0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e1 = mcs(mol)
    mol.set_geom_([
            ["O" , (0. , 0.     ,-0.001)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]])
    e2 = mcs(mol)
    print(g1[0,2] - (e1-e2)/0.002*lib.param.BOHR)

    mol = gto.M(
        atom = 'H 0 0 0; H 0 0 1.76',
        basis = '631g',
        unit='Bohr')
    mf = scf.RHF(mol).run(conv_tol=1e-14)
    mycc = ccsd.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-10
    mycc.kernel()
    g1 = Gradients(mycc).kernel()
#[[ 0.          0.         -0.07080036]
# [ 0.          0.          0.07080036]]