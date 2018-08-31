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

from __future__ import division, print_function
import numpy as np
from pyscf.nao.m_csphar import csphar
from pyscf.nao.m_xjl import xjl
import warnings
import scipy
try:
  import numba
  from pyscf.nao.m_xjl_numba import get_bessel_xjl_numba, calc_oo2co
  use_numba = True
except:
  warnings.warn("numba not installed, using python routines")
  use_numba = False

import sys

#
#
#
def coulomb_am(self, sp1, R1, sp2, R2, **kvargs):
  """
    Computes Coulomb overlap for an atom pair. The atom pair is given by a pair of species indices and the coordinates of the atoms.
    <a|r^-1|b> = \iint a(r)|r-r'|b(r')  dr dr'
    Args: 
      self: class instance of ao_matelem_c
      sp1,sp2 : specie indices, and
      R1,R2 :   respective coordinates
    Result:
      matrix of Coulomb overlaps
    The procedure uses the angular momentum algebra and spherical Bessel transform. It is almost a repetition of bilocal overlaps.
  """

  shape = [self.ao1.sp2norbs[sp] for sp in (sp1,sp2)]
  oo2co = np.zeros(shape)
  R2mR1 = np.array(R2)-np.array(R1)
  dist,ylm = np.sqrt(sum(R2mR1*R2mR1)), csphar( R2mR1, 2*self.jmx+1 )
  cS = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype=np.complex128)
  cmat = np.zeros((self.jmx*2+1,self.jmx*2+1), dtype=np.complex128)
  rS = np.zeros((self.jmx*2+1,self.jmx*2+1))

  f1f2_mom = np.zeros((self.nr))
  l2S = np.zeros((2*self.jmx+1), dtype = np.float64)
  _j = self.jmx

#  use_numba = False

#  import h5py
#  import matplotlib.pyplot as plt
#  fig = plt.figure(1, figsize=(15, 10))
#  log_mom_fortran = h5py.File("pb_coul_aux.hdf5", "r")["sp_local2functs_mom"]
#  psi_range = np.arange(self.ao1.psi_log_mom[0].shape[1])
#  for sp in range(len(self.ao1.psi_log_mom)):
#      for mu in range(self.ao1.psi_log_mom[sp].shape[0]):
#        ax = fig.add_subplot(2, 3, mu+1)
#        error = np.zeros(self.ao1.psi_log_mom[sp].shape[0])
#        pyt = self.ao1.psi_log_mom[sp][mu, :]
#        fort = log_mom_fortran["specie_{0}/ir_mu2v".format(sp+1)].value[mu, :]
#        ax.plot(psi_range, pyt, "b", linewidth=3, label="python")
#        ax_twin = ax.twinx()
#        ax_twin.plot(psi_range, fort, "--g", linewidth=3, label="fortran mu")
#        for mu2 in range(self.ao1.psi_log_mom[sp].shape[0]):
#            fort = log_mom_fortran["specie_{0}/ir_mu2v".format(sp+1)].value[mu2, :]
#            error[mu2] = np.sum(abs(fort-pyt))
#        mu_fort = np.argmin(error)
#        fort = log_mom_fortran["specie_{0}/ir_mu2v".format(sp+1)].value[mu_fort, :]
#        ax.plot(psi_range, fort, "--r", linewidth=3, label="fortran mumod")
#        ax.legend()
#        ax.set_title("mu_python = {0}, mu_fort = {1}".format(mu+1, mu_fort+1), fontsize=20)
#        #print("look fort: ", np.sum(abs(fort-pyt)))
#
#  fig.tight_layout()
#  fig.savefig("psi_log_diff.pdf", format="pdf")
#  #plt.show()
#  #print("shape: ", self.ao1.psi_log_mom[0].shape)
#  #sys.exit()
      #self.ao1.psi_log_mom[sp] = log_mom_fortran["specie_{0}/ir_mu2v".format(sp+1)].value
#  use_numba=False

  if use_numba:
    bessel_pp = np.zeros((_j*2+1, self.nr))
    for L in range(2*_j+1):
        bessel_pp[L, :] = scipy.special.spherical_jn(L, dist*self.kk)*self.kk
    calc_oo2co(bessel_pp, self.interp_pp.dg_jt, np.array(self.ao1.sp2info[sp1]),
      np.array(self.ao1.sp2info[sp2]), self.ao1.psi_log_mom[sp1], self.ao1.psi_log_mom[sp2],
      self.njm, self._gaunt_iptr, self._gaunt_data, ylm, _j, self.jmx, self._tr_c2r,
      self._conj_c2r, l2S, cS, rS, cmat, oo2co)
  else:
    bessel_pp = np.zeros((_j*2+1, self.nr))
    for L in range(2*_j+1):
        bessel_pp[L, :] = scipy.special.spherical_jn(L, dist*self.kk)*self.kk

    #print("##############################################")
    for mu2,l2,s2,f2 in self.ao1.sp2info[sp2]:
      for mu1,l1,s1,f1 in self.ao1.sp2info[sp1]:
        f1f2_mom = self.ao1.psi_log_mom[sp2][mu2,:] * self.ao1.psi_log_mom[sp1][mu1,:]
        #print("mu1 = {0}, mu2 = {1}: sum(f1f2_mom) = ".format(mu1, mu2), np.sum(abs(f1f2_mom)))
        l2S.fill(0.0)
        for l3 in range( abs(l1-l2), l1+l2+1):
          l2S[l3] = (f1f2_mom[:]*bessel_pp[l3,:]).sum() + f1f2_mom[0]*bessel_pp[l3,0]/self.interp_pp.dg_jt*0.995
        #print("sum(S) = ", np.sum(abs(l2S)))

        cS.fill(0.0)
        for m1 in range(-l1,l1+1):
          for m2 in range(-l2,l2+1):
            gc,m3 = self.get_gaunt(l1,-m1,l2,m2), m2-m1
            for l3ind,l3 in enumerate(range(abs(l1-l2),l1+l2+1)):
              if abs(m3) > l3 : continue
              cS[m1+_j,m2+_j] = cS[m1+_j,m2+_j] + l2S[l3]*ylm[ l3*(l3+1)+m3] *\
                    gc[l3ind] * (-1.0)**((3*l1+l2+l3)//2+m2)
        self.c2r_( l1,l2, self.jmx,cS,rS,cmat)
        oo2co[s1:f1,s2:f2] = rS[-l1+_j:l1+_j+1,-l2+_j:l2+_j+1]
  #sys.exit()

  oo2co = oo2co * (4*np.pi)**2 * self.interp_pp.dg_jt
  return oo2co


if __name__ == '__main__':
  from pyscf.nao.m_system_vars import system_vars_c
  from pyscf.nao.m_prod_log import prod_log_c
  from pyscf.nao.m_ao_matelem import ao_matelem_c
  
  sv = system_vars_c("siesta")
  ra = np.array([0.0, 0.1, 0.2])
  rb = np.array([0.0, 0.1, 0.0])
  coulo = ao_matelem_c(sv.ao_log).coulomb_am(me, 0, ra, 0, rb)
