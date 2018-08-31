#
# This file is an example to set the environment.
# The configs will be used in dmrgci.py and chemps2.py
#

import os
from pyscf import lib

# To install Block as the FCI solver for CASSCF, see
#       http://sunqm.github.io/Block/build.html
#       https://github.com/sanshar/Block
BLOCKEXE = '/path/to/Block/block.spin_adapted'
BLOCKEXE_COMPRESS_NEVPT = '/path/to/serially/compiled/Block/block.spin_adapted'
#BLOCKSCRATCHDIR = os.path.join('./scratch', str(os.getpid()))
BLOCKSCRATCHDIR = os.path.join(lib.param.TMPDIR, str(os.getpid()))
#BLOCKRUNTIMEDIR = '.'
BLOCKRUNTIMEDIR = str(os.getpid())
MPIPREFIX = 'mpirun'  # change to srun for SLURM job system


# Use ChemPS2 as the FCI solver for CASSCF
# building PyChemPS2, a python module will be generated in
#       /path/to/ChemPS2/build/PyChemPS2
# see more details in the ChemPS2 document
#       https://github.com/SebWouters/CheMPS2
PYCHEMPS2BIN = '/path/to/CheMPS2/build/PyCheMPS2/PyCheMPS2.so'
