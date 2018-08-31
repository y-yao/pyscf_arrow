import os
import subprocess as sp

if os.path.isfile("FCIDUMP_new"):
  sp.check_call('rm FCIDUMP_new', shell = True)
sp.check_call('python checkLz_fcidump.py FCIDUMP', shell=True)
while os.path.isfile("FCIDUMP_new"):
  sp.check_call('mv FCIDUMP_new FCIDUMP_progress', shell = True)
  sp.check_call('python checkLz_fcidump.py FCIDUMP_progress', shell = True)
sp.check_call('mv FCIDUMP_progress FCIDUMP_finished', shell = True)
