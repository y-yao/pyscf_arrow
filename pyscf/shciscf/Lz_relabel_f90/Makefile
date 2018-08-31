# This Makefile is to be used with gmake.  On Linux systems, gmake is the default make.

system := $(shell uname)
hostname := $(shell uname -n)

ifeq ($(system),Linux)

#FC = ifort
#FFLAGS = -zero -save -extend_source -w -r8 -O3 -pad
#LDFLAGS = -nothread
 FC=gfortran
# Debug
#FFLAGS= -O3 -ffree-line-length-none -Wall -g -fbounds-check
# Production
 FFLAGS= -O3 -ffree-line-length-none -Wall

endif

.SUFFIXES: .f90 .f95 .o .f .c

.f90.o:
	$(FC) $(FFLAGS) -o $@ -c $<

OBJS = fix_orb_sym.o

fix_orb_sym: $(OBJS)
	$(FC) $(FFLAGS) $(OBJS) -o $@
