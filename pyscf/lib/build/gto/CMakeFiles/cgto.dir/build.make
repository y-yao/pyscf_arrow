# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yuanyao/pyscf_my/pyscf/lib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yuanyao/pyscf_my/pyscf/lib/build

# Include any dependencies generated for this target.
include gto/CMakeFiles/cgto.dir/depend.make

# Include the progress variables for this target.
include gto/CMakeFiles/cgto.dir/progress.make

# Include the compile flags for this target's objects.
include gto/CMakeFiles/cgto.dir/flags.make

gto/CMakeFiles/cgto.dir/fill_int2c.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fill_int2c.c.o: ../gto/fill_int2c.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fill_int2c.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fill_int2c.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2c.c

gto/CMakeFiles/cgto.dir/fill_int2c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fill_int2c.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2c.c > CMakeFiles/cgto.dir/fill_int2c.c.i

gto/CMakeFiles/cgto.dir/fill_int2c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fill_int2c.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2c.c -o CMakeFiles/cgto.dir/fill_int2c.c.s

gto/CMakeFiles/cgto.dir/fill_int2c.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fill_int2c.c.o.requires

gto/CMakeFiles/cgto.dir/fill_int2c.c.o.provides: gto/CMakeFiles/cgto.dir/fill_int2c.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fill_int2c.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fill_int2c.c.o.provides

gto/CMakeFiles/cgto.dir/fill_int2c.c.o.provides.build: gto/CMakeFiles/cgto.dir/fill_int2c.c.o

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o: ../gto/fill_nr_3c.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fill_nr_3c.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_nr_3c.c

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fill_nr_3c.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_nr_3c.c > CMakeFiles/cgto.dir/fill_nr_3c.c.i

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fill_nr_3c.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_nr_3c.c -o CMakeFiles/cgto.dir/fill_nr_3c.c.s

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.requires

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.provides: gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.provides

gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.provides.build: gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o

gto/CMakeFiles/cgto.dir/fill_r_3c.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fill_r_3c.c.o: ../gto/fill_r_3c.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fill_r_3c.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fill_r_3c.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_3c.c

gto/CMakeFiles/cgto.dir/fill_r_3c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fill_r_3c.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_3c.c > CMakeFiles/cgto.dir/fill_r_3c.c.i

gto/CMakeFiles/cgto.dir/fill_r_3c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fill_r_3c.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_3c.c -o CMakeFiles/cgto.dir/fill_r_3c.c.s

gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.requires

gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.provides: gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.provides

gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.provides.build: gto/CMakeFiles/cgto.dir/fill_r_3c.c.o

gto/CMakeFiles/cgto.dir/fill_int2e.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fill_int2e.c.o: ../gto/fill_int2e.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fill_int2e.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fill_int2e.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2e.c

gto/CMakeFiles/cgto.dir/fill_int2e.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fill_int2e.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2e.c > CMakeFiles/cgto.dir/fill_int2e.c.i

gto/CMakeFiles/cgto.dir/fill_int2e.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fill_int2e.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_int2e.c -o CMakeFiles/cgto.dir/fill_int2e.c.s

gto/CMakeFiles/cgto.dir/fill_int2e.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fill_int2e.c.o.requires

gto/CMakeFiles/cgto.dir/fill_int2e.c.o.provides: gto/CMakeFiles/cgto.dir/fill_int2e.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fill_int2e.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fill_int2e.c.o.provides

gto/CMakeFiles/cgto.dir/fill_int2e.c.o.provides.build: gto/CMakeFiles/cgto.dir/fill_int2e.c.o

gto/CMakeFiles/cgto.dir/fill_r_4c.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fill_r_4c.c.o: ../gto/fill_r_4c.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fill_r_4c.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fill_r_4c.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_4c.c

gto/CMakeFiles/cgto.dir/fill_r_4c.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fill_r_4c.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_4c.c > CMakeFiles/cgto.dir/fill_r_4c.c.i

gto/CMakeFiles/cgto.dir/fill_r_4c.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fill_r_4c.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fill_r_4c.c -o CMakeFiles/cgto.dir/fill_r_4c.c.s

gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.requires

gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.provides: gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.provides

gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.provides.build: gto/CMakeFiles/cgto.dir/fill_r_4c.c.o

gto/CMakeFiles/cgto.dir/ft_ao.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/ft_ao.c.o: ../gto/ft_ao.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/ft_ao.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/ft_ao.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao.c

gto/CMakeFiles/cgto.dir/ft_ao.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/ft_ao.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao.c > CMakeFiles/cgto.dir/ft_ao.c.i

gto/CMakeFiles/cgto.dir/ft_ao.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/ft_ao.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao.c -o CMakeFiles/cgto.dir/ft_ao.c.s

gto/CMakeFiles/cgto.dir/ft_ao.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/ft_ao.c.o.requires

gto/CMakeFiles/cgto.dir/ft_ao.c.o.provides: gto/CMakeFiles/cgto.dir/ft_ao.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/ft_ao.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/ft_ao.c.o.provides

gto/CMakeFiles/cgto.dir/ft_ao.c.o.provides.build: gto/CMakeFiles/cgto.dir/ft_ao.c.o

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o: ../gto/ft_ao_deriv.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_7)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/ft_ao_deriv.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao_deriv.c

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/ft_ao_deriv.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao_deriv.c > CMakeFiles/cgto.dir/ft_ao_deriv.c.i

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/ft_ao_deriv.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/ft_ao_deriv.c -o CMakeFiles/cgto.dir/ft_ao_deriv.c.s

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.requires

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.provides: gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.provides

gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.provides.build: gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o: ../gto/grid_ao_drv.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_8)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/grid_ao_drv.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/grid_ao_drv.c

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/grid_ao_drv.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/grid_ao_drv.c > CMakeFiles/cgto.dir/grid_ao_drv.c.i

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/grid_ao_drv.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/grid_ao_drv.c -o CMakeFiles/cgto.dir/grid_ao_drv.c.s

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.requires

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.provides: gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.provides

gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.provides.build: gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o

gto/CMakeFiles/cgto.dir/fastexp.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/fastexp.c.o: ../gto/fastexp.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_9)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/fastexp.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/fastexp.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/fastexp.c

gto/CMakeFiles/cgto.dir/fastexp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/fastexp.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/fastexp.c > CMakeFiles/cgto.dir/fastexp.c.i

gto/CMakeFiles/cgto.dir/fastexp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/fastexp.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/fastexp.c -o CMakeFiles/cgto.dir/fastexp.c.s

gto/CMakeFiles/cgto.dir/fastexp.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/fastexp.c.o.requires

gto/CMakeFiles/cgto.dir/fastexp.c.o.provides: gto/CMakeFiles/cgto.dir/fastexp.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/fastexp.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/fastexp.c.o.provides

gto/CMakeFiles/cgto.dir/fastexp.c.o.provides.build: gto/CMakeFiles/cgto.dir/fastexp.c.o

gto/CMakeFiles/cgto.dir/deriv1.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/deriv1.c.o: ../gto/deriv1.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_10)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/deriv1.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/deriv1.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv1.c

gto/CMakeFiles/cgto.dir/deriv1.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/deriv1.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv1.c > CMakeFiles/cgto.dir/deriv1.c.i

gto/CMakeFiles/cgto.dir/deriv1.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/deriv1.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv1.c -o CMakeFiles/cgto.dir/deriv1.c.s

gto/CMakeFiles/cgto.dir/deriv1.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/deriv1.c.o.requires

gto/CMakeFiles/cgto.dir/deriv1.c.o.provides: gto/CMakeFiles/cgto.dir/deriv1.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/deriv1.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/deriv1.c.o.provides

gto/CMakeFiles/cgto.dir/deriv1.c.o.provides.build: gto/CMakeFiles/cgto.dir/deriv1.c.o

gto/CMakeFiles/cgto.dir/deriv2.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/deriv2.c.o: ../gto/deriv2.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_11)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/deriv2.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/deriv2.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv2.c

gto/CMakeFiles/cgto.dir/deriv2.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/deriv2.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv2.c > CMakeFiles/cgto.dir/deriv2.c.i

gto/CMakeFiles/cgto.dir/deriv2.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/deriv2.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/deriv2.c -o CMakeFiles/cgto.dir/deriv2.c.s

gto/CMakeFiles/cgto.dir/deriv2.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/deriv2.c.o.requires

gto/CMakeFiles/cgto.dir/deriv2.c.o.provides: gto/CMakeFiles/cgto.dir/deriv2.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/deriv2.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/deriv2.c.o.provides

gto/CMakeFiles/cgto.dir/deriv2.c.o.provides.build: gto/CMakeFiles/cgto.dir/deriv2.c.o

gto/CMakeFiles/cgto.dir/nr_ecp.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/nr_ecp.c.o: ../gto/nr_ecp.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_12)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/nr_ecp.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/nr_ecp.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp.c

gto/CMakeFiles/cgto.dir/nr_ecp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/nr_ecp.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp.c > CMakeFiles/cgto.dir/nr_ecp.c.i

gto/CMakeFiles/cgto.dir/nr_ecp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/nr_ecp.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp.c -o CMakeFiles/cgto.dir/nr_ecp.c.s

gto/CMakeFiles/cgto.dir/nr_ecp.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/nr_ecp.c.o.requires

gto/CMakeFiles/cgto.dir/nr_ecp.c.o.provides: gto/CMakeFiles/cgto.dir/nr_ecp.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/nr_ecp.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/nr_ecp.c.o.provides

gto/CMakeFiles/cgto.dir/nr_ecp.c.o.provides.build: gto/CMakeFiles/cgto.dir/nr_ecp.c.o

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o: ../gto/nr_ecp_deriv.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_13)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/nr_ecp_deriv.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp_deriv.c

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/nr_ecp_deriv.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp_deriv.c > CMakeFiles/cgto.dir/nr_ecp_deriv.c.i

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/nr_ecp_deriv.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/nr_ecp_deriv.c -o CMakeFiles/cgto.dir/nr_ecp_deriv.c.s

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.requires

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.provides: gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.provides

gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.provides.build: gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o: gto/CMakeFiles/cgto.dir/flags.make
gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o: ../gto/autocode/auto_eval1.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_14)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/cgto.dir/autocode/auto_eval1.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/gto/autocode/auto_eval1.c

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cgto.dir/autocode/auto_eval1.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/gto/autocode/auto_eval1.c > CMakeFiles/cgto.dir/autocode/auto_eval1.c.i

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cgto.dir/autocode/auto_eval1.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/gto/autocode/auto_eval1.c -o CMakeFiles/cgto.dir/autocode/auto_eval1.c.s

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.requires:
.PHONY : gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.requires

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.provides: gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.requires
	$(MAKE) -f gto/CMakeFiles/cgto.dir/build.make gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.provides.build
.PHONY : gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.provides

gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.provides.build: gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o

# Object files for target cgto
cgto_OBJECTS = \
"CMakeFiles/cgto.dir/fill_int2c.c.o" \
"CMakeFiles/cgto.dir/fill_nr_3c.c.o" \
"CMakeFiles/cgto.dir/fill_r_3c.c.o" \
"CMakeFiles/cgto.dir/fill_int2e.c.o" \
"CMakeFiles/cgto.dir/fill_r_4c.c.o" \
"CMakeFiles/cgto.dir/ft_ao.c.o" \
"CMakeFiles/cgto.dir/ft_ao_deriv.c.o" \
"CMakeFiles/cgto.dir/grid_ao_drv.c.o" \
"CMakeFiles/cgto.dir/fastexp.c.o" \
"CMakeFiles/cgto.dir/deriv1.c.o" \
"CMakeFiles/cgto.dir/deriv2.c.o" \
"CMakeFiles/cgto.dir/nr_ecp.c.o" \
"CMakeFiles/cgto.dir/nr_ecp_deriv.c.o" \
"CMakeFiles/cgto.dir/autocode/auto_eval1.c.o"

# External object files for target cgto
cgto_EXTERNAL_OBJECTS =

../libcgto.so: gto/CMakeFiles/cgto.dir/fill_int2c.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/fill_r_3c.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/fill_int2e.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/fill_r_4c.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/ft_ao.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/fastexp.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/deriv1.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/deriv2.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/nr_ecp.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o
../libcgto.so: gto/CMakeFiles/cgto.dir/build.make
../libcgto.so: ../libnp_helper.so
../libcgto.so: /opt/libUbuntuCompat/libf77blas.so.3gf
../libcgto.so: /opt/libUbuntuCompat/libatlas.so.3gf
../libcgto.so: gto/CMakeFiles/cgto.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared library ../../libcgto.so"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cgto.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
gto/CMakeFiles/cgto.dir/build: ../libcgto.so
.PHONY : gto/CMakeFiles/cgto.dir/build

gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fill_int2c.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fill_nr_3c.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fill_r_3c.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fill_int2e.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fill_r_4c.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/ft_ao.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/ft_ao_deriv.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/grid_ao_drv.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/fastexp.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/deriv1.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/deriv2.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/nr_ecp.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/nr_ecp_deriv.c.o.requires
gto/CMakeFiles/cgto.dir/requires: gto/CMakeFiles/cgto.dir/autocode/auto_eval1.c.o.requires
.PHONY : gto/CMakeFiles/cgto.dir/requires

gto/CMakeFiles/cgto.dir/clean:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/gto && $(CMAKE_COMMAND) -P CMakeFiles/cgto.dir/cmake_clean.cmake
.PHONY : gto/CMakeFiles/cgto.dir/clean

gto/CMakeFiles/cgto.dir/depend:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanyao/pyscf_my/pyscf/lib /home/yuanyao/pyscf_my/pyscf/lib/gto /home/yuanyao/pyscf_my/pyscf/lib/build /home/yuanyao/pyscf_my/pyscf/lib/build/gto /home/yuanyao/pyscf_my/pyscf/lib/build/gto/CMakeFiles/cgto.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : gto/CMakeFiles/cgto.dir/depend
