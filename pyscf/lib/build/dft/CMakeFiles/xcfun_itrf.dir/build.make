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
include dft/CMakeFiles/xcfun_itrf.dir/depend.make

# Include the progress variables for this target.
include dft/CMakeFiles/xcfun_itrf.dir/progress.make

# Include the compile flags for this target's objects.
include dft/CMakeFiles/xcfun_itrf.dir/flags.make

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o: dft/CMakeFiles/xcfun_itrf.dir/flags.make
dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o: ../dft/xcfun_itrf.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/dft && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/dft/xcfun_itrf.c

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/dft && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/dft/xcfun_itrf.c > CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.i

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/dft && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/dft/xcfun_itrf.c -o CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.s

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.requires:
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.requires

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.provides: dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.requires
	$(MAKE) -f dft/CMakeFiles/xcfun_itrf.dir/build.make dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.provides.build
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.provides

dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.provides.build: dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o

# Object files for target xcfun_itrf
xcfun_itrf_OBJECTS = \
"CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o"

# External object files for target xcfun_itrf
xcfun_itrf_EXTERNAL_OBJECTS =

../libxcfun_itrf.so: dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o
../libxcfun_itrf.so: dft/CMakeFiles/xcfun_itrf.dir/build.make
../libxcfun_itrf.so: dft/CMakeFiles/xcfun_itrf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared library ../../libxcfun_itrf.so"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/dft && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xcfun_itrf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
dft/CMakeFiles/xcfun_itrf.dir/build: ../libxcfun_itrf.so
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/build

dft/CMakeFiles/xcfun_itrf.dir/requires: dft/CMakeFiles/xcfun_itrf.dir/xcfun_itrf.c.o.requires
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/requires

dft/CMakeFiles/xcfun_itrf.dir/clean:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/dft && $(CMAKE_COMMAND) -P CMakeFiles/xcfun_itrf.dir/cmake_clean.cmake
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/clean

dft/CMakeFiles/xcfun_itrf.dir/depend:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanyao/pyscf_my/pyscf/lib /home/yuanyao/pyscf_my/pyscf/lib/dft /home/yuanyao/pyscf_my/pyscf/lib/build /home/yuanyao/pyscf_my/pyscf/lib/build/dft /home/yuanyao/pyscf_my/pyscf/lib/build/dft/CMakeFiles/xcfun_itrf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dft/CMakeFiles/xcfun_itrf.dir/depend
