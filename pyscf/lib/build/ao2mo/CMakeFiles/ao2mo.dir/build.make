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
include ao2mo/CMakeFiles/ao2mo.dir/depend.make

# Include the progress variables for this target.
include ao2mo/CMakeFiles/ao2mo.dir/progress.make

# Include the compile flags for this target's objects.
include ao2mo/CMakeFiles/ao2mo.dir/flags.make

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o: ao2mo/CMakeFiles/ao2mo.dir/flags.make
ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o: ../ao2mo/restore_eri.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/ao2mo.dir/restore_eri.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/restore_eri.c

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ao2mo.dir/restore_eri.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/restore_eri.c > CMakeFiles/ao2mo.dir/restore_eri.c.i

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ao2mo.dir/restore_eri.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/restore_eri.c -o CMakeFiles/ao2mo.dir/restore_eri.c.s

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.requires:
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.requires

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.provides: ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.requires
	$(MAKE) -f ao2mo/CMakeFiles/ao2mo.dir/build.make ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.provides.build
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.provides

ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.provides.build: ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o: ao2mo/CMakeFiles/ao2mo.dir/flags.make
ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o: ../ao2mo/nr_ao2mo.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/ao2mo.dir/nr_ao2mo.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_ao2mo.c

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ao2mo.dir/nr_ao2mo.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_ao2mo.c > CMakeFiles/ao2mo.dir/nr_ao2mo.c.i

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ao2mo.dir/nr_ao2mo.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_ao2mo.c -o CMakeFiles/ao2mo.dir/nr_ao2mo.c.s

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.requires:
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.requires

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.provides: ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.requires
	$(MAKE) -f ao2mo/CMakeFiles/ao2mo.dir/build.make ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.provides.build
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.provides

ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.provides.build: ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o: ao2mo/CMakeFiles/ao2mo.dir/flags.make
ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o: ../ao2mo/nr_incore.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/ao2mo.dir/nr_incore.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_incore.c

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ao2mo.dir/nr_incore.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_incore.c > CMakeFiles/ao2mo.dir/nr_incore.c.i

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ao2mo.dir/nr_incore.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/nr_incore.c -o CMakeFiles/ao2mo.dir/nr_incore.c.s

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.requires:
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.requires

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.provides: ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.requires
	$(MAKE) -f ao2mo/CMakeFiles/ao2mo.dir/build.make ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.provides.build
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.provides

ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.provides.build: ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o: ao2mo/CMakeFiles/ao2mo.dir/flags.make
ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o: ../ao2mo/r_ao2mo.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/yuanyao/pyscf_my/pyscf/lib/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/ao2mo.dir/r_ao2mo.c.o   -c /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/r_ao2mo.c

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/ao2mo.dir/r_ao2mo.c.i"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -E /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/r_ao2mo.c > CMakeFiles/ao2mo.dir/r_ao2mo.c.i

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/ao2mo.dir/r_ao2mo.c.s"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && /usr/bin/cc  $(C_DEFINES) $(C_FLAGS) -S /home/yuanyao/pyscf_my/pyscf/lib/ao2mo/r_ao2mo.c -o CMakeFiles/ao2mo.dir/r_ao2mo.c.s

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.requires:
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.requires

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.provides: ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.requires
	$(MAKE) -f ao2mo/CMakeFiles/ao2mo.dir/build.make ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.provides.build
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.provides

ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.provides.build: ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o

# Object files for target ao2mo
ao2mo_OBJECTS = \
"CMakeFiles/ao2mo.dir/restore_eri.c.o" \
"CMakeFiles/ao2mo.dir/nr_ao2mo.c.o" \
"CMakeFiles/ao2mo.dir/nr_incore.c.o" \
"CMakeFiles/ao2mo.dir/r_ao2mo.c.o"

# External object files for target ao2mo
ao2mo_EXTERNAL_OBJECTS =

../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o
../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o
../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o
../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o
../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/build.make
../libao2mo.so: ../libcvhf.so
../libao2mo.so: ../libnp_helper.so
../libao2mo.so: /opt/libUbuntuCompat/libf77blas.so.3gf
../libao2mo.so: /opt/libUbuntuCompat/libatlas.so.3gf
../libao2mo.so: ../libcgto.so
../libao2mo.so: ../libnp_helper.so
../libao2mo.so: /opt/libUbuntuCompat/libf77blas.so.3gf
../libao2mo.so: /opt/libUbuntuCompat/libatlas.so.3gf
../libao2mo.so: ao2mo/CMakeFiles/ao2mo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C shared library ../../libao2mo.so"
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ao2mo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
ao2mo/CMakeFiles/ao2mo.dir/build: ../libao2mo.so
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/build

ao2mo/CMakeFiles/ao2mo.dir/requires: ao2mo/CMakeFiles/ao2mo.dir/restore_eri.c.o.requires
ao2mo/CMakeFiles/ao2mo.dir/requires: ao2mo/CMakeFiles/ao2mo.dir/nr_ao2mo.c.o.requires
ao2mo/CMakeFiles/ao2mo.dir/requires: ao2mo/CMakeFiles/ao2mo.dir/nr_incore.c.o.requires
ao2mo/CMakeFiles/ao2mo.dir/requires: ao2mo/CMakeFiles/ao2mo.dir/r_ao2mo.c.o.requires
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/requires

ao2mo/CMakeFiles/ao2mo.dir/clean:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo && $(CMAKE_COMMAND) -P CMakeFiles/ao2mo.dir/cmake_clean.cmake
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/clean

ao2mo/CMakeFiles/ao2mo.dir/depend:
	cd /home/yuanyao/pyscf_my/pyscf/lib/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yuanyao/pyscf_my/pyscf/lib /home/yuanyao/pyscf_my/pyscf/lib/ao2mo /home/yuanyao/pyscf_my/pyscf/lib/build /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo /home/yuanyao/pyscf_my/pyscf/lib/build/ao2mo/CMakeFiles/ao2mo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : ao2mo/CMakeFiles/ao2mo.dir/depend

