# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build

# Include any dependencies generated for this target.
include CMakeFiles/Resizer_static.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/Resizer_static.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Resizer_static.dir/flags.make

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o: CMakeFiles/Resizer_static.dir/flags.make
CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o: /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/progs/demos/Resizer/Resizer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o -c /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/progs/demos/Resizer/Resizer.cpp

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/progs/demos/Resizer/Resizer.cpp > CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.i

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0/progs/demos/Resizer/Resizer.cpp -o CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.s

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.requires:

.PHONY : CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.requires

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.provides: CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.requires
	$(MAKE) -f CMakeFiles/Resizer_static.dir/build.make CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.provides.build
.PHONY : CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.provides

CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.provides.build: CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o


# Object files for target Resizer_static
Resizer_static_OBJECTS = \
"CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o"

# External object files for target Resizer_static
Resizer_static_EXTERNAL_OBJECTS =

bin/Resizer_static: CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o
bin/Resizer_static: CMakeFiles/Resizer_static.dir/build.make
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libGLU.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libGL.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libSM.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libICE.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libX11.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXext.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXrandr.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXxf86vm.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXi.so
bin/Resizer_static: lib/libglut.a
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libGL.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libSM.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libICE.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libX11.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXext.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXrandr.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXxf86vm.so
bin/Resizer_static: /usr/lib/x86_64-linux-gnu/libXi.so
bin/Resizer_static: CMakeFiles/Resizer_static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin/Resizer_static"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Resizer_static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Resizer_static.dir/build: bin/Resizer_static

.PHONY : CMakeFiles/Resizer_static.dir/build

CMakeFiles/Resizer_static.dir/requires: CMakeFiles/Resizer_static.dir/progs/demos/Resizer/Resizer.cpp.o.requires

.PHONY : CMakeFiles/Resizer_static.dir/requires

CMakeFiles/Resizer_static.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Resizer_static.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Resizer_static.dir/clean

CMakeFiles/Resizer_static.dir/depend:
	cd /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0 /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/freeglut-3.0.0 /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build /media/david/Hard_Disk/ENSTA/ANNEE_2/IN203/Promotion_2021/TravauxPratiques/Projet/Projet_laby_ants/libgui/thirdparty/build/CMakeFiles/Resizer_static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Resizer_static.dir/depend

