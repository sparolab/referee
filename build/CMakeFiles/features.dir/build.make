# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build

# Include any dependencies generated for this target.
include CMakeFiles/features.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/features.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/features.dir/flags.make

CMakeFiles/features.dir/src/features.cpp.o: CMakeFiles/features.dir/flags.make
CMakeFiles/features.dir/src/features.cpp.o: /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp/src/features.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/features.dir/src/features.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/features.dir/src/features.cpp.o -c /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp/src/features.cpp

CMakeFiles/features.dir/src/features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/features.dir/src/features.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp/src/features.cpp > CMakeFiles/features.dir/src/features.cpp.i

CMakeFiles/features.dir/src/features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/features.dir/src/features.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp/src/features.cpp -o CMakeFiles/features.dir/src/features.cpp.s

# Object files for target features
features_OBJECTS = \
"CMakeFiles/features.dir/src/features.cpp.o"

# External object files for target features
features_EXTERNAL_OBJECTS =

libfeatures.a: CMakeFiles/features.dir/src/features.cpp.o
libfeatures.a: CMakeFiles/features.dir/build.make
libfeatures.a: CMakeFiles/features.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libfeatures.a"
	$(CMAKE_COMMAND) -P CMakeFiles/features.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/features.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/features.dir/build: libfeatures.a

.PHONY : CMakeFiles/features.dir/build

CMakeFiles/features.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/features.dir/cmake_clean.cmake
.PHONY : CMakeFiles/features.dir/clean

CMakeFiles/features.dir/depend:
	cd /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/cpp /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build /media/bhbhchoi/6EEA0A20EA09E4E5/Research/Project/ReFeree_journal/github/referee/build/CMakeFiles/features.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/features.dir/depend

