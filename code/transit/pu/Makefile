# Makefile - prepared for pu
#
# The original Makefile for the pu module (which is no longer in use) was
# created by Patricio Rojo (pato@astro.cornell.edu). This Makefile was written
# by AJ Foster (aj.foster@knights.ucf.edu).
#
# Makefile tasks:
#
# `make` - Build and compile the libpu shared object file and object archive.
# `make clean` - Remove all compiled (non-source) files that are created.
#
# If you are interested in the commands being run by this makefile, you may add
# "VERBOSE=1" to the end of any `make` command, i.e.:
#
#     make VERBOSE=1
#
# This will display the exact commands being used for building, etc.
#

#############
# Variables #
#############

# Archiver command
#
ARCHIVER = ar ruc

# Compiler command
#
COMPILER = ${SCRIPTS_DIR}ccd-gcc.sh

# Directories
#
C_FILES_DIR = ./src/
H_FILES_DIR = ./include/
SCRIPTS_DIR = ./scripts/

# Files to be compiled
#
C_FILES = iomisc \
					messagep \
					numerical \
					procopt \
					sampling \
					spline \
					voigt \
					xmalloc

# Files to be cleaned (non-source)
#
O_FILES = $(C_FILES_DIR)*.o \
					$(C_FILES_DIR)*.o.d \
					./libpu.a \
					./libpu.so.1


#####################
# Compilation Flags #
#####################

# Warning flags
#
# -Wall: Display all warnings
# -Winline: Display warnings about inlining functions
# -Wno-unused-result: Ignore results about unused function results
# -pedantic: Show warnings related to the C standard
#
WARN_FLAG = -Wall \
						-Winline \
						-Wno-unused-result \
						-pedantic

# Included directories
#
INCL_FLAG = -I/usr/X11R6/include \
						-I./include/ \
						-I./src/ \
						-I.

# Library inclusions
#
LIBR_FLAG = -L/usr/X11R6/lib

# Other flags
#
OTHR_FLAG = -ffast-math \
						-fgnu89-inline \
						-fPIC \
						-O2 \
						-std=c99 \
						-g3 \
						-gdwarf-2

# Combination of the previous flags
#
COMP_FLAG = $(WARN_FLAG) $(INCL_FLAG) $(LIBR_FLAG) $(OTHR_FLAG)

# Archive flags
#
PLATFORM := $(shell uname)
# These next options are only needed on linux builds
ifeq ($(PLATFORM), Linux)
ARCH_FLAG = -Wl,--whole-archive

# Flags for linking the m library
#
LIBM_FLAG = -Wl,--no-whole-archive -lm

# Flags for naming the library
#
NAME_FLAG = -Wl,-soname -Wl,libpu.so.1
endif

# Library linking must be last in the GCC command
#
LINK_FLAG = -lm


##############
# Make Tasks #
##############

# First, set verbosity
#
Q = @
ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
	endif
endif

# Main task: `make`
#
# - Create .o files
# - Compile pu shared object library
#
.PHONY: all
all: archive library

# Create object archive
#
# Called by "all"
# Depends on .o files
#
.PHONY: archive
archive: $(C_FILES:%=$(C_FILES_DIR)%.o)
	@echo "Creating object archive \"libpu.a\"."
	$(Q) $(ARCHIVER) libpu.a $(C_FILES:%=$(C_FILES_DIR)%.o)
	@touch libpu.a

# Clean task: `make clean`
#
# - Remove object files
# - Remove shared object files
#
.PHONY: clean
clean:
	@echo "Removing non-source files."
	$(Q) rm -f $(O_FILES)

# Compile Shared Object
#
# Called by "all"
# Depends on .o files
#
.PHONY: library
library: $(C_FILES:%=$(C_FILES_DIR)%.o)
	@echo "Building shared object \"libpu\"."
	$(Q) $(COMPILER) -shared \
			 $(ARCH_FLAG) $(C_FILES:%=$(C_FILES_DIR)%.o) \
			 $(LIBM_FLAG) \
			 $(NAME_FLAG) \
			 -o libpu.so.1 \
			 $(LINK_FLAG)

# Create .o files
#
# Called by "archive" and "library"
# Called by "all"
# Depends on .c files
#
.SUFFIXES:
.SUFFIXES: .c .o
%.o: %.c
	@echo "Compiling $@."
	$(Q) $(COMPILER) -c $(COMP_FLAG) -o $@ $<

