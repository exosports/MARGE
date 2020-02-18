# Makefile - prepared for Transit
#
# The original Makefile for the Transit module (which is no longer in use) was
# created by Patricio Rojo (pato@astro.cornell.edu). This Makefile was written
# by AJ Foster (aj.foster@knights.ucf.edu).
#
# Makefile tasks:
#
# `make` - Build and compile the transit executable and python extension.
# `make clean` - Remove all compiled (non-source) files that are created.
# `make test` - Build, compile, and run the test suite.
#
# If you are interested in the commands being run by this makefile, you may add
# "VERBOSE=1" to the end of any `make` command, i.e.:
#
# 		make VERBOSE=1
#
# This will display the exact commands being used for building, etc.
#

#############
# Variables #
#############

# Compiler command
#
COMPILER = ${SCRIPTS_DIR}ccd-gcc.sh

# Directories
#
C_FILES_DIR = ./src/
H_FILES_DIR = ./include/
T_FILES_DIR = ./test/
SCRIPTS_DIR = ./scripts/

# Files to be compiled
#
C_FILES = $(wildcard $(C_FILES_DIR)*.c) \
					$(wildcard $(T_FILES_DIR)*.c)

# Files to be cleaned (non-source)
#
O_FILES = $(C_FILES_DIR)*.o \
					$(C_FILES_DIR)*.o.d \
					$(T_FILES_DIR)*.o \
					$(T_FILES_DIR)*.o.d \
					transit \
					transit.d \
					test_transit \
					./python/transit_module.py \
					./python/_transit_module.so \
					./python/transit_wrap.c \
					./src/transit_wrap.c


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
						-I/usr/include/python2.7 \
						-I./include/ \
						-I./src/ \
						-I./test/include/ \
						-I./test/ \
						-I. \
						-I../pu/include

# Library inclusions
#
LIBR_FLAG = -L/usr/X11R6/lib \
						-L../pu

# Other flags
#
OTHR_FLAG = -DTRANSIT \
						-ffast-math \
						-fgnu89-inline \
						-fPIC \
						-O2 \
						-std=c99 \
						-g3 \
						-gdwarf-2

# Combination of the previous flags
#
COMP_FLAG = $(WARN_FLAG) $(INCL_FLAG) $(LIBR_FLAG) $(OTHR_FLAG)

# Target of the complication, i.e. the name of the executable to build
#
TARGET = transit

# Library linking must be last in the GCC command
#
LINK_FLAG = -lm -lpu

# These flags relate to compiling / running the test suite
#
TEST_FLAG = -DTEST_TRANSIT


##############
# Make Tasks #
##############

# First, set verbosity
#
Q = @
O = > /dev/null
ifdef VERBOSE
	ifeq ("$(origin VERBOSE)", "command line")
		Q =
		O =
	endif
endif

# Second, make special considerations for testing
#
ifneq (,$(findstring test,$(MAKECMDGOALS)))
	COMP_FLAG += $(TEST_FLAG)
	TARGET = test_transit
endif

# Main task: `make`
#
# - Create .o files
# - Compile Transit binary
# - Run swig.sh
# - Run setup.py
#
.PHONY: all
all: swig binary python

# Clean task: `make clean`
#
# - Remove object files
# - Remove executable files
#
.PHONY: clean
clean:
	@echo "Removing non-source files."
	$(Q) rm -f $(O_FILES)

# Test task
#
.PHONY: test
test: clean binary
	@echo "Starting test suite."
	$(Q) ./test_transit

# Compile Transit Binary
#
# Called by "all"
# Depends on .o files
#
.PHONY: binary
binary: $(C_FILES:.c=.o)
	@echo "Building executable \"$(TARGET)\"."
	$(Q) $(COMPILER) $(COMP_FLAG) -o $(TARGET) $(filter %.c %.o,$^) $(LINK_FLAG)

# Python task
#
# Called by "all"
# Depends on "binary"
#
.PHONY: python
python: binary
	@echo "Building python extension."
	$(Q) python $(SCRIPTS_DIR)setup.py build_ext --inplace $(O)
	@echo "Cleaning up python build."
	$(Q) sh $(SCRIPTS_DIR)python_clean.sh

# Swig task
#
# Called by "all"
#
.PHONY: swig
swig:
	@echo "Running swig."
	$(Q) sh $(SCRIPTS_DIR)swig.sh

# Create .o files
#
# Called by "binary"
# Called by "all"
# Depends on .c files
#
.SUFFIXES:
.SUFFIXES: .c .o
%.o: %.c
	@echo "Compiling $@."
	$(Q) $(COMPILER) -c $(COMP_FLAG) -o $@ $<

