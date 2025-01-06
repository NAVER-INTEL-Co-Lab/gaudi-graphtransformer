# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2

# Include directories
INCLUDES = -Isrc -Itests -I/usr/lib/habanatools/include

# Source files
SRCS = \
    tests/implementation/sparse_matrix_mul_fwd_f32_test.cpp \
    src/entry_points.cpp \
    matrix_mul_fwd_f32.c

# Target executable
TARGET = sparse_test

# Libraries
LIBS = \
    -lpthread \
    /usr/lib/habanatools/libtpcsim_shared.so \
    /usr/lib/habanatools/libtpc_tests_core_ext.so \
    /usr/lib/habanatools/libTpcElfReader.so

# Build rules
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
