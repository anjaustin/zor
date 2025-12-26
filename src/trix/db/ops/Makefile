# DB Cooper Native Operations
# NEON-optimized ternary similarity

CC ?= gcc
CFLAGS = -O3 -Wall -Wextra -fPIC

# Detect architecture
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_M),arm64)
    CFLAGS += -march=armv8-a
    SIMD = NEON
else ifeq ($(UNAME_M),aarch64)
    CFLAGS += -march=armv8-a
    SIMD = NEON
else ifeq ($(UNAME_M),x86_64)
    CFLAGS += -mavx2
    SIMD = AVX2
else
    SIMD = scalar
endif

# Shared library extension
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = dylib
    LIB_FLAGS = -dynamiclib
else
    LIB_EXT = so
    LIB_FLAGS = -shared
endif

TARGET = libcooper.$(LIB_EXT)
SOURCES = db_cooper_ops.c
HEADERS = db_cooper_ops.h

.PHONY: all clean test info

all: $(TARGET)
	@echo "Built $(TARGET) with $(SIMD) support"

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(LIB_FLAGS) -o $@ $(SOURCES)

test: $(TARGET) test_cooper.c
	$(CC) $(CFLAGS) -o test_cooper test_cooper.c -L. -lcooper -Wl,-rpath,.
	./test_cooper

clean:
	rm -f $(TARGET) test_cooper *.o

info:
	@echo "Architecture: $(UNAME_M)"
	@echo "SIMD: $(SIMD)"
	@echo "Target: $(TARGET)"
