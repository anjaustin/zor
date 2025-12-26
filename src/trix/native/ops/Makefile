# TriX Native Ops - Build System
#
# Targets:
#   make          - Build shared library
#   make test     - Build and run tests
#   make bench    - Build and run benchmarks
#   make clean    - Remove build artifacts

CC = gcc
CFLAGS = -O3 -march=native -fPIC -Wall -Wextra
LDFLAGS = -shared -lm

# Output
LIB = libtrix_ops.so
TEST = test_ops
BENCH = bench_ops

# Sources
SRCS = trix_ops.c
OBJS = $(SRCS:.c=.o)

.PHONY: all clean test bench

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c trix_ops.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Test executable
test: test_ops.c $(LIB)
	$(CC) $(CFLAGS) -o $(TEST) test_ops.c -L. -ltrix_ops -lm -Wl,-rpath,.
	./$(TEST)

# Benchmark executable  
bench: bench_ops.c $(LIB)
	$(CC) $(CFLAGS) -o $(BENCH) bench_ops.c -L. -ltrix_ops -lm -Wl,-rpath,.
	./$(BENCH)

clean:
	rm -f $(OBJS) $(LIB) $(TEST) $(BENCH)
