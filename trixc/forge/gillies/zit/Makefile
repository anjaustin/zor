# ZIT - Zero-Instruction Topology
# Makefile

CC = gcc
CFLAGS = -O3 -Wall -std=c99

all: zit_demo

zit_demo: zit_demo.c zit.c zit.h
	$(CC) $(CFLAGS) -o zit_demo zit_demo.c zit.c -lm

test_viz_api: test_viz_api.c zit.c zit.h
	$(CC) $(CFLAGS) -o test_viz_api test_viz_api.c zit.c -lm

test: test_viz_api
	./test_viz_api

clean:
	rm -f zit_demo test_viz_api *.o

.PHONY: all test clean
