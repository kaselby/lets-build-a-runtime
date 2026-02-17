CC = cc
CFLAGS = -O2 -Wall -fPIC -DACCELERATE_NEW_LAPACK
CSRC = ../csrc

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	LDFLAGS = -shared -framework Accelerate
	LIB_EXT = dylib
else
	LDFLAGS = -shared -lopenblas
	LIB_EXT = so
endif

all: libablation.$(LIB_EXT)

OPS_SRC = $(wildcard $(CSRC)/ops/*.c)

libablation.$(LIB_EXT): ablation.c $(OPS_SRC)
	$(CC) $(CFLAGS) $(LDFLAGS) -I$(CSRC) -o $@ $^

clean:
	rm -f libablation.$(LIB_EXT)

.PHONY: all clean
