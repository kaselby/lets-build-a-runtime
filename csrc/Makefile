CC = cc
CFLAGS = -O2 -Wall -fPIC -DACCELERATE_NEW_LAPACK

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	# macOS: use Accelerate framework for CBLAS
	LDFLAGS = -shared -framework Accelerate
	LIB_EXT = dylib
else
	# Linux: use OpenBLAS
	LDFLAGS = -shared -lopenblas
	LIB_EXT = so
endif

SRCS = ops.c

all: libruntime.$(LIB_EXT) libexecutor.$(LIB_EXT) libablation.$(LIB_EXT)

# Per-op kernels (loaded by c_backend.py)
libruntime.$(LIB_EXT): $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compiled executor: dispatch loop + kernels (loaded by executor.py)
libexecutor.$(LIB_EXT): executor.c $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Ablation: variant kernels + parametric dispatch (loaded by bench script)
libablation.$(LIB_EXT): ablation.c $(SRCS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

clean:
	rm -f libruntime.$(LIB_EXT) libexecutor.$(LIB_EXT) libablation.$(LIB_EXT)

.PHONY: all clean
