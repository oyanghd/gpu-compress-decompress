SUBDIRS = CUDAMPI CUDA MPI Serial

all:
	@for dir in $(SUBDIRS); do \
		make -C $$dir; \
	done

clean:
	@for dir in $(SUBDIRS); do \
		make clean -C $$dir; \
	done

.PHONY: all clean $(SUBDIRS)