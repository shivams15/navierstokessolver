CC = mpiCC
CFLAGS = -std=c++11 -I$(PETSC_DIR)/include -I$(PETSC_DIR)/$(PETSC_ARCH)/include -I$(MPI_DIR)/include
LFLAGS = -L$(PETSC_DIR)/$(PETSC_ARCH)/lib -Wl,--rpath,$(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
OBJ = Grid.o FluidSolver.o MAIN_Solver.o

main: $(OBJ)
	$(CC) -o $@ $^ $(LFLAGS) $(CFLAGS)
	cp -f main ../bin/main

Grid.o: Grid.cpp Grid.h
	$(CC) -c -o $@ $< $(LFLAGS) $(CFLAGS) 

FluidSolver.o: FluidSolver.cpp FluidSolver.h Grid.h
	$(CC) -c -o $@ $< $(LFLAGS) $(CFLAGS)

MAIN_Solver.o: MAIN_Solver.cpp FluidSolver.h Grid.h
	$(CC) -c -o $@ $< $(LFLAGS) $(CFLAGS)

clean:
	rm -f main ../bin/main $(OBJ)