#include <iostream>
#include <fstream>
#include <string.h>
#include "petsc.h"
#include "Grid.h"
#include "FluidSolver.h"

using namespace std;

int main(int argc, char *argv[])
{
	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char *)0, "Initializing Program");
	if(argc < 3)
		cout << "Grid data file or simulation data file not provided!\n";
	else{
		Grid *grid = new Grid(argv[1]);
		if(!grid->setup) cout << "Grid setup failed!\n";
		else{
			FluidSolver *fSolver = new FluidSolver(argv[2], grid);
			if(fSolver->setup) fSolver->Solve();
		}
	}

	return 0;
}