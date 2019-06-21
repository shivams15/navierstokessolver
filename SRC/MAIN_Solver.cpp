#include <iostream>
#include "Grid.h"
#include "FluidSolver.h"

using namespace std;

int main(int argc, char *argv[])
{
	PetscErrorCode ierr;
	ierr = PetscInitialize(&argc, &argv, (char *)0, "Initializing Program");
	if(argc < 3) cout << "Grid data file or simulation data file not provided!\n";
	else{
		Grid grid {argv[1]};
		if(!grid.setup) cout << "Grid setup failed!\n";
		else{
			FluidSolver fSolver {argv[2], &grid};
			if(fSolver.setup) fSolver.Solve();
		}
	}

	return 0;
}