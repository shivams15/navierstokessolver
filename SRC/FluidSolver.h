#ifndef __FLUIDSOLVER_H
#define __FLUIDSOLVER_H
#include "petsc.h"
#include "Grid.h"

struct FluidField{
	Vec u;
	Vec v;
	Vec phi;
	Vec p;
};

class FluidSolver{
private:
	Grid *grid;
	double dt;
	double finalTime;
	double re;
	double uLeft, vLeft;
	double uRight, vRight;
	double uBottom, vBottom;
	double uTop, vTop;
	Mat LHS_u, LHS_v, LHS_phi;
	Vec RHS_u, RHS_v, RHS_phi;
	Mat lap_u, lap_v, lap_phi;
	Mat bc_u, bc_v, bc_phi;
	Mat dudx, dvdy, dpdx, dpdy;
	Mat dt_u, dt_v;
	KSP uSolver, vSolver, phiSolver;
	double ub[4] = {0.0, 0.0, 0.0, 0.0};
	double vb[4] = {0.0, 0.0, 0.0, 0.0};
	int ParseDataFile(FILE *f1);
	int SolverInitialize();
	void SolverSetup();
	void CreateFluidField(FluidField *fField);
	void CreateMatrix(Mat *A, int m, int n);
	int* StencilLaplacian(int i, int j, char *var);
	bool ApplyGoverningEquation(int i, int j, char *var);
	// void ConstructRHS();
	// void ConstructRHS_u();
	// void ConstructRHS_v();
	// void ConstructRHS_phi();
public:
	FluidField *prevField;
	FluidField *nextField;
	FluidSolver(char *fname, Grid *grid);
	void Solve();
	int bcType[4] = {-1, -1, -1, -1};
	bool setup = false;
};

#endif