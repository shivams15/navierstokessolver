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
	FluidField *prevField;
	FluidField *nextField;
	Grid *grid;
	double dt;
	double finalTime;
	double re;
	Mat LHS_u, LHS_v, LHS_phi;
	Vec RHS_u, RHS_v, RHS_phi;
	Mat lap_u, lap_v, lap_phi;
	Mat bc_u, bc_v, bc_phi;
	Mat dudx, dvdy, dpdx, dpdy;
	Mat dt_u, dt_v;
	MatNullSpace NSP;
	KSP uSolver, vSolver, phiSolver;
	int bcType[4] = {-1, -1, -1, -1};
	double ub[4] = {0.0, 0.0, 0.0, 0.0};
	double vb[4] = {0.0, 0.0, 0.0, 0.0};
	int saveIter = 50;
	int ParseDataFile(FILE *f1);
	int SolverInitialize();
	void ProcessGrid();
	void SolverSetup();
	void CreateFluidField(FluidField **fField);
	void CreateMatrix(Mat *A, int m, int n);
	int* StencilLaplacian(int i, int j, const char *var);
	bool ApplyGoverningEquation(int i, int j, const char *var);
	void ConfigureKSPSolver(KSP *solver, Mat *A);
	void ConfigureKSPSolver1(KSP *solver, Mat *A);
	void ConstructLHS_u();
	void ConstructLHS_v();
	void ConstructLHS_phi();
	void ConstructRHS_u();
	void ConstructRHS_v();
	void ConstructRHS_phi(Vec *u_star, Vec *v_star);
	double ConvectiveDerivative_u(int i, int j, PetscScalar *u, PetscScalar *v, Point **pts);
	double ConvectiveDerivative_v(int i, int j, PetscScalar *u, PetscScalar *v, Point **pts);
	void ExportData(int iter, FluidField *field);
	double SlopeLimiter(int i, int j, Point **pts, PetscScalar *var, const char *dir);
	double minmode(double a, double b);
public:
	bool setup = false;
	FluidSolver(char *fname, Grid *grid);
	void Solve();
};

#endif