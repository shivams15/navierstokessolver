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
	FluidField *prevField, *nextField;
	Grid* grid;
	double dt;	//time step
	double finalTime;	//end time 
	double re;	//Reynolds number
	Mat LHS_u, LHS_v, LHS_phi;			
	Vec RHS_u, RHS_v, RHS_phi;
	Mat lap_u, lap_v, lap_phi;	
	Mat dudx, dvdy, dpdx, dpdy;
	Mat dt_u, dt_v;
	MatNullSpace NSP;				
	KSP uSolver, vSolver, phiSolver;	//KSP solvers for u, v, and phi
	int bcType[4] = {-1, -1, -1, -1};	//type of boundary conditions on each boundary
	double ub[4] = {0.0, 0.0, 0.0, 0.0};	//boundary values of u(if specified)
	double vb[4] = {0.0, 0.0, 0.0, 0.0};	//boundary values of v(if specified)
	int saveIter = 50;		//data is exported after every <saveIter> iterations
	bool ParseDataFile(ifstream& fs);
	int SolverInitialize();
	void ProcessGrid();
	void SolverSetup();
	void CreateFluidField(FluidField*& fField);
	void CreateMatrix(Mat *A, int m, int n);
	int* StencilLaplacian(int i, int j, const char *var);
	int StencilLaplacian_PressureGhostPoint(int i, int j, int* stencil, double* weights);
	bool ApplyGoverningEquation(int i, int j, const char *var);
	void ConfigureKSPSolver(KSP *solver, Mat *A);
	void ConfigureKSPSolver1(KSP *solver, Mat *A);
	void ConstructLHS_u();
	void ConstructLHS_v();
	void ConstructLHS_phi();
	void ConstructRHS_u();
	void ConstructRHS_v();
	void ConstructRHS_phi(Vec *u_star, Vec *v_star);
	double ConvectiveDerivative_u(int i, int j, PetscScalar *u, PetscScalar *v, Points& pts);
	double ConvectiveDerivative_v(int i, int j, PetscScalar *u, PetscScalar *v, Points& pts);
	void ExportData(int iter, FluidField *field);
	void VecMinMax(double* min, double* max, Points& pts, int n1, int n2, Vec* u);
	double SlopeLimiter(int i, int j, Points& pts, PetscScalar *var, const char *dir);
	double minmode(double a, double b);
public:
	bool setup = false;
	FluidSolver(char* fname, Grid* grid);
	void Solve();
};

#endif