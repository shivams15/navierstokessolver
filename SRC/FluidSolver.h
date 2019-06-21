#ifndef __FLUIDSOLVER_H
#define __FLUIDSOLVER_H
#include "petsc.h"
#include "Grid.h"

struct FluidField{
	Vec u;
	Vec v;
	Vec phi;
	FluidField(int n){
		VecCreateSeq(PETSC_COMM_SELF,n,&u);
		VecCreateSeq(PETSC_COMM_SELF,n,&v);
		VecCreateSeq(PETSC_COMM_SELF,n,&phi);
		VecSet(u,0.0);
		VecSet(v,0.0);
		VecSet(phi,0.0);
	}
};

class FluidSolver{
private:
	FluidField *prevField;
	Grid* grid;
	double dt;	//time step
	double finalTime;	//end time 
	double re;	//Reynolds number
	int saveIter = 50;		//data is exported after every <saveIter> iterations
	Mat LHS_V, LHS_phi;			
	Vec RHS_u, RHS_v, RHS_phi;
	MatNullSpace NSP;				
	KSP uSolver, phiSolver;	//KSP solvers for u, v, and phi
	vector<vector<double>> divPhi;
	Vec convectiveDer_u0, convectiveDer_v0;
	int SolverInitialize();
	void SolverSetup();
	void ConfigureKSPSolver(KSP *solver, Mat *A);
	void ConfigureKSPSolver1(KSP *solver, Mat *A);
	void ConstructGhostStencils();
	void ConstructLHS();
	void AddGhostStencils(int, int, int, double);
	double EvaluateGhostStencil_V(double* V, int i, int j, int e, int d);
	double EvaluateGhostStencil_P(double* V, int i, int j, int e);
	void DiffusiveFlux(vector<double>& D, int i, int j, int d);
	void ConvectiveFlux(vector<double>& C, int i, int j);
	vector<double> SlopeLimiter(double* u, double* v, int i, int j, int d, vector<double>& h);
	void ConstructRHS_V();
	void ConstructRHS_phi(Vec& u_star, Vec& v_star);
	double Div_V(Vec& Ux, Vec& Uy, int i, int j);
	vector<double> GradP(int i, int j);
	void ApplyBoundaryConditions(int, int);
	void CorrectVelocities(Vec& Ux, Vec& Uy);
	void ExportData(int iter);
	void CreateMatrix(Mat *A, int m, int n);
	bool ParseDataFile(ifstream& fs);
public:
	bool setup = false;
	FluidSolver(char* fname, Grid* grid);
	void Solve();
};

double minmode(double a, double b);

#endif