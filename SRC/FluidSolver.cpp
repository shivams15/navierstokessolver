#include <iostream>
#include <string>
#include "FluidSolver.h"
#include "petscksp.h"

using namespace std;

FluidSolver::FluidSolver(char* fname, Grid* grid){
	this->grid = grid;
	ifstream fs{fname};
	if(ParseDataFile(fs) && SolverInitialize()) SolverSetup();
	fs.close();
}

//Initializes the FluidSolver object with user-provided parameters
int FluidSolver::SolverInitialize(){
	if(dt <= 0){
		cout << "Time step should be positive\n";
		return 0;
	}
	else if(dt > finalTime){
		cout << "Time step should be less than final time!\n";
		return 0;
	}
	else if(re <= 0){
		cout << "Reynolds number should be positive\n";
		return 0;
	}
	else if(saveIter <= 0){
		cout << "saveIter must be greater than zero!\n";
		return 0;
	}
	else{
		prevField = new FluidField(grid->N);
		CreateMatrix(&LHS_V, grid->N, grid->N);
		CreateMatrix(&LHS_phi, grid->N, grid->N);
		VecCreateSeq(PETSC_COMM_SELF, grid->N, &RHS_u);
		VecCreateSeq(PETSC_COMM_SELF, grid->N, &RHS_v);
		VecCreateSeq(PETSC_COMM_SELF, grid->N, &RHS_phi);
		VecCreateSeq(PETSC_COMM_SELF, grid->N, &convectiveDer_u0);
		VecCreateSeq(PETSC_COMM_SELF, grid->N, &convectiveDer_v0);
		VecSet(convectiveDer_u0, 0.0);
		VecSet(convectiveDer_v0, 0.0);
		divPhi = vector<vector<double>>(grid->N,{0,0});
		return 1;
	}
}

void FluidSolver::SolverSetup(){
	ConstructGhostStencils();
	ConstructLHS();
	ConfigureKSPSolver(&uSolver, &LHS_V);
	ConfigureKSPSolver1(&phiSolver, &LHS_phi);
	// KSPSetInitialGuessNonzero(phiSolver, PETSC_TRUE);

	cout<<"Solver Setup Complete!\n";
	setup =true;
}

//Sets the solver type, preconditioner, tolerance, and other options
void FluidSolver::ConfigureKSPSolver(KSP *solver, Mat *A){
	PC pc;
	KSPCreate(PETSC_COMM_SELF, solver);
	KSPSetOperators(*solver, *A, *A);
	KSPSetType(*solver, KSPGMRES);
	KSPGetPC(*solver, &pc);
	PCSetType(pc, PCILU);
	KSPSetTolerances(*solver, 1.e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetUp(*solver);
}

//Sets the solver type, preconditioner, tolerance, and other options
void FluidSolver::ConfigureKSPSolver1(KSP *solver, Mat *A){
	PC pc;
	KSPCreate(PETSC_COMM_SELF, solver);
	KSPSetOperators(*solver, *A, *A);
	KSPSetType(*solver, KSPBCGSL);
	KSPGetPC(*solver, &pc);
	PCSetType(pc, PCBJACOBI);
	KSPSetTolerances(*solver, 1.e-8, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetUp(*solver);
}

void FluidSolver::ConstructGhostStencils(){
	for(auto& i: grid->edges){
		if(i.bcType <= WALL){
			i.ghost.emplace_back(Stencil{{-1},{{0,0}}});
			i.ghost.emplace_back(Stencil{{1},{{0,0}}});
			if(i.bcType == INLET_UNI) {
				if(i.nx == 0) i.ghost[0].constant = {0,2*i.bcInfo};
				else i.ghost[0].constant = {2*i.bcInfo,0};
			}
			else if(i.bcType == WALL) {
				if(i.nx != 0) i.ghost[0].constant = {0,2*i.bcInfo};
				else i.ghost[0].constant = {2*i.bcInfo,0};
			}
		}
		else if(i.bcType == NEUMANN){
			i.ghost.emplace_back(Stencil{{1},{{0,0}},{0,0}});
			i.ghost.emplace_back(Stencil{{2.5,-2.0,0.5},{{0,0},{-i.nx,-i.ny},{-2*i.nx,-2*i.ny}}});
		}
	}
}

void FluidSolver::ConstructLHS(){
	auto& c = grid->cells;
	double w = 0.0;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	vector<int> Nx = {-1,1,0,0};
	vector<int> Ny = {0,0,-1,1};

	for(int i = 0; i < c.size(); i++){
		for(int j = 0; j < c[i].size(); j++){
			if(c[i][j].id < 0) continue;
			for(int k = 0; k < 4; k++){
				int nx = Nx[k], ny = Ny[k];
				if(grid->inDomain(i+nx,j+ny)){
					if(nx != 0) w = 2.0/(hx[i]*(hx[i]+hx[i+nx]));
					else w = 2.0/(hy[j]*(hy[j]+hy[j+ny]));
					MatSetValues(LHS_V,1,&(c[i][j].id),1,&(c[i+nx][j+ny].id),&w,ADD_VALUES);
					MatSetValues(LHS_phi,1,&(c[i][j].id),1,&(c[i+nx][j+ny].id),&w,ADD_VALUES);
				}
				else{
					if(nx != 0) w = 1.0/pow(hx[i],2);
					else w = 1.0/pow(hy[j],2);
					AddGhostStencils(i,j,c[i][j].edges[k],w);
				}
				w = -w;
				MatSetValues(LHS_V,1,&(c[i][j].id),1,&(c[i][j].id),&w,ADD_VALUES);
				MatSetValues(LHS_phi,1,&(c[i][j].id),1,&(c[i][j].id),&w,ADD_VALUES);
			}
		}
	}

	MatAssemblyBegin(LHS_V, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_V, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(LHS_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_phi, MAT_FINAL_ASSEMBLY);

	MatScale(LHS_V,-dt/(2*re));
	MatShift(LHS_V,1.0);
	MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, 0, &NSP);
	MatSetNullSpace(LHS_phi, NSP);
	MatSetTransposeNullSpace(LHS_phi, NSP);
}

void FluidSolver::AddGhostStencils(int i, int j, int e, double w){
	auto& c = grid->cells;
	Edge& E = grid->edges[e];
	Stencil& s0 = E.ghost[0];
	Stencil& s1 = E.ghost[1];

	for(int l = 0; l < s0.weights.size(); l++) {
		double w1 = w*s0.weights[l];
		int id = c[i+s0.support[l][0]][j+s0.support[l][1]].id;
		MatSetValues(LHS_V,1,&(c[i][j].id),1,&id,&w1,ADD_VALUES);
	}

	for(int l = 0; l < s1.weights.size(); l++) {
		double w1 = w*s1.weights[l];
		int id = c[i+s1.support[l][0]][j+s1.support[l][1]].id;
		MatSetValues(LHS_phi,1,&(c[i][j].id),1,&id,&w1,ADD_VALUES);
	}
}

double FluidSolver::EvaluateGhostStencil_V(double* V, int i, int j, int e, int d){
	auto& c = grid->cells;
	Stencil& s = grid->edges[e].ghost[0];
	double r {0.0};
	for(int l = 0; l < s.weights.size(); l++) r += s.weights[l]*V[c[i+s.support[l][0]][j+s.support[l][1]].id];
	r += s.constant[d];
	return r;
}

double FluidSolver::EvaluateGhostStencil_P(double* V, int i, int j, int e){
	auto& c = grid->cells;
	Stencil& s = grid->edges[e].ghost[1];
	double r {0.0};
	for(int l = 0; l < s.weights.size(); l++) r += s.weights[l]*V[c[i+s.support[l][0]][j+s.support[l][1]].id];
	return r;
}

void FluidSolver::DiffusiveFlux(vector<double>& D, int i, int j, int d){
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	double* V;

	if(d == 0) VecGetArray(prevField->u,&V);
	else VecGetArray(prevField->v,&V);

	if(grid->inDomain(i-1,j)) D[0] = (1/re)*(V[c[i][j].id] - V[c[i-1][j].id])/(hx[i] + hx[i-1]);
	else D[0] = (0.5/re/hx[i])*(V[c[i][j].id] - EvaluateGhostStencil_V(V,i,j,c[i][j].edges[0],d));
	if(grid->inDomain(i+1,j)) D[1] = (1/re)*(V[c[i+1][j].id] - V[c[i][j].id])/(hx[i] + hx[i+1]);
	else D[1] = -(0.5/re/hx[i])*(V[c[i][j].id] - EvaluateGhostStencil_V(V,i,j,c[i][j].edges[1],d));
	if(grid->inDomain(i,j-1)) D[2] = (1/re)*(V[c[i][j].id] - V[c[i][j-1].id])/(hy[j] + hy[j-1]);
	else D[2] = (0.5/re/hy[i])*(V[c[i][j].id] - EvaluateGhostStencil_V(V,i,j,c[i][j].edges[2],d));
	if(grid->inDomain(i,j+1)) D[3] = (1/re)*(V[c[i][j+1].id] - V[c[i][j].id])/(hy[j] + hy[j+1]);
	else D[3] = -(0.5/re/hy[i])*(V[c[i][j].id] - EvaluateGhostStencil_V(V,i,j,c[i][j].edges[3],d));

	if(d == 0) VecRestoreArray(prevField->u, &V);
	else VecRestoreArray(prevField->v, &V);
}

void FluidSolver::ConvectiveFlux(vector<double>& C, int i, int j){
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	double *u, *v, u1, u2, v1, v2;
	VecGetArray(prevField->u, &u);
	VecGetArray(prevField->v, &v);

	vector<double> SL = SlopeLimiter(u,v,i,j,0,hx);
	u2 = u[c[i][j].id] - hx[i]/2*SL[0];
	v2 = v[c[i][j].id] - hx[i]/2*SL[1];

	if(c[i][j].edges[0] == -1){
		auto sl = SlopeLimiter(u,v,i-1,j,0,hx);
		u1 = u[c[i-1][j].id] + hx[i-1]/2*sl[0];
		v1 = v[c[i-1][j].id] + hx[i-1]/2*sl[1];
	}
	else{
		u1 = 0.5*(u[c[i][j].id] + EvaluateGhostStencil_V(u,i,j,c[i][j].edges[0],0));
		v1 = 0.5*(v[c[i][j].id] + EvaluateGhostStencil_V(v,i,j,c[i][j].edges[0],1));
	}

	C[0] = 0.5*(pow(u1,2)+pow(u2,2) - abs(u1+u2)*(u2-u1));
	C[1] = 0.5*(u1*v1 + u2*v2 - 0.5*abs(v1+v2)*(u2-u1) - 0.5*abs(u2+u1)*(v2-v1));

	u1 = u[c[i][j].id] + hx[i]/2*SL[0];
	v1 = v[c[i][j].id] + hx[i]/2*SL[1];

	if(c[i][j].edges[1] == -1){
		auto sl = SlopeLimiter(u,v,i+1,j,0,hx);
		u2 = u[c[i+1][j].id] - hx[i+1]/2*sl[0];
		v2 = v[c[i+1][j].id] - hx[i+1]/2*sl[1];
	}
	else{
		u2 = 0.5*(u[c[i][j].id] + EvaluateGhostStencil_V(u,i,j,c[i][j].edges[1],0));
		v2 = 0.5*(v[c[i][j].id] + EvaluateGhostStencil_V(v,i,j,c[i][j].edges[1],1));
	}

	C[2] = 0.5*(pow(u1,2)+pow(u2,2) - abs(u1+u2)*(u2-u1));
	C[3] = 0.5*(u1*v1 + u2*v2 - 0.5*abs(v1+v2)*(u2-u1) - 0.5*abs(u2+u1)*(v2-v1));

	SL = SlopeLimiter(u,v,i,j,1,hy);
	u2 = u[c[i][j].id] - hy[j]/2*SL[0];
	v2 = v[c[i][j].id] - hy[j]/2*SL[1];

	if(c[i][j].edges[2] == -1){
		auto sl = SlopeLimiter(u,v,i,j-1,1,hy);
		u1 = u[c[i][j-1].id] + hy[j-1]/2*sl[0];
		v1 = v[c[i][j-1].id] + hy[j-1]/2*sl[1];
	}
	else{
		u1 = 0.5*(u[c[i][j].id] + EvaluateGhostStencil_V(u,i,j,c[i][j].edges[2],0));
		v1 = 0.5*(v[c[i][j].id] + EvaluateGhostStencil_V(v,i,j,c[i][j].edges[2],1));
	}

	C[5] = 0.5*(pow(v1,2)+pow(v2,2) - abs(v1+v2)*(v2-v1));
	C[4] = 0.5*(u1*v1 + u2*v2 - 0.5*abs(v1+v2)*(u2-u1) - 0.5*abs(u2+u1)*(v2-v1));

	u1 = u[c[i][j].id] + hy[j]/2*SL[0];
	v1 = v[c[i][j].id] + hy[j]/2*SL[1];

	if(c[i][j].edges[3] == -1){
		auto sl = SlopeLimiter(u,v,i,j+1,1,hy);
		u2 = u[c[i][j+1].id] - hy[j+1]/2*sl[0];
		v2 = v[c[i][j+1].id] - hy[j+1]/2*sl[1];
	}
	else{
		u2 = 0.5*(u[c[i][j].id] + EvaluateGhostStencil_V(u,i,j,c[i][j].edges[3],0));
		v2 = 0.5*(v[c[i][j].id] + EvaluateGhostStencil_V(v,i,j,c[i][j].edges[3],1));
	}

	C[7] = 0.5*(pow(v1,2)+pow(v2,2) - abs(v1+v2)*(v2-v1));
	C[6] = 0.5*(u1*v1 + u2*v2 - 0.5*abs(v1+v2)*(u2-u1) - 0.5*abs(u2+u1)*(v2-v1));

	VecRestoreArray(prevField->u, &u);
	VecRestoreArray(prevField->v, &v);
}

vector<double> FluidSolver::SlopeLimiter(double* u, double* v, int i, int j, int d, vector<double>& h){
	auto& c = grid->cells;	
	double u1, u2, v1, v2;
	vector<double> a(2), b(2);

	if(d == 0){
		if(grid->inDomain(i+1,j)) {
			a[0] = 2*(u[c[i+1][j].id] - u[c[i][j].id])/(h[i+1]+h[i]);
			a[1] = 2*(v[c[i+1][j].id] - v[c[i][j].id])/(h[i+1]+h[i]);
		}
		else {
			a[0] = (EvaluateGhostStencil_V(u,i,j,c[i][j].edges[1],0) - u[c[i][j].id])/h[i];
			a[1] = (EvaluateGhostStencil_V(v,i,j,c[i][j].edges[1],1) - v[c[i][j].id])/h[i];
		}
		if(grid->inDomain(i-1,j)) {
			b[0] = 2*(u[c[i][j].id] - u[c[i-1][j].id])/(h[i-1]+h[i]);
			b[1] = 2*(v[c[i][j].id] - v[c[i-1][j].id])/(h[i-1]+h[i]);
		}
		else {
			b[0] = (u[c[i][j].id] - EvaluateGhostStencil_V(u,i,j,c[i][j].edges[0],0))/h[i];
			b[1] = (v[c[i][j].id] - EvaluateGhostStencil_V(v,i,j,c[i][j].edges[0],1))/h[i];
		}
	}
	else{
		if(grid->inDomain(i,j+1)) {
			a[0] = 2*(u[c[i][j+1].id] - u[c[i][j].id])/(h[j+1]+h[j]);
			a[1] = 2*(v[c[i][j+1].id] - v[c[i][j].id])/(h[j+1]+h[j]);
		}
		else {
			a[0] = (EvaluateGhostStencil_V(u,i,j,c[i][j].edges[3],0) - u[c[i][j].id])/h[j];
			a[1] = (EvaluateGhostStencil_V(v,i,j,c[i][j].edges[3],1) - v[c[i][j].id])/h[j];
		}
		if(grid->inDomain(i,j-1)) {
			b[0] = 2*(u[c[i][j].id] - u[c[i][j-1].id])/(h[j-1]+h[j]);
			b[1] = 2*(v[c[i][j].id] - v[c[i][j-1].id])/(h[j-1]+h[j]);
		}
		else {
			b[0] = (u[c[i][j].id] - EvaluateGhostStencil_V(u,i,j,c[i][j].edges[2],0))/h[j];
			b[1] = (v[c[i][j].id] - EvaluateGhostStencil_V(v,i,j,c[i][j].edges[2],1))/h[j];
		}
	}
	return {minmode(a[0],b[0]),minmode(a[1],b[1])};
}

void FluidSolver::ConstructRHS_V(){
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	vector<double> D(4);
	vector<double> C(8);
	double v;

	VecSet(RHS_u,0.0);
	VecSet(RHS_v,0.0);
	VecAXPY(RHS_u,1.0,prevField->u);
	VecAXPY(RHS_v,1.0,prevField->v);
	VecAXPY(RHS_u,0.5*dt,convectiveDer_u0);
	VecAXPY(RHS_v,0.5*dt,convectiveDer_v0);

	for(int i = 0; i < c.size(); i++){
		for(int j = 0; j < c[i].size(); j++){
			if(c[i][j].id < 0) continue;
			DiffusiveFlux(D,i,j,0);
			v = dt*((D[1]-D[0])/hx[i] + (D[3]-D[2])/hy[j]);
			VecSetValues(RHS_u,1,&(c[i][j].id),&v,ADD_VALUES);
			DiffusiveFlux(D,i,j,1);
			v = dt*((D[1]-D[0])/hx[i] + (D[3]-D[2])/hy[j]);
			VecSetValues(RHS_v,1,&(c[i][j].id),&v,ADD_VALUES);
			ConvectiveFlux(C,i,j);
			v = (C[2]-C[0])/hx[i] + (C[6]-C[4])/hy[j];
			VecSetValues(convectiveDer_u0,1,&(c[i][j].id),&v,INSERT_VALUES);
			v *= -1.5*dt;
			VecSetValues(RHS_u,1,&(c[i][j].id),&v,ADD_VALUES);
			v = (C[3]-C[1])/hx[i] + (C[7]-C[5])/hy[j];
			VecSetValues(convectiveDer_v0,1,&(c[i][j].id),&v,INSERT_VALUES);
			v *= -1.5*dt;
			VecSetValues(RHS_v,1,&(c[i][j].id),&v,ADD_VALUES);
			ApplyBoundaryConditions(i,j);
		}
	}
}

void FluidSolver::ConstructRHS_phi(Vec& Ux, Vec& Uy){
	auto& c = grid->cells;
	double v;

	VecSet(RHS_phi,0.0);

	for(int i = 0; i < c.size(); i++){
		for(int j = 0; j < c[i].size(); j++){
			if(c[i][j].id < 0) continue;
			v = Div_V(Ux, Uy,i,j)/dt;
			VecSetValues(RHS_phi,1,&(c[i][j].id),&v,INSERT_VALUES);
		}
	}
}

double FluidSolver::Div_V(Vec& Ux, Vec& Uy, int i, int j){
	vector<double> V(4);
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	double *u, *v, r;

	VecGetArray(Ux, &u);
	VecGetArray(Uy, &v);

	if(c[i][j].edges[0] == -1){
		r = hx[i]/(hx[i-1]+hx[i]);
		V[0] = u[c[i-1][j].id]*r + u[c[i][j].id]*(1-r);
	}
	else V[0] = 0.5*(u[c[i][j].id]+EvaluateGhostStencil_V(u,i,j,c[i][j].edges[0],0));

	if(c[i][j].edges[1] == -1){
		r = hx[i]/(hx[i+1]+hx[i]);
		V[1] = u[c[i+1][j].id]*r + u[c[i][j].id]*(1-r);
	}
	else V[1] = 0.5*(u[c[i][j].id]+EvaluateGhostStencil_V(u,i,j,c[i][j].edges[1],0));

	if(c[i][j].edges[2] == -1){
		double r = hy[j]/(hy[j-1]+hy[j]);
		V[2] = v[c[i][j-1].id]*r + v[c[i][j].id]*(1-r);
	}
	else V[2] = 0.5*(v[c[i][j].id]+EvaluateGhostStencil_V(v,i,j,c[i][j].edges[2],1));

	if(c[i][j].edges[3] == -1){
		double r = hy[j]/(hy[j+1]+hy[j]);
		V[3] = v[c[i][j+1].id]*r + v[c[i][j].id]*(1-r);
	}
	else V[3] = 0.5*(v[c[i][j].id]+EvaluateGhostStencil_V(v,i,j,c[i][j].edges[3],1));

	VecRestoreArray(Ux, &u);
	VecRestoreArray(Uy, &v);

	return (V[1]-V[0])/hx[i] + (V[3]-V[2])/hy[j];
}

vector<double> FluidSolver::GradP(int i, int j){
	vector<double> V(4);
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	double *p, r;

	VecGetArray(prevField->phi, &p);

	if(c[i][j].edges[0] == -1){
		r = hx[i]/(hx[i-1]+hx[i]);
		V[0] = p[c[i-1][j].id]*r + p[c[i][j].id]*(1-r);
	}
	else V[0] = 0.5*(p[c[i][j].id]+EvaluateGhostStencil_P(p,i,j,c[i][j].edges[0]));

	if(c[i][j].edges[1] == -1){
		r = hx[i]/(hx[i+1]+hx[i]);
		V[1] = p[c[i+1][j].id]*r + p[c[i][j].id]*(1-r);
	}
	else V[1] = 0.5*(p[c[i][j].id]+EvaluateGhostStencil_P(p,i,j,c[i][j].edges[1]));

	if(c[i][j].edges[2] == -1){
		double r = hy[j]/(hy[j-1]+hy[j]);
		V[2] = p[c[i][j-1].id]*r + p[c[i][j].id]*(1-r);
	}
	else V[2] = 0.5*(p[c[i][j].id]+EvaluateGhostStencil_P(p,i,j,c[i][j].edges[2]));

	if(c[i][j].edges[3] == -1){
		double r = hy[j]/(hy[j+1]+hy[j]);
		V[3] = p[c[i][j+1].id]*r + p[c[i][j].id]*(1-r);
	}
	else V[3] = 0.5*(p[c[i][j].id]+EvaluateGhostStencil_P(p,i,j,c[i][j].edges[3]));

	VecRestoreArray(prevField->phi, &p);

	return {(V[1]-V[0])/hx[i], (V[3]-V[2])/hy[j]};
}

void FluidSolver::ApplyBoundaryConditions(int i, int j){
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	bool calc {false};
	double w, W, D;

	for(auto& k: c[i][j].edges){
		if(k == -1) continue;
		Edge& e = grid->edges[k];
		if(!calc){
			if(e.nx != 0){
				if(!grid->inDomain(i,j+1)) D = 2.0*(divPhi[c[i][j].id][0] - divPhi[c[i][j-1].id][0])/(hy[j]+hy[j-1]);
				else if (!grid->inDomain(i,j-1)) D = 2.0*(divPhi[c[i][j+1].id][0] - divPhi[c[i][j].id][0])/(hy[j]+hy[j+1]);
				else D = divPhi[c[i][j+1].id][0]/(hy[j]+hy[j+1]) - divPhi[c[i][j-1].id][0]/(hy[j]+hy[j-1]) - divPhi[c[i][j].id][0]*(1/(hy[j]+hy[j+1]) - 1/(hy[j]+hy[j-1]));
			}
			else{
				if(!grid->inDomain(i+1,j)) D = 2.0*(divPhi[c[i][j].id][1] - divPhi[c[i-1][j].id][1])/(hx[i]+hx[i-1]);
				else if (!grid->inDomain(i-1,j)) D = 2.0*(divPhi[c[i+1][j].id][1] - divPhi[c[i][j].id][1])/(hx[i]+hx[i+1]);
				else D = divPhi[c[i+1][j].id][1]/(hx[i]+hx[i+1]) - divPhi[c[i-1][j].id][1]/(hx[i]+hx[i-1]) - divPhi[c[i][j].id][1]*(1/(hx[i]+hx[i+1]) - 1/(hx[i]+hx[i-1]));
			}
			calc = true;
		}
		if(e.bcType == NEUMANN){
			if(e.nx != 0){
				w = dt*e.nx*hx[i]*D;
				W = dt*(0.5/re/pow(hx[i],2))*w;
				VecSetValues(RHS_v,1,&(c[i][j].id),&W,ADD_VALUES);
			}
			else{
				w = dt*e.ny*hy[j]*D;
				W = dt*(0.5/re/pow(hy[j],2))*w;
				VecSetValues(RHS_u,1,&(c[i][j].id),&W,ADD_VALUES);
			}
		}
		else{
			if(e.nx != 0){
				W = dt*(0.5/re/pow(hx[i],2))*e.ghost[0].constant[0];
				VecSetValues(RHS_u,1,&(c[i][j].id),&W,ADD_VALUES);
				w = 2*dt*(divPhi[c[i][j].id][1] + e.nx*hx[i]*D/2);
				W = dt*(0.5/re/pow(hx[i],2))*(e.ghost[0].constant[1] + w);
				VecSetValues(RHS_v,1,&(c[i][j].id),&W,ADD_VALUES);
			}
			else{
				W = dt*(0.5/re/pow(hy[j],2))*e.ghost[0].constant[1];
				VecSetValues(RHS_v,1,&(c[i][j].id),&W,ADD_VALUES);
				w = 2*dt*(divPhi[c[i][j].id][0] + e.ny*hy[j]*D/2);
				W = dt*(0.5/re/pow(hy[j],2))*(e.ghost[0].constant[0] + w);
				VecSetValues(RHS_u,1,&(c[i][j].id),&W,ADD_VALUES);
			}
		}
	}
}

void FluidSolver::CorrectVelocities(Vec& Ux, Vec& Uy){
	auto& c = grid->cells;
	double *u, *v, r;
	vector<double> g(2);

	VecGetArray(Ux,&u);
	VecGetArray(Uy,&v);

	for(int i = 0; i < c.size(); i++){
		for(int j = 0; j < c[i].size(); j++){
			if(c[i][j].id < 0) continue;
			g = GradP(i,j);
			divPhi[c[i][j].id] = g;
			r = u[c[i][j].id] - dt*g[0];
			VecSetValues(prevField->u,1,&(c[i][j].id),&r,INSERT_VALUES);
			r = v[c[i][j].id] - dt*g[1];
			VecSetValues(prevField->v,1,&(c[i][j].id),&r,INSERT_VALUES);
		}
	}

	VecRestoreArray(Ux,&u);
	VecRestoreArray(Uy,&v);
}

void FluidSolver::Solve(){
	int iter = 1;
	Vec u,v;
	double umax, umin, vmax, vmin;
	VecCreateSeq(PETSC_COMM_SELF,grid->N,&u);
	VecCreateSeq(PETSC_COMM_SELF,grid->N,&v);

	cout<<"Initiating Solver..\n";

	do{
		ConstructRHS_V();
		KSPSolve(uSolver, RHS_u, u);
		KSPSolve(uSolver, RHS_v, v);
		ConstructRHS_phi(u,v);
		MatNullSpaceRemove(NSP, RHS_phi);
		KSPSolve(phiSolver, RHS_phi, prevField->phi);
		CorrectVelocities(u, v);

		VecMin(prevField->u, NULL, &umin);
		VecMin(prevField->v, NULL, &vmin);
		VecMax(prevField->u, NULL, &umax);
		VecMax(prevField->v, NULL, &vmax);

		if((iter-1)%10 == 0) printf("iter\tumin\t\tumax\t\tvmin\t\tvmax\n");
		printf("%d\t%lf\t%lf\t%lf\t%lf\n", iter, umin, umax, vmin, vmax);
		if(saveIter > 0 && iter%saveIter == 0) ExportData(iter);

		iter++;
	}while(dt*iter <= finalTime);

	cout<<"Solution Complete!\n";
}

void FluidSolver::ExportData(int iter){
	double *u, *v, *lPhi, *p, *phi;
	char f1[255];
	auto& c = grid->cells;
	auto& hx = grid->hx;
	auto& hy = grid->hy;
	Vec lapPhi, P;

	VecCreateSeq(PETSC_COMM_SELF,grid->N, &lapPhi);
	VecCreateSeq(PETSC_COMM_SELF,grid->N, &P);
	MatMult(LHS_phi,prevField->phi, lapPhi);
	VecCopy(prevField->phi, P);
	VecAXPY(P, -dt/(2*re), lapPhi);
	VecGetArray(prevField->u, &u);
	VecGetArray(prevField->v, &v);
	VecGetArray(lapPhi, &lPhi);
	VecGetArray(P, &p);
	VecGetArray(prevField->phi, &phi);

	sprintf(f1, "FlowData_%d.csv", iter);
	ofstream fs{f1};
	fs<<"Point_X,Point_Y,Point_Z,U,V,Pr\n";

	for(int i = 0; i < c.size(); i++){
		for(int j = 0; j < c[i].size(); j++){
			if(c[i][j].id < 0) continue;
			fs<<c[i][j].x<<","<<c[i][j].y<<","<<0.0<<","<<u[c[i][j].id]<<","<<v[c[i][j].id]<<","<<p[c[i][j].id]<<endl;
			for(auto& k: c[i][j].edges){
				if(k < 0) continue;
				Edge& e = grid->edges[k];
				fs<<c[i][j].x + e.nx*hx[i]/2<<","<<c[i][j].y + e.ny*hy[j]/2<<","<<0.0<<","
						<<0.5*(u[c[i][j].id] + EvaluateGhostStencil_V(u,i,j,k,0))<<","
						<<0.5*(v[c[i][j].id] + EvaluateGhostStencil_V(v,i,j,k,1))<<","
						<<0.5*(phi[c[i][j].id] + EvaluateGhostStencil_P(phi,i,j,k))-dt*lPhi[c[i][j].id]/(2*re)<<endl;
			}
		}
	}

	fs.close();
	VecRestoreArray(prevField->u,&u);
	VecRestoreArray(prevField->v,&v);
	VecRestoreArray(lapPhi, &lPhi);
	VecRestoreArray(P, &p);
	VecRestoreArray(prevField->phi, &phi);
	VecDestroy(&lapPhi);
	VecDestroy(&P);
}

void FluidSolver::CreateMatrix(Mat *A, int n1, int n2){
	MatCreate(PETSC_COMM_SELF, A);
	MatSetType(*A, MATAIJ);
	MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n1, n2);
	MatSetFromOptions(*A);
	MatSetUp(*A);
}

//Reads user parameters from the data file provided
bool FluidSolver::ParseDataFile(ifstream& fs){
	string inp{};
	stringstream s;
	bool err = false;

	if(!fs){
		cout << "Simulation data file not found!\n";
		return false;
	}

	while(!fs.eof()){
		fs>>inp;
		if(fs.eof()) break;
		if(inp == "BC") {
			if((fs>>ws).get() != '{') {err = true; break;}
			int i = 0;
			while(fs.good()) {
				if((fs>>ws).peek() == '}') {
					if(i == grid->edges.size()) fs.ignore(); 
					else err = true;
					break;
				}
				getline(fs, inp);
				if(fs.eof() || i == grid->edges.size()) {err = true; break;}
				s.str(inp);
				s>>grid->edges[i].bcType>>grid->edges[i].bcInfo;
				s.clear();
				i++;
			}
		}
		else if(inp == "dt") fs>>dt;
		else if(inp == "final_time") fs>>finalTime;
		else if(inp == "re") fs>>re;
		else if(inp == "saveIter") fs>>saveIter;
		else {err = true; break;}

		if(!fs.good()) {err = true; break;}
	}
	if(err){
		cout<<"Invalid data file format!\n";
		return false;
	}
	return true;
}

double minmode(double a, double b){
	if(a*b > 0) return (a*min(1.0, abs(b/a)));
	else return 0;
}