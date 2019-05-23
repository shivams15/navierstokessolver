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

//Creates a new FluidField structure
void FluidSolver::CreateFluidField(FluidField*& fField){
	fField = new FluidField;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &(fField->u));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &(fField->v));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(fField->phi));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(fField->p));
	VecSet(fField->u,0.0);
	VecSet(fField->v,0.0);
	VecSet(fField->phi,0.0);
	VecSet(fField->p,0.0);
}

//Assigns an 'id' to each point in u, v, and p grids
void FluidSolver::ProcessGrid(){
	int tmp = 0;
	int nx = grid->nx;
	int ny = grid->ny;
	Points* pts = &grid->ugrid;

	for(int i = 0; i < ny + 2; i++){
 		for(int j = 0; j < nx + 3; j++){
 			if((*pts)[i][j].type == GHOST_CORNER)
 				continue;
 			else if((*pts)[i][j].type == GHOST_LEFT && bcType[0] == DIRICHLET)
 				continue;
 			else if((*pts)[i][j].type == GHOST_RIGHT && bcType[1] == DIRICHLET)
 				continue;
 			else{
 				(*pts)[i][j].id = tmp;
 				tmp++;
 			}
 		}
 	}
 	grid->nPoints[0] = tmp;

 	tmp = 0;
 	pts = &grid->vgrid;
 	for(int i = 0; i < ny + 3; i++){
 		for(int j = 0; j < nx + 2; j++){
 			if((*pts)[i][j].type == GHOST_CORNER)
 				continue;
 			else if((*pts)[i][j].type == GHOST_BOTTOM && bcType[2] == DIRICHLET)
 				continue;
 			else if((*pts)[i][j].type == GHOST_TOP && bcType[3] == DIRICHLET)
 				continue;
 			else{
 				(*pts)[i][j].id = tmp;
 				tmp++;
 			}
 		}
 	}
 	grid->nPoints[1] = tmp;

 	tmp = 0;
 	pts = &grid->pgrid;
 	for(int i = 0; i < ny + 2; i++){
 		for(int j = 0; j < nx + 2; j++){
 			if((*pts)[i][j].type == GHOST_CORNER)
 				continue;
 			else{
 				(*pts)[i][j].id = tmp;
 				tmp++;
 			}
 		}
 	}
 	grid->nPoints[2] = tmp;
}

//Configures all solvers
void FluidSolver::SolverSetup(){	
	ConstructLHS_u();
	ConstructLHS_v();
	ConstructLHS_phi();

	ConfigureKSPSolver(&uSolver, &LHS_u);
	ConfigureKSPSolver(&vSolver, &LHS_v);
	ConfigureKSPSolver1(&phiSolver, &LHS_phi);
	KSPSetInitialGuessNonzero(phiSolver, PETSC_TRUE);

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
	KSPSetTolerances(*solver, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
	KSPSetUp(*solver);
}

/*
Constructs the LHS for the x-momentum equation, with the pressure gradient term completely ignored
A semi-implicit numerical method is used. 
Viscous fluxes are approximated using the Crank-Nicholson scheme.
Convective fluxes are discretized explicity.
*/
void FluidSolver::ConstructLHS_u(){
	int nx = grid->nx;
	int ny = grid->ny;
	double hx = grid->hx;
	double hy = grid->hy;

	double lapWeights[5] = {1.0/pow(hx,2), 1.0/pow(hx,2), -2.0/pow(hx,2) -2.0/pow(hy,2), 1.0/pow(hy,2), 1.0/pow(hy,2)}; 
	double dtWeight[1] = {1.0/dt};
	double unit[1] = {1.0};
	double weights[3] = {0.0, 0.0, 0.0};
	int stencil[3] = {0, 0, 0};

	Points& pts = grid->ugrid;

	for(int i = 0; i < ny + 2; i++){
		for(int j = 0; j < nx + 3; j++){
			if(ApplyGoverningEquation(i, j, "u")){
				MatSetValues(lap_u,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"u"),(PetscScalar *)lapWeights,INSERT_VALUES);
				MatSetValues(dt_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)dtWeight,INSERT_VALUES);
				weights[0] = 1.0/hx; weights[1] = -1.0/hx;
				stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
				MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT){
					MatSetValues(LHS_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					weights[0] = 1.0/hx; weights[1] = -1.0/hx;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
					MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_LEFT && bcType[0] == NEUMANN){
					weights[0] = 1.0; weights[1] = -1.0;
					stencil[0] = pts[i][j].id; stencil[1] = pts[i][j+2].id; 
					MatSetValues(LHS_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					weights[0] = -2.0/hx; weights[1] = 3.0/hx; weights[2] = -1.0/hx;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j+1].id; stencil[2] = grid->pgrid[i][j+2].id;
					MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT && bcType[1] == NEUMANN){
					weights[0] = 1.0; weights[1] = -1.0;
					stencil[0] = pts[i][j].id; stencil[1] = pts[i][j-2].id; 
					MatSetValues(LHS_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					weights[0] = 2.0/hx; weights[1] = -3.0/hx; weights[2] = 1.0/hx;
					stencil[0] = grid->pgrid[i][j-1].id; stencil[1] = grid->pgrid[i][j-2].id; stencil[2] = grid->pgrid[i][j-3].id;
					MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_BOTTOM){
					stencil[0] = pts[i][j].id; stencil[1] = pts[i+1][j].id;
					if(bcType[2] == DIRICHLET){
						weights[0] = 1.0; 
						weights[1] = 1.0;
					}
					else{
						weights[0] = 1.0; 
						weights[1] = -1.0;
					}
					MatSetValues(LHS_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(j == 1){
						weights[0] = -2.0/hx; weights[1] = 3.0/hx; weights[2] = -1.0/hx;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j+1].id; stencil[2] = grid->pgrid[i][j+2].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else if(j == nx+1){
						weights[0] = 2.0/hx; weights[1] = -3.0/hx; weights[2] = 1.0/hx;
						stencil[0] = grid->pgrid[i][j-1].id; stencil[1] = grid->pgrid[i][j-2].id; stencil[2] = grid->pgrid[i][j-3].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else{
						weights[0] = 1.0/hx; weights[1] = -1.0/hx;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
				else if(pts[i][j].type == GHOST_TOP){
					stencil[0] = pts[i][j].id; stencil[1] = pts[i-1][j].id; 
					if(bcType[3] == DIRICHLET){
						weights[0] = 1.0; 
						weights[1] = 1.0;
					}
					else{
						weights[0] = 1.0; 
						weights[1] = -1.0;
					}
					MatSetValues(LHS_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(j == 1){
						weights[0] = -2.0/hx; weights[1] = 3.0/hx; weights[2] = -1.0/hx;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j+1].id; stencil[2] = grid->pgrid[i][j+2].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else if(j == nx+1){
						weights[0] = 2.0/hx; weights[1] = -3.0/hx; weights[2] = 1.0/hx;
						stencil[0] = grid->pgrid[i][j-1].id; stencil[1] = grid->pgrid[i][j-2].id; stencil[2] = grid->pgrid[i][j-3].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else{
						weights[0] = 1.0/hx; weights[1] = -1.0/hx;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
			}
		}
	}

	MatAssemblyBegin(LHS_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dt_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dt_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dpdx, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dpdx, MAT_FINAL_ASSEMBLY);

	MatAXPY(LHS_u, dt, dt_u, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_u, -dt/(2*re), lap_u, DIFFERENT_NONZERO_PATTERN);
}

/*
Constructs the LHS for the v-momentum equation, with the pressure term completely ignored
A semi-implicit numerical method is used. 
Viscous fluxes are approximated using the Crank-Nicholson scheme.
Convective fluxes are discretized explicity.
*/
void FluidSolver::ConstructLHS_v(){
	int nx = grid->nx;
	int ny = grid->ny;
	double hx = grid->hx;
	double hy = grid->hy;

	double lapWeights[5] = {1.0/pow(hx,2), 1.0/pow(hx,2), -2.0/pow(hx,2) -2.0/pow(hy,2), 1.0/pow(hy,2), 1.0/pow(hy,2)}; 
	double dtWeight[1] = {1.0/dt};
	double unit[1] = {1.0};
	double weights[3] = {0.0, 0.0, 0.0};
	int stencil[3] = {0, 0, 0};

	Points& pts = grid->vgrid;

	for(int i = 0; i < ny + 3; i++){
		for(int j = 0; j < nx + 2; j++){
			if(ApplyGoverningEquation(i, j, "v")){
				MatSetValues(lap_v,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"v"),(PetscScalar *)lapWeights,INSERT_VALUES);
				MatSetValues(dt_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)dtWeight,INSERT_VALUES);
				weights[0] = 1.0/hy; weights[1] = -1.0/hy;
				stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
				MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT){
					MatSetValues(LHS_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					weights[0] = 1.0/hy; weights[1] = -1.0/hy;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
					MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_BOTTOM && bcType[2] == NEUMANN){
					weights[0] = 1.0; weights[1] = -1.0;
					stencil[0] = pts[i][j].id; stencil[1] = pts[i+2][j].id; 
					MatSetValues(LHS_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					weights[0] = -2.0/hy; weights[1] = 3.0/hy; weights[2] = -1.0/hy;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i+1][j].id; stencil[2] = grid->pgrid[i+2][j].id;
					MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP && bcType[3] == NEUMANN){
					weights[0] = 1.0; weights[1] = -1.0;
					stencil[0] = pts[i][j].id; stencil[1] = pts[i-2][j].id; 
					MatSetValues(LHS_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					weights[0] = 2.0/hy; weights[1] = -3.0/hy; weights[2] = 1.0/hy;
					stencil[0] = grid->pgrid[i-1][j].id; stencil[1] = grid->pgrid[i-2][j].id; stencil[2] = grid->pgrid[i-3][j].id;
					MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_LEFT){
					stencil[0] = pts[i][j].id; stencil[1] = pts[i][j+1].id; 
					if(bcType[0] == DIRICHLET){
						weights[0] = 1.0; 
						weights[1] = 1.0;
					}
					else{
						weights[0] = 1.0; 
						weights[1] = -1.0;
					}
					MatSetValues(LHS_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(i == 1){
						weights[0] = -2.0/hy; weights[1] = 3.0/hy; weights[2] = -1.0/hy;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i+1][j].id; stencil[2] = grid->pgrid[i+2][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else if(i == ny+1){
						weights[0] = 2.0/hy; weights[1] = -3.0/hy; weights[2] = 1.0/hy;
						stencil[0] = grid->pgrid[i-1][j].id; stencil[1] = grid->pgrid[i-2][j].id; stencil[2] = grid->pgrid[i-3][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else{
						weights[0] = 1.0/hy; weights[1] = -1.0/hy;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
				else if(pts[i][j].type == GHOST_RIGHT){
					stencil[0] = pts[i][j].id; stencil[1] = pts[i][j-1].id; 
					if(bcType[1] == DIRICHLET){
						weights[0] = 1.0; 
						weights[1] = 1.0;
					}
					else{
						weights[0] = 1.0; 
						weights[1] = -1.0;
					}
					MatSetValues(LHS_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(i == 1){
						weights[0] = -2.0/hy; weights[1] = 3.0/hy; weights[2] = -1.0/hy;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i+1][j].id; stencil[2] = grid->pgrid[i+2][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else if(i == ny+1){
						weights[0] = 2.0/hy; weights[1] = -3.0/hy; weights[2] = 1.0/hy;
						stencil[0] = grid->pgrid[i-1][j].id; stencil[1] = grid->pgrid[i-2][j].id; stencil[2] = grid->pgrid[i-3][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					else{
						weights[0] = 1.0/hy; weights[1] = -1.0/hy;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
			}
		}
	}

	MatAssemblyBegin(LHS_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dt_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dt_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dpdy, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dpdy, MAT_FINAL_ASSEMBLY);

	MatAXPY(LHS_v, dt, dt_v, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_v, -dt/(2*re), lap_v, DIFFERENT_NONZERO_PATTERN);
}

/*
Constructs the LHS for the Poisson equation for phi
*/
void FluidSolver::ConstructLHS_phi(){
	int nx = grid->nx;
	int ny = grid->ny;
	double hx = grid->hx;
	double hy = grid->hy;

	double lapWeights[5] = {1.0/pow(hx,2), 1.0/pow(hx,2), -2.0/pow(hx,2) -2.0/pow(hy,2), 1.0/pow(hy,2), 1.0/pow(hy,2)}; 
	double unit[1] = {1.0};
	double weights[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	int stencil[7] = {0, 0, 0, 0, 0, 0, 0};

	Points& pts = grid->pgrid;
	for(int i = 0; i < ny + 2; i++){
		for(int j = 0; j < nx + 2; j++){
			if(ApplyGoverningEquation(i, j, "p")){
				MatSetValues(lap_phi,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"p"),(PetscScalar *)lapWeights,INSERT_VALUES);
				MatSetValues(LHS_phi,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"p"),(PetscScalar *)lapWeights,INSERT_VALUES);
				weights[0] = 1.0/hx; weights[1] = -1.0/hx;
				stencil[0] = grid->ugrid[i][j+1].id; stencil[1] = grid->ugrid[i][j].id;
				MatSetValues(dudx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				weights[0] = 1.0/hy; weights[1] = -1.0/hy;
				stencil[0] = grid->vgrid[i+1][j].id; stencil[1] = grid->vgrid[i][j].id;
				MatSetValues(dvdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
			}
			else{
				weights[0] = 1.0; weights[1] = -1.0; weights[2] = 0;
				stencil[0] = pts[i][j].id;
				if(pts[i][j].type == GHOST_BOTTOM){
					stencil[1] = pts[i+1][j].id; stencil[2] = pts[i+2][j].id; 
					if(bcType[2] == NEUMANN) weights[1] = -2.0; weights[2] = 1.0;
				}
				else if(pts[i][j].type == GHOST_TOP){
					stencil[1] = pts[i-1][j].id; stencil[2] = pts[i-2][j].id;
					if(bcType[3] == NEUMANN) weights[1] = -2.0; weights[2] = 1.0; 
				}
				else if(pts[i][j].type == GHOST_LEFT){
					stencil[1] = pts[i][j+1].id; stencil[2] = pts[i][j+2].id; 
					if(bcType[0] == NEUMANN) weights[1] = -2.0; weights[2] = 1.0; 
				}
				else if(pts[i][j].type == GHOST_RIGHT){
					stencil[1] = pts[i][j-1].id; stencil[2] = pts[i][j-2].id;
					if(bcType[1] == NEUMANN) weights[1] = -2.0; weights[2] = 1.0;   
				}
				if(pts[i][j].type != GHOST_CORNER){
					MatSetValues(LHS_phi,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
					MatSetValues(lap_phi,1,&(pts[i][j].id),StencilLaplacian_PressureGhostPoint(i,j,stencil,weights),stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
			}
		}
	}

	MatAssemblyBegin(LHS_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dudx, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dudx, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dvdy, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dvdy, MAT_FINAL_ASSEMBLY);

	MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_TRUE, 0, 0, &NSP);
	MatSetNullSpace(LHS_phi, NSP);
	MatSetTransposeNullSpace(LHS_phi, NSP);

}

//Constructs the RHS for the x-momentum equation with the pressure gradient term ignored
void FluidSolver::ConstructRHS_u(){
	VecSet(RHS_u, 0.0);
	Vec uLap, dudt, bc, temp, convectiveDer;
	double tmp;
	double hx = grid->hx;
	double hy = grid->hy;
	PetscScalar *dphidx = new PetscScalar[grid->nPoints[0]];
	PetscScalar *u = new PetscScalar[grid->nPoints[0]];
	PetscScalar *v = new PetscScalar[grid->nPoints[1]];
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &uLap);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &dudt);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &bc);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &temp);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &convectiveDer);

	VecSet(bc, 0.0);
	VecSet(convectiveDer, 0.0);

	MatMult(lap_u, prevField->u, uLap);
	MatMult(dt_u, prevField->u, dudt);
	MatMult(dpdx, prevField->phi, temp);
	VecGetArray(temp, &dphidx);
	VecGetArray(prevField->u, &u);
	VecGetArray(prevField->v, &v);

	Points& pts = grid->ugrid;

	for(int i = 0; i < grid->ny+2; i++){
		for(int j = 0; j < grid->nx+3; j++){
			if(ApplyGoverningEquation(i, j, "u")){
				tmp = ConvectiveDerivative_u(i, j, u, v, pts);
				VecSetValues(convectiveDer, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT)
					VecSetValues(bc, 1, &(pts[i][j].id), &ub[pts[i][j].type-1], INSERT_VALUES);
				else if(pts[i][j].type == GHOST_BOTTOM){
					if(bcType[2] == DIRICHLET) tmp = 2*(ub[2] + 0.5*dt*(dphidx[pts[i][j].id] + dphidx[pts[i+1][j].id]));
					else tmp = dt*(dphidx[pts[i][j].id] - dphidx[pts[i+1][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP){
					if(bcType[3] == DIRICHLET) tmp = 2*(ub[3] + 0.5*dt*(dphidx[pts[i][j].id] + dphidx[pts[i-1][j].id]));
					else tmp = dt*(dphidx[pts[i][j].id] - dphidx[pts[i-1][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
			}
		}
	}

	VecAXPY(RHS_u, dt, dudt);
	VecAXPY(RHS_u, dt/(2*re), uLap);
	VecAXPY(RHS_u, -dt, convectiveDer);
	VecAXPY(RHS_u, 1.0, bc);

	VecDestroy(&uLap);
	VecDestroy(&convectiveDer);
	VecDestroy(&bc);
	VecDestroy(&temp);
	VecDestroy(&dudt);
}

//Constructs the RHS for the y-momentum equation with the pressure gradient term ignored
void FluidSolver::ConstructRHS_v(){
	VecSet(RHS_v, 0.0);

	Vec vLap, dvdt, bc, temp, convectiveDer;
	double tmp;
	double hx = grid->hx;
	double hy = grid->hy;
	PetscScalar *dphidy = new PetscScalar[grid->nPoints[1]];
	PetscScalar *u = new PetscScalar[grid->nPoints[0]];
	PetscScalar *v = new PetscScalar[grid->nPoints[1]];
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &vLap);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &dvdt);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &bc);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &temp);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &convectiveDer);

	VecSet(bc, 0.0);
	VecSet(convectiveDer, 0.0);

	MatMult(lap_v, prevField->v, vLap);
	MatMult(dt_v, prevField->v, dvdt);
	MatMult(dpdy, prevField->phi, temp);
	VecGetArray(temp, &dphidy);
	VecGetArray(prevField->u, &u);
	VecGetArray(prevField->v, &v);

	Points& pts = grid->vgrid;

	for(int i = 0; i < grid->ny+3; i++){
		for(int j = 0; j < grid->nx+2; j++){
			if(ApplyGoverningEquation(i, j, "v")){
				tmp = ConvectiveDerivative_v(i, j, u, v, pts);
				VecSetValues(convectiveDer, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT)
					VecSetValues(bc, 1, &(pts[i][j].id), &vb[pts[i][j].type-1], INSERT_VALUES);
				else if(pts[i][j].type == GHOST_LEFT){
					if(bcType[0] == DIRICHLET) tmp = 2*(vb[0] + 0.5*dt*(dphidy[pts[i][j].id] + dphidy[pts[i][j+1].id]));
					else tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i][j+1].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT){
					if(bcType[1] == DIRICHLET) tmp = 2*(vb[1] + 0.5*dt*(dphidy[pts[i][j].id] + dphidy[pts[i][j-1].id]));
					else tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i][j-1].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
			}
		}
	}

	VecAXPY(RHS_v, dt, dvdt);
	VecAXPY(RHS_v, dt/(2*re), vLap);
	VecAXPY(RHS_v, -dt, convectiveDer);
	VecAXPY(RHS_v, 1.0, bc);

	VecDestroy(&vLap);
	VecDestroy(&convectiveDer);
	VecDestroy(&bc);
	VecDestroy(&temp);
	VecDestroy(&dvdt);
}

//Constructs the RHS for the Poisson equation for phi
void FluidSolver::ConstructRHS_phi(Vec *u_star, Vec *v_star){
	VecSet(RHS_phi, 0.0);

	Vec dUdX, dVdY;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &dUdX);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &dVdY);

	MatMult(dudx, *u_star, dUdX);
	MatMult(dvdy, *v_star, dVdY);

	VecAXPY(RHS_phi, 1.0/dt, dUdX);
	VecAXPY(RHS_phi, 1.0/dt, dVdY);

	VecDestroy(&dUdX);
	VecDestroy(&dVdY);
}

/*
Calculates the convective flux in x-direction using Roe's flux difference splitting.
Van Leer's MUSCL scheme is used to approximate flux at the cell faces to second order accuracy.
*/
double FluidSolver::ConvectiveDerivative_u(int i, int j, PetscScalar *u, PetscScalar *v, Points& pts){
	double urr, url, ulr, ull, v_U, v_D, E_R, E_L, E_U, E_D, der;
	double hx = grid->hx;
	double hy = grid->hy;

	url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "x")*hx/2;
	ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "x")*hx/2;

	if(ApplyGoverningEquation(i, j-1, "u"))
		ull = u[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, u, "x")*hx/2;
	else 
		ull = 0.5*(u[pts[i][j-1].id] + u[pts[i][j].id]);

	if(ApplyGoverningEquation(i, j+1, "u"))
		urr = u[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, u, "x")*hx/2;
	else 
		urr = 0.5*(u[pts[i][j+1].id] + u[pts[i][j].id]);

	E_R = 0.5*(pow(urr,2) + pow(url,2) - abs(urr + url)*(urr - url));
	E_L = 0.5*(pow(ulr,2) + pow(ull,2) - abs(ulr + ull)*(ulr - ull));

	url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "y")*hy/2;
	ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "y")*hy/2;

	if(ApplyGoverningEquation(i-1, j, "u"))
		ull = u[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, u, "y")*hy/2;
	else 
		ull = 0.5*(u[pts[i-1][j].id] + u[pts[i][j].id]);

	if(ApplyGoverningEquation(i+1, j, "u"))
		urr = u[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, u, "y")*hy/2;
	else 
		urr = 0.5*(u[pts[i+1][j].id] + u[pts[i][j].id]);

	v_U = 0.5*(v[grid->vgrid[i+1][j].id] + v[grid->vgrid[i+1][j-1].id]);
	v_D = 0.5*(v[grid->vgrid[i][j].id] + v[grid->vgrid[i][j-1].id]);
	E_U = 0.5*(v_U*(urr + url) - abs(v_U)*(urr - url));
	E_D = 0.5*(v_D*(ulr + ull) - abs(v_D)*(ulr - ull));
	der = (E_R - E_L)/hx + (E_U - E_D)/hy;

	return der;
}

/*
Calculates the convective flux in y-direction using Roe's flux difference splitting.
Van Leer's MUSCL scheme is used to approximate flux at the cell faces to second order accuracy.
*/
double FluidSolver::ConvectiveDerivative_v(int i, int j, PetscScalar *u, PetscScalar *v, Points& pts){
	double vrr, vrl, vlr, vll, u_R, u_L, E_R, E_L, E_U, E_D, der;
	double hx = grid->hx;
	double hy = grid->hy;

	vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "x")*hx/2;
	vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "x")*hx/2;

	if(ApplyGoverningEquation(i, j-1, "v")) 
		vll = v[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, v, "x")*hx/2;
	else 
		vll = 0.5*(v[pts[i][j-1].id] + v[pts[i][j].id]);

	if(ApplyGoverningEquation(i, j+1, "v"))
		vrr = v[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, v, "x")*hx/2;
	else 
		vrr = 0.5*(v[pts[i][j+1].id] + v[pts[i][j].id]);

	u_R = 0.5*(u[grid->ugrid[i][j+1].id] + u[grid->ugrid[i-1][j+1].id]);
	u_L = 0.5*(u[grid->ugrid[i][j].id] + u[grid->ugrid[i-1][j].id]);
	E_R = 0.5*(u_R*(vrr + vrl) - abs(u_R)*(vrr - vrl));
	E_L = 0.5*(u_L*(vlr + vll) - abs(u_L)*(vlr - vll));

	vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "y")*hy/2;
	vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "y")*hy/2;

	if(ApplyGoverningEquation(i-1, j, "v"))
		vll = v[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, v, "y")*hy/2;
	else 
		vll = 0.5*(v[pts[i-1][j].id] + v[pts[i][j].id]);

	if(ApplyGoverningEquation(i+1, j, "v"))
		vrr = v[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, v, "y")*hy/2;
	else 
		vrr = 0.5*(v[pts[i+1][j].id] + v[pts[i][j].id]);

	E_U = 0.5*(pow(vrr,2) + pow(vrl,2) - abs(vrr + vrl)*(vrr - vrl));
	E_D = 0.5*(pow(vlr,2) + pow(vll,2) - abs(vlr + vll)*(vlr - vll));

	der = (E_R - E_L)/hx + (E_U - E_D)/hy;

	return der;
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
	else if(bcType[0] == -1 || bcType[1] == -1 || bcType[2] == -1 || bcType[3] == -1){
		cout << "Specify boundary conditions at all boundaries!\n";
		return 0;
	}
	else if(saveIter <= 0){
		cout << "saveIter must be greater than zero!\n";
		return 0;
	}
	else{
		ProcessGrid();
		CreateFluidField(prevField);
		CreateFluidField(nextField);
		CreateMatrix(&LHS_u, grid->nPoints[0], grid->nPoints[0]);
		CreateMatrix(&lap_u, grid->nPoints[0], grid->nPoints[0]);
		CreateMatrix(&dt_u, grid->nPoints[0], grid->nPoints[0]);
		CreateMatrix(&LHS_v, grid->nPoints[1], grid->nPoints[1]);
		CreateMatrix(&lap_v, grid->nPoints[1], grid->nPoints[1]);
		CreateMatrix(&dt_v, grid->nPoints[1], grid->nPoints[1]);
		CreateMatrix(&LHS_phi, grid->nPoints[2], grid->nPoints[2]);
		CreateMatrix(&lap_phi, grid->nPoints[2], grid->nPoints[2]);
		CreateMatrix(&dudx, grid->nPoints[2], grid->nPoints[0]);
		CreateMatrix(&dvdy, grid->nPoints[2], grid->nPoints[1]);
		CreateMatrix(&dpdx, grid->nPoints[0], grid->nPoints[2]);
		CreateMatrix(&dpdy, grid->nPoints[1], grid->nPoints[2]);
		VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &RHS_u);
		VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &RHS_v);
		VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &RHS_phi);
		return 1;
	}
}

//Solves the system of equations to obtain the velocity and pressure fields after the specified number of time steps
void FluidSolver::Solve(){
	int iter = 1;
	PetscReal umax, vmax, umin, vmin;
	Vec u_star, v_star, dphidx, dphidy, lapPhi;

	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &u_star);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &v_star);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &dphidx);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &dphidy);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &lapPhi);

	do{
		ConstructRHS_u();
		ConstructRHS_v();

		//Calculating the intermediate velocities
		KSPSolve(uSolver, RHS_u, u_star);
		KSPSolve(vSolver, RHS_v, v_star);

		ConstructRHS_phi(&u_star, &v_star);
		MatNullSpaceRemove(NSP, RHS_phi);

		//Calculating phi
		KSPSolve(phiSolver, RHS_phi, nextField->phi);

		MatMult(dpdx, nextField->phi, dphidx);
		MatMult(dpdy, nextField->phi, dphidy);
		MatMult(lap_phi, nextField->phi, lapPhi);

		//Projection step: projecting the intermediate velocities back to the space of divergence-free vector fields to obtain the actual velocities
		VecAXPY(u_star, -dt, dphidx);
		VecAXPY(v_star, -dt, dphidy);

		VecCopy(u_star, nextField->u);
		VecCopy(v_star, nextField->v);
		VecCopy(nextField->phi, nextField->p);

		//Calculating the pressure values
		VecAXPY(nextField->p, -dt/(2*re), lapPhi);

		VecCopy(nextField->u, prevField->u);
		VecCopy(nextField->v, prevField->v);
		VecCopy(nextField->phi, prevField->phi);

		VecMinMax(&umin,&umax,grid->ugrid,grid->ny+2,grid->nx+3,&nextField->u);
		VecMinMax(&vmin,&vmax,grid->vgrid,grid->ny+3,grid->nx+2,&nextField->v);

		if((iter-1)%10 == 0)
			printf("iter\tumin\t\tumax\t\tvmin\t\tvmax\n");
		printf("%d\t%lf\t%lf\t%lf\t%lf\n", iter, umin, umax, vmin, vmax);

		if(saveIter > 0 && iter%saveIter == 0)
			ExportData(iter, nextField);
		iter++;
	}while(dt*iter <= finalTime);

	VecDestroy(&u_star);
	VecDestroy(&v_star);
	VecDestroy(&dphidx);
	VecDestroy(&dphidy);
	VecDestroy(&lapPhi);
}

void FluidSolver::ExportData(int iter, FluidField *field){
	PetscScalar *u = new PetscScalar[grid->nPoints[0]];
	PetscScalar *v = new PetscScalar[grid->nPoints[1]];
	PetscScalar *p = new PetscScalar[grid->nPoints[2]];
	VecGetArray(field->u, &u);
	VecGetArray(field->v, &v);
	VecGetArray(field->p, &p);
	char f1[255];
	char f2[255];
	char f3[255];
	sprintf(f1, "u_%d.csv", iter);
	sprintf(f2, "v_%d.csv", iter);
	sprintf(f3, "p_%d.csv", iter);
	ofstream fs{f1};
	Points pts = grid->ugrid;

	for(int i = 0; i < grid->ny+2; i++){
		for(int j = 0; j < grid->nx+3; j++){
			if(pts[i][j].type < GHOST_LEFT)
				fs<<pts[i][j].x<<","<<pts[i][j].y<<","<<u[pts[i][j].id]<<endl;
			else if(pts[i][j].type == GHOST_BOTTOM)
				fs<<0.5*(pts[i][j].x + pts[i+1][j].x)<<","<<0.5*(pts[i][j].y + pts[i+1][j].y)<<","<<0.5*(u[pts[i][j].id] + u[pts[i+1][j].id])<<endl;
			else if(pts[i][j].type == GHOST_TOP)
				fs<<0.5*(pts[i][j].x + pts[i-1][j].x)<<","<<0.5*(pts[i][j].y + pts[i-1][j].y)<<","<<0.5*(u[pts[i][j].id] + u[pts[i-1][j].id])<<endl;
		}
	}
	fs.close();

	fs.open(f2);
	pts = grid->vgrid;

	for(int i = 0; i < grid->ny+3; i++){
		for(int j = 0; j < grid->nx+2; j++){
			if(pts[i][j].type < GHOST_LEFT)
				fs<<pts[i][j].x<<","<<pts[i][j].y<<","<<v[pts[i][j].id]<<endl;
			else if(pts[i][j].type == GHOST_LEFT)
				fs<<0.5*(pts[i][j].x + pts[i][j+1].x)<<","<<0.5*(pts[i][j].y + pts[i][j+1].y)<<","<<0.5*(v[pts[i][j].id] + v[pts[i][j+1].id])<<endl;
			else if(pts[i][j].type == GHOST_RIGHT)
				fs<<0.5*(pts[i][j].x + pts[i][j-1].x)<<","<<0.5*(pts[i][j].y + pts[i][j-1].y)<<","<<0.5*(v[pts[i][j].id] + v[pts[i][j-1].id])<<endl;
		}
	}
	fs.close();

	fs.open(f3);
	pts = grid->pgrid;

	for(int i = 0; i < grid->ny+2; i++){
		for(int j = 0; j < grid->nx+2; j++){
			if(pts[i][j].type == INTERNAL)
				fs<<pts[i][j].x<<","<<pts[i][j].y<<","<<p[pts[i][j].id]<<endl;
			else if(pts[i][j].type == GHOST_LEFT)
				fs<<0.5*(pts[i][j].x + pts[i][j+1].x)<<","<<0.5*(pts[i][j].y + pts[i][j+1].y)<<","<<0.5*(p[pts[i][j].id] + p[pts[i][j+1].id])<<endl;
			else if(pts[i][j].type == GHOST_RIGHT)
				fs<<0.5*(pts[i][j].x + pts[i][j-1].x)<<","<<0.5*(pts[i][j].y + pts[i][j-1].y)<<","<<0.5*(p[pts[i][j].id] + p[pts[i][j-1].id])<<endl;
			else if(pts[i][j].type == GHOST_BOTTOM)
				fs<<0.5*(pts[i][j].x + pts[i+1][j].x)<<","<<0.5*(pts[i][j].y + pts[i+1][j].y)<<","<<0.5*(p[pts[i][j].id] + p[pts[i+1][j].id])<<endl;
			else if(pts[i][j].type == GHOST_TOP)
				fs<<0.5*(pts[i][j].x + pts[i-1][j].x)<<","<<0.5*(pts[i][j].y + pts[i-1][j].y)<<","<<0.5*(p[pts[i][j].id] + p[pts[i-1][j].id])<<endl;
		}
	}
	fs.close();
}

//Finds the minimum and maximum values of a vector(ignoring the ghost points)
void FluidSolver::VecMinMax(double* min, double* max, Points& pts, int n1, int n2, Vec* u){
	*min = INFINITY;
	*max = -INFINITY;
	int size;
	VecGetSize(*u, &size);
	PetscScalar *u0 = new PetscScalar[size];
	VecGetArray(*u, &u0);

	for(int i=0; i<n1; i++){
		for(int j=0; j<n2; j++){
			if(pts[i][j].type < GHOST_LEFT){
				if(u0[pts[i][j].id] > *max)
					*max = u0[pts[i][j].id];
				else if(u0[pts[i][j].id] < *min)
					*min = u0[pts[i][j].id];
			}
		}
	}
}

//Returns the stencil required for numerical approximation of the Laplacian at a point
int* FluidSolver::StencilLaplacian(int i, int j, const char *var){
	int *stencil = new int[5];
	Points* pts;
	if(!strcmp(var,"u"))
		pts = &grid->ugrid;
	else if(!strcmp(var,"v"))
		pts = &grid->vgrid;
	else if(!strcmp(var,"p"))
		pts = &grid->pgrid;
	stencil[0] = (*pts)[i][j+1].id;
	stencil[1] = (*pts)[i][j-1].id;
	stencil[2] = (*pts)[i][j].id;
	stencil[3] = (*pts)[i+1][j].id;
	stencil[4] = (*pts)[i-1][j].id;

	return stencil;
}

//Constructs the stencil required for approximation of the Laplacian at a pressure ghost point
int FluidSolver::StencilLaplacian_PressureGhostPoint(int i, int j, int* stencil, double* weights){
	Points& pts = grid->pgrid;
	double hx = grid->hx;
	double hy = grid->hy;
	int size = 0;

	if(pts[i][j].type == GHOST_BOTTOM || pts[i][j].type == GHOST_TOP){
		weights[0] = 2.0/pow(hy,2); weights[1] = -5.0/pow(hy,2); weights[2] = 4.0/pow(hy,2); weights[3] = -1.0/pow(hy,2);
		if(pts[i][j].type == GHOST_BOTTOM){
			stencil[0] = pts[i][j].id; stencil[1] = pts[i+1][j].id; stencil[2] = pts[i+2][j].id; stencil[3] = pts[i+3][j].id;
		}
		else{
			stencil[0] = pts[i][j].id; stencil[1] = pts[i-1][j].id; stencil[2] = pts[i-2][j].id; stencil[3] = pts[i-3][j].id;
		}

		if(pts[i][j+1].type == GHOST_CORNER || pts[i][j-1].type == GHOST_CORNER){
			size = 7;
			weights[0] += 2.0/pow(hx,2); weights[4] = -5.0/pow(hx,2); weights[5] = 4.0/pow(hx,2); weights[6] = -1.0/pow(hx,2);
			if(pts[i][j+1].type == GHOST_CORNER){
				stencil[4] = pts[i][j-1].id; stencil[5] = pts[i][j-2].id; stencil[6] = pts[i][j-3].id;
			}
			else{
				stencil[4] = pts[i][j+1].id; stencil[5] = pts[i][j+2].id; stencil[6] = pts[i][j+3].id;
			}
		}
		else{
			size = 6;
			weights[4] = 1.0/pow(hx,2); weights[0] += -2.0/pow(hx,2); weights[5] = 1.0/pow(hx,2);
			stencil[4] = pts[i][j+1].id; stencil[5] = pts[i][j-1].id;
		}
	}
	else if(pts[i][j].type == GHOST_RIGHT || pts[i][j].type == GHOST_LEFT){
		weights[0] = 2.0/pow(hx,2); weights[1] = -5.0/pow(hx,2); weights[2] = 4.0/pow(hx,2); weights[3] = -1.0/pow(hx,2);
		if(pts[i][j].type == GHOST_LEFT){
			stencil[0] = pts[i][j].id; stencil[1] = pts[i][j+1].id; stencil[2] = pts[i][j+2].id; stencil[3] = pts[i][j+3].id;
		}
		else{
			stencil[0] = pts[i][j].id; stencil[1] = pts[i][j-1].id; stencil[2] = pts[i][j-2].id; stencil[3] = pts[i][j-3].id;
		}

		if(pts[i+1][j].type == GHOST_CORNER || pts[i-1][j].type == GHOST_CORNER){
			size = 7;
			weights[0] += 2.0/pow(hy,2); weights[4] = -5.0/pow(hy,2); weights[5] = 4.0/pow(hy,2); weights[6] = -1.0/pow(hy,2);
			if(pts[i+1][j].type == GHOST_CORNER){
				stencil[4] = pts[i-1][j].id; stencil[5] = pts[i-2][j].id; stencil[6] = pts[i-3][j].id;
			}
			else{
				stencil[4] = pts[i+1][j].id; stencil[5] = pts[i+2][j].id; stencil[6] = pts[i+3][j].id;
			} 
		}
		else{
			size = 6;
			weights[4] = 1.0/pow(hy,2); weights[0] += -2.0/pow(hy,2); weights[5] = 1.0/pow(hy,2);
			stencil[4] = pts[i+1][j].id; stencil[5] = pts[i-1][j].id;
		}
	}
	return size;
}

bool FluidSolver::ApplyGoverningEquation(int i, int j, const char *var){
	Points* pts;
	if(!strcmp(var,"u"))
		pts = &grid->ugrid;
	else if(!strcmp(var,"v"))
		pts = &grid->vgrid;
	else if(!strcmp(var,"p"))
		pts = &grid->pgrid;
	if((*pts)[i][j].type == INTERNAL)
		return true;
	else if((*pts)[i][j].type >= GHOST_LEFT)
		return false;
	else if(bcType[(*pts)[i][j].type-1] == NEUMANN)
		return true;
	else return false;
}

//Calculates slope limiter for the MUSCL reconstruction of interface fluxes
double FluidSolver::SlopeLimiter(int i, int j, Points& pts, PetscScalar *var, const char *dir){
	double s = 0;
	double hx = grid->hx;
	double hy = grid->hy;
	if(!strcmp(dir, "x"))
		s = minmode((var[pts[i][j].id] - var[pts[i][j-1].id])/hx, (var[pts[i][j+1].id] - var[pts[i][j].id])/hx);
	else if(!strcmp(dir, "y"))
		s = minmode((var[pts[i][j].id] - var[pts[i-1][j].id])/hy, (var[pts[i+1][j].id] - var[pts[i][j].id])/hy);
	return s;
}

double FluidSolver::minmode(double a, double b){
	if(a*b > 0)
		return (a*min(1.0, abs(b/a)));
	else
		return 0;
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
	bool err = false;

	if(!fs){
		cout << "Simulation data file not found!\n";
		return false;
	}

	while(!fs.eof()){
		fs>>inp;
		if(fs.eof()) break;
		if(inp == "leftbc"){
			fs>>bcType[0];
			if(!fs.good()) {err = true; break;}
			if(bcType[0] > 1 || bcType[0] < 0){
				cout<<"Invalid boundary condition type!\n";
				return false;
			}
			if(bcType[0] ==  DIRICHLET) fs>>ub[0]>>vb[0];
		}
		else if(inp == "rightbc"){
			fs>>bcType[1];
			if(!fs.good()) {err = true; break;}
			if(bcType[1] > 1 || bcType[1] < 0){
				cout<<"Invalid boundary condition type!\n";
				return false;
			}
			if(bcType[1] ==  DIRICHLET) fs>>ub[1]>>vb[1];
		}
		else if(inp == "bottombc"){
			fs>>bcType[2];
			if(!fs.good()) {err = true; break;}
			if(bcType[2] > 1 || bcType[2] < 0){
				cout<<"Invalid boundary condition type!\n";
				return false;
			}
			if(bcType[2] ==  DIRICHLET) fs>>ub[2]>>vb[2];
		}
		else if(inp == "topbc"){
			fs>>bcType[3];
			if(!fs.good()) {err = true; break;}
			if(bcType[3] > 1 || bcType[3] < 0){
				cout<<"Invalid boundary condition type!\n";
				return false;
			}
			if(bcType[3] ==  DIRICHLET) fs>>ub[3]>>vb[3];
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