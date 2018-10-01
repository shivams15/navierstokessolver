#include <iostream>
#include <fstream>
#include <string.h>
#include "Grid.h"
#include "FluidSolver.h"
#include "petsc.h"
#include "petscksp.h"

using namespace std;

FluidSolver::FluidSolver(char *fname, Grid *grid){
	this->grid = grid;

	FILE *f1 = fopen(fname,"r");

	if(ParseDataFile(f1) && SolverInitialize())
		SolverSetup();
}

void FluidSolver::CreateFluidField(FluidField **fField){
	*fField = new FluidField[1];
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &((*fField)->u));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &((*fField)->v));
	// VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(fField->p));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(*fField)->phi);
	VecSet((*fField)->u,0.0);
	VecSet((*fField)->v,0.0);
	// VecSet(fField->p,0.0);
	VecSet((*fField)->phi,0.0);
}

void FluidSolver::SolverSetup(){
	CreateMatrix(&LHS_u, grid->nPoints[0], grid->nPoints[0]);
	CreateMatrix(&lap_u, grid->nPoints[0], grid->nPoints[0]);
	CreateMatrix(&dt_u, grid->nPoints[0], grid->nPoints[0]);
	CreateMatrix(&bc_u, grid->nPoints[0], grid->nPoints[0]);
	CreateMatrix(&LHS_v, grid->nPoints[1], grid->nPoints[1]);
	CreateMatrix(&lap_v, grid->nPoints[1], grid->nPoints[1]);
	CreateMatrix(&dt_v, grid->nPoints[1], grid->nPoints[1]);
	CreateMatrix(&bc_v, grid->nPoints[1], grid->nPoints[1]);
	CreateMatrix(&LHS_phi, grid->nPoints[2], grid->nPoints[2]);
	CreateMatrix(&lap_phi, grid->nPoints[2], grid->nPoints[2]);
	CreateMatrix(&bc_phi, grid->nPoints[2], grid->nPoints[2]);
	CreateMatrix(&dudx, grid->nPoints[2], grid->nPoints[0]);
	CreateMatrix(&dvdy, grid->nPoints[2], grid->nPoints[1]);
	CreateMatrix(&dpdx, grid->nPoints[0], grid->nPoints[2]);
	CreateMatrix(&dpdy, grid->nPoints[1], grid->nPoints[2]);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &RHS_u);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &RHS_v);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &RHS_phi);


	int nx = grid->nx;
	int ny = grid->ny;
	double hx = grid->hx;
	double hy = grid->hy;

	double lapWeights[5] = {1.0/pow(hx,2), 1.0/pow(hx,2), -2.0/pow(hx,2) -2.0/pow(hy,2), 1.0/pow(hy,2), 1.0/pow(hy,2)}; 
	double dtWeights[1] = {1.0/dt};
	double unit[1] = {1.0};
	double weights[3] = {0.0, 0.0, 0.0};
	int stencil[3] = {0, 0, 0};

	Point **pts = grid->ugrid;
	for(int i = 0; i < ny + 2; i++){
		for(int j = 0; j < nx + 3; j++){
			if(ApplyGoverningEquation(i, j, "u")){
				MatSetValues(lap_u,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"u"),(PetscScalar *)lapWeights,INSERT_VALUES);
				MatSetValues(dt_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)dtWeights,INSERT_VALUES);
				weights[0] = 1.0/hx; weights[1] = -1.0/hx;
				stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
				MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT){
					MatSetValues(bc_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					weights[0] = 1.0/hx; weights[1] = -1.0/hx;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
					MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_LEFT){
					if(bcType[0] == DIRICHLET)
						MatSetValues(bc_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					else{
						weights[0] = 1.0; weights[1] = -1.0;
						stencil[0] = pts[i][j].id; stencil[1] = pts[i][j+2].id; 
						MatSetValues(bc_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					weights[0] = -2.0/hx; weights[1] = 3.0/hx; weights[2] = -1.0/hx;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j+1].id; stencil[2] = grid->pgrid[i][j+2].id;
					MatSetValues(dpdx,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT){
					if(bcType[1] == DIRICHLET)
						MatSetValues(bc_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					else{
						weights[0] = 1.0; weights[1] = -1.0;
						stencil[0] = pts[i][j].id; stencil[1] = pts[i][j-2].id; 
						MatSetValues(bc_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
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
					MatSetValues(bc_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(j != 1 && j!= nx+1){
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
					MatSetValues(bc_u,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(j != 1 && j!= nx+1){
						weights[0] = 1.0/hx; weights[1] = -1.0/hx;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i][j-1].id;
						MatSetValues(dpdx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
				else 
					MatSetValues(bc_u,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
			}
		}
	}

	pts = grid->vgrid;
	for(int i = 0; i < ny + 3; i++){
		for(int j = 0; j < nx + 2; j++){
			if(ApplyGoverningEquation(i, j, "v")){
				MatSetValues(lap_v,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"v"),(PetscScalar *)lapWeights,INSERT_VALUES);
				MatSetValues(dt_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)dtWeights,INSERT_VALUES);
				weights[0] = 1.0/hy; weights[1] = -1.0/hy;
				stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
				MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT){
					MatSetValues(bc_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					weights[0] = 1.0/hy; weights[1] = -1.0/hy;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
					MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_BOTTOM){
					if(bcType[2] == DIRICHLET)
						MatSetValues(bc_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					else{
						weights[0] = 1.0; weights[1] = -1.0;
						stencil[0] = pts[i][j].id; stencil[1] = pts[i+2][j].id; 
						MatSetValues(bc_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
					weights[0] = -2.0/hy; weights[1] = 3.0/hy; weights[2] = -1.0/hy;
					stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i+1][j].id; stencil[2] = grid->pgrid[i+2][j].id;
					MatSetValues(dpdy,1,&(pts[i][j].id),3,stencil,(PetscScalar *)weights,INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP){
					if(bcType[3] == DIRICHLET)
						MatSetValues(bc_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					else{
						weights[0] = 1.0; weights[1] = -1.0;
						stencil[0] = pts[i][j].id; stencil[1] = pts[i-2][j].id; 
						MatSetValues(bc_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
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
					MatSetValues(bc_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(i != 1 && i!= ny+1){
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
					MatSetValues(bc_v,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					if(i != 1 && i!= ny+1){
						weights[0] = 1.0/hy; weights[1] = -1.0/hy;
						stencil[0] = grid->pgrid[i][j].id; stencil[1] = grid->pgrid[i-1][j].id;
						MatSetValues(dpdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					}
				}
				else
					MatSetValues(bc_v,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
			}
		}
	}

	Vec nsp;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &nsp);

	pts = grid->pgrid;
	for(int i = 0; i < ny + 2; i++){
		for(int j = 0; j < nx + 2; j++){
			if(ApplyGoverningEquation(i, j, "p")){
				MatSetValues(lap_phi,1,&(pts[i][j].id),5,StencilLaplacian(i,j,"p"),(PetscScalar *)lapWeights,INSERT_VALUES);
				weights[0] = 1.0/hx; weights[1] = -1.0/hx;
				stencil[0] = grid->ugrid[i][j+1].id; stencil[1] = grid->ugrid[i][j].id;
				MatSetValues(dudx,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				weights[0] = 1.0/hy; weights[1] = -1.0/hy;
				stencil[0] = grid->vgrid[i+1][j].id; stencil[1] = grid->vgrid[i][j].id;
				MatSetValues(dvdy,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				weights[0] = 1.0;
				VecSetValues(nsp, 1, &(pts[i][j].id), weights, INSERT_VALUES);
			}
			else{
				weights[0] = 1.0; weights[1] = -1.0;
				stencil[0] = pts[i][j].id;
				if(pts[i][j].type == GHOST_BOTTOM)
					stencil[1] = pts[i+1][j].id; 
				else if(pts[i][j].type == GHOST_TOP)
					stencil[1] = pts[i-1][j].id; 
				else if(pts[i][j].type == GHOST_LEFT)
					stencil[1] = pts[i][j+1].id; 
				else if(pts[i][j].type == GHOST_RIGHT)
					stencil[1] = pts[i][j-1].id; 
				if(pts[i][j].type != GHOST_CORNER){
					MatSetValues(bc_phi,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
					VecSetValues(nsp, 1, &(pts[i][j].id), weights, INSERT_VALUES);
				}
				else{
					MatSetValues(bc_phi,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
					weights[0] = 0.0;
					VecSetValues(nsp, 1, &(pts[i][j].id), weights, INSERT_VALUES);
				}
			}
		}
	}
	double norm;
	VecNorm(nsp, NORM_2, &norm);
	VecScale(nsp, 1.0/norm);
	MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 1, &nsp, &NSP);

	MatAssemblyBegin(LHS_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dt_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dt_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(bc_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(bc_u, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(LHS_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dt_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dt_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(bc_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(bc_v, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(LHS_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(LHS_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(lap_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(lap_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(bc_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(bc_phi, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dudx, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dudx, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dvdy, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dvdy, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dpdx, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dpdx, MAT_FINAL_ASSEMBLY);
	MatAssemblyBegin(dpdy, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(dpdy, MAT_FINAL_ASSEMBLY);

	MatAXPY(LHS_u, dt, dt_u, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_u, -dt/(2*re), lap_u, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_u, 1.0, bc_u, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_v, dt, dt_v, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_v, -dt/(2*re), lap_v, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_v, 1.0, bc_v, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_phi, 1.0, lap_phi, DIFFERENT_NONZERO_PATTERN);
	MatAXPY(LHS_phi, 1.0, bc_phi, DIFFERENT_NONZERO_PATTERN);

	// PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_ASCII_DENSE);
	// MatView(bc_phi, PETSC_VIEWER_STDOUT_SELF);
	// MatView(bc_u, PETSC_VIEWER_STDOUT_SELF);
	// MatView(dpdy, PETSC_VIEWER_STDOUT_SELF);
	MatSetNullSpace(LHS_phi, NSP);
	MatSetTransposeNullSpace(LHS_phi, NSP);
	ConfigureKSPSolver(&uSolver, &LHS_u);
	ConfigureKSPSolver(&vSolver, &LHS_v);
	ConfigureKSPSolver(&phiSolver, &LHS_phi);

	setup =true;
}

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
	VecAXPY(RHS_u, dt, dudt);
	VecAXPY(RHS_u, dt/(2*re), uLap);
	double urr, url, ulr, ull, v_U, v_D, E_R, E_L, E_U, E_D;

	Point **pts = grid->ugrid;

	for(int i = 0; i < grid->ny+2; i++){
		for(int j = 0; j < grid->nx+3; j++){
			if(ApplyGoverningEquation(i, j, "u")){
				url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "x")*hx/2;
				ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "x")*hx/2;
				if(ApplyGoverningEquation(i, j-1, "u"))
					ull = u[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, u, "x")*hx/2;
				else ull = 0.5*(u[pts[i][j-1].id] + u[pts[i][j].id]);
				if(ApplyGoverningEquation(i, j+1, "u"))
					urr = u[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, u, "x")*hx/2;
				else urr = 0.5*(u[pts[i][j+1].id] + u[pts[i][j].id]);
				E_R = 0.5*(pow(urr,2) + pow(url,2) - abs(urr + url)*(urr - url));
				E_L = 0.5*(pow(ulr,2) + pow(ull,2) - abs(ulr + ull)*(ulr - ull));

				url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "y")*hy/2;
				ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "y")*hy/2;
				if(ApplyGoverningEquation(i-1, j, "u"))
					ull = u[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, u, "y")*hy/2;
				else ull = 0.5*(u[pts[i-1][j].id] + u[pts[i][j].id]);
				if(ApplyGoverningEquation(i+1, j, "u"))
					urr = u[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, u, "y")*hy/2;
				else urr = 0.5*(u[pts[i+1][j].id] + u[pts[i][j].id]);
				v_U = 0.5*(v[grid->vgrid[i+1][j].id] + v[grid->vgrid[i+1][j-1].id]);
				v_D = 0.5*(v[grid->vgrid[i][j].id] + v[grid->vgrid[i][j-1].id]);
				E_U = 0.5*(v_U*(urr + url) - abs(v_U)*(urr - url));
				E_D = 0.5*(v_D*(ulr + ull) - abs(v_D)*(ulr - ull));
				// cout << E_R <<" " << E_L<<  " " << E_U << " "<< E_D<<"\n";
				tmp = (E_R - E_L)/hx + (E_U - E_D)/hy;
				// cout << hx << " " << hy << "\n";
				// cout << tmp << "\n";
				VecSetValues(convectiveDer, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT)
					VecSetValues(bc, 1, &(pts[i][j].id), &ub[pts[i][j].type-1], INSERT_VALUES);
				else if(pts[i][j].type == GHOST_LEFT && bcType[0] == NEUMANN){
					tmp = dt*(dphidx[pts[i][j].id] - dphidx[pts[i][j+2].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT && bcType[1] == NEUMANN){
					tmp = dt*(dphidx[pts[i][j].id] - dphidx[pts[i][j-2].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_BOTTOM && bcType[2] == DIRICHLET){
					tmp = 2*(ub[2] + 0.5*dt*(dphidx[pts[i][j].id] + dphidx[pts[i+1][j].id]));
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP && bcType[3] == DIRICHLET){
					tmp = 2*(ub[3] + 0.5*dt*(dphidx[pts[i][j].id] + dphidx[pts[i-1][j].id]));
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
			}
		}
	}

	VecAXPY(RHS_u, -dt, convectiveDer);
	VecAXPY(RHS_u, 1.0, bc);
	// VecView(RHS_u, PETSC_VIEWER_STDOUT_SELF);

}

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
	VecAXPY(RHS_v, dt, dvdt);
	VecAXPY(RHS_v, dt/(2*re), vLap);
	double vrr, vrl, vlr, vll, u_R, u_L, E_R, E_L, E_U, E_D;

	Point **pts = grid->vgrid;

	for(int i = 0; i < grid->ny+3; i++){
		for(int j = 0; j < grid->nx+2; j++){
			if(ApplyGoverningEquation(i, j, "v")){
				vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "x")*hx/2;
				vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "x")*hx/2;
				if(ApplyGoverningEquation(i, j-1, "v")) 
					vll = v[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, v, "x")*hx/2;
				else vll = 0.5*(v[pts[i][j-1].id] + v[pts[i][j].id]);
				if(ApplyGoverningEquation(i, j+1, "v"))
					vrr = v[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, v, "x")*hx/2;
				else vrr = 0.5*(v[pts[i][j+1].id] + v[pts[i][j].id]);
				u_R = 0.5*(u[grid->ugrid[i][j+1].id] + u[grid->ugrid[i-1][j+1].id]);
				u_L = 0.5*(u[grid->ugrid[i][j].id] + u[grid->ugrid[i-1][j].id]);
				E_R = 0.5*(u_R*(vrr + vrl) - abs(u_R)*(vrr - vrl));
				E_L = 0.5*(u_L*(vlr + vll) - abs(u_L)*(vlr - vll));

				vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "y")*hy/2;
				vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "y")*hy/2;
				if(ApplyGoverningEquation(i-1, j, "v"))
					vll = v[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, v, "y")*hy/2;
				else vll = 0.5*(v[pts[i-1][j].id] + v[pts[i][j].id]);
				if(ApplyGoverningEquation(i+1, j, "v"))
					vrr = v[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, v, "y")*hy/2;
				else vrr = 0.5*(v[pts[i+1][j].id] + v[pts[i][j].id]);
				E_U = 0.5*(pow(vrr,2) + pow(vrl,2) - abs(vrr + vrl)*(vrr - vrl));
				E_D = 0.5*(pow(vlr,2) + pow(vll,2) - abs(vlr + vll)*(vlr - vll));

				tmp = (E_R - E_L)/hx + (E_U - E_D)/hy;
				VecSetValues(convectiveDer, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT)
					VecSetValues(bc, 1, &(pts[i][j].id), &vb[pts[i][j].type-1], INSERT_VALUES);
				else if(pts[i][j].type == GHOST_BOTTOM && bcType[2] == NEUMANN){
					tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i+2][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP && bcType[3] == NEUMANN){
					tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i-2][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_LEFT && bcType[0] == DIRICHLET){
					tmp = 2*(vb[0] + 0.5*dt*(dphidy[pts[i][j].id] + dphidy[pts[i][j+1].id]));
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT && bcType[1] == DIRICHLET){
					tmp = 2*(vb[1] + 0.5*dt*(dphidy[pts[i][j].id] + dphidy[pts[i][j-1].id]));
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
			}
		}
	}
	VecAXPY(RHS_v, -dt, convectiveDer);
	VecAXPY(RHS_v, 1.0, bc);
}

void FluidSolver::ConstructRHS_phi(Vec *u_star, Vec *v_star){
	VecSet(RHS_phi, 0.0);
	Vec dUdX, dVdY;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &dUdX);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &dVdY);
	MatMult(dudx, *u_star, dUdX);
	MatMult(dvdy, *v_star, dVdY);
	VecAXPY(RHS_phi, 1.0/dt, dUdX);
	VecAXPY(RHS_phi, 1.0/dt, dVdY);
}

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
		CreateFluidField(&prevField);
		// VecView(prevField->u, PETSC_VIEWER_STDOUT_SELF);
		CreateFluidField(&nextField);
		return 1;
	}
}

void FluidSolver::Solve(){
	int iter = 1;
	PetscReal umax, vmax, divmax;
	Vec u_star, v_star, dphidx, dphidy, divx, divy, div;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &u_star);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &v_star);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &dphidx);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &dphidy);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &divx);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &divy);
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &div);

	do{
		ConstructRHS_u();
		ConstructRHS_v();
		KSPSolve(uSolver, RHS_u, u_star);
		KSPSolve(vSolver, RHS_v, v_star);

		// VecView(u_star, PETSC_VIEWER_STDOUT_SELF);

		ConstructRHS_phi(&u_star, &v_star);
		// VecView(v_star, PETSC_VIEWER_STDOUT_SELF);
		// VecView(RHS_phi, PETSC_VIEWER_STDOUT_SELF);

		// VecView(nextField->phi, PETSC_VIEWER_STDOUT_SELF);

		KSPSolve(phiSolver, RHS_phi, nextField->phi);

		MatMult(dpdx, nextField->phi, dphidx);
		MatMult(dpdy, nextField->phi, dphidy);
		VecAXPY(u_star, -dt, dphidx);
		VecAXPY(v_star, -dt, dphidy);

		// VecView(u_star, PETSC_VIEWER_STDOUT_SELF);


		VecCopy(u_star, nextField->u);
		VecCopy(v_star, nextField->v);

		MatMult(dudx, nextField->u, divx);
		MatMult(dvdy, nextField->v, divy);
		VecSet(div, 0.0);
		VecAXPY(div, 1.0, divx);
		VecAXPY(div, 1.0, divy);

		VecCopy(nextField->u, prevField->u);
		VecCopy(nextField->v, prevField->v);
		VecCopy(nextField->phi, prevField->phi);

		VecMax(nextField->u, NULL, &umax);
		VecMax(nextField->v, NULL, &vmax);
		VecMax(div, NULL, &divmax);

		printf("%d %lf %lf %lf\n", iter, umax, vmax, divmax);

		if(saveIter > 0 && iter%saveIter == 0)
			ExportData(iter, nextField);
		iter++;
	}while(dt*iter <= finalTime);

	// VecView(nextField->phi, PETSC_VIEWER_STDOUT_SELF);
}

void FluidSolver::ExportData(int iter, FluidField *field){
	PetscScalar *u = new PetscScalar[grid->nPoints[0]];
	PetscScalar *v = new PetscScalar[grid->nPoints[1]];
	VecGetArray(field->u, &u);
	VecGetArray(field->v, &v);
	char f1[255];
	char f2[255];
	sprintf(f1, "u_%d.txt", iter);
	sprintf(f2, "v_%d.txt", iter);
	FILE *f = fopen(f1, "w");
	Point **pts = grid->ugrid;

	for(int i = 0; i < grid->ny+2; i++){
		for(int j = 0; j < grid->nx+3; j++){
			if(pts[i][j].type < GHOST_LEFT)
				fprintf(f, "%lf %lf %lf\n", pts[i][j].x, pts[i][j].y, u[pts[i][j].id]);
			else if(pts[i][j].type == GHOST_BOTTOM)
				fprintf(f, "%lf %lf %lf\n", 0.5*(pts[i][j].x + pts[i+1][j].x), 0.5*(pts[i][j].y + pts[i+1][j].y), 0.5*(u[pts[i][j].id] + u[pts[i+1][j].id]));
			else if(pts[i][j].type == GHOST_TOP)
				fprintf(f, "%lf %lf %lf\n", 0.5*(pts[i][j].x + pts[i-1][j].x), 0.5*(pts[i][j].y + pts[i-1][j].y), 0.5*(u[pts[i][j].id] + u[pts[i-1][j].id]));
		}
	}
	fclose(f);

	f = fopen(f2, "w");
	pts = grid->vgrid;

	for(int i = 0; i < grid->ny+3; i++){
		for(int j = 0; j < grid->nx+2; j++){
			if(pts[i][j].type < GHOST_LEFT)
				fprintf(f, "%lf %lf %lf\n", pts[i][j].x, pts[i][j].y, v[pts[i][j].id]);
			else if(pts[i][j].type == GHOST_LEFT)
				fprintf(f, "%lf %lf %lf\n", 0.5*(pts[i][j].x + pts[i][j+1].x), 0.5*(pts[i][j].y + pts[i][j+1].y), 0.5*(v[pts[i][j].id] + v[pts[i][j+1].id]));
			else if(pts[i][j].type == GHOST_RIGHT)
				fprintf(f, "%lf %lf %lf\n", 0.5*(pts[i][j].x + pts[i][j-1].x), 0.5*(pts[i][j].y + pts[i][j-1].y), 0.5*(v[pts[i][j].id] + v[pts[i][j-1].id]));
		}
	}
	fclose(f);
}

int* FluidSolver::StencilLaplacian(int i, int j, char *var){
	int *stencil = new int[5];
	Point **pts;
	if(!strcmp(var,"u"))
		pts = grid->ugrid;
	else if(!strcmp(var,"v"))
		pts = grid->vgrid;
	else if(!strcmp(var,"p"))
		pts = grid->pgrid;
	stencil[0] = pts[i][j+1].id;
	stencil[1] = pts[i][j-1].id;
	stencil[2] = pts[i][j].id;
	stencil[3] = pts[i+1][j].id;
	stencil[4] = pts[i-1][j].id;

	return stencil;
}

bool FluidSolver::ApplyGoverningEquation(int i, int j, char *var){
	Point **pts;
	if(!strcmp(var,"u"))
		pts = grid->ugrid;
	else if(!strcmp(var,"v"))
		pts = grid->vgrid;
	else if(!strcmp(var,"p"))
		pts = grid->pgrid;
	if(pts[i][j].type == INTERNAL)
		return true;
	else if(pts[i][j].type >= GHOST_LEFT)
		return false;
	else if(bcType[pts[i][j].type-1] == NEUMANN)
		return true;
	else return false;
}

double FluidSolver::SlopeLimiter(int i, int j, Point **pts, PetscScalar *var, char *dir){
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

int FluidSolver::ParseDataFile(FILE *f1){
	char inp[255];
	int err = 0;

	if(!f1){
		cout << "Simulation data file not found!\n";
		err++;
	}
	else{
		while(!feof(f1)){
			if(fscanf(f1, "%s", inp) > 0){
				if(!strcmp(inp, "leftbc")){
					if(fscanf(f1, "%d", &bcType[0]) < 1) err++;
					else if(bcType[0] > 1){
						cout << "Invalid boundary condition type!\n";
						err++;
					}
					else if(bcType[0] == DIRICHLET){
						if(fscanf(f1, "%s", inp) > 0 && !strcmp(inp, "value")){
							if(fscanf(f1, "%lf %lf", &ub[0], &vb[0]) < 2) err++;
						}
						else err++;
					}
				}
				else if(!strcmp(inp, "rightbc")){
					if(fscanf(f1, "%d", &bcType[1]) < 1) err++;
					else if(bcType[1] > 1){
						cout << "Invalid boundary condition type!\n";
						err++;
					}
					else if(bcType[1] == DIRICHLET){
						if(fscanf(f1, "%s", inp) > 0 && !strcmp(inp, "value")){
							if(fscanf(f1, "%lf %lf", &ub[1], &vb[1]) < 2) err++;
						}
						else err++;
					}
				}
				else if(!strcmp(inp, "bottombc")){
					if(fscanf(f1, "%d", &bcType[2]) < 1) err++;
					else if(bcType[2] > 1){
						cout << "Invalid boundary condition type!\n";
						err++;
					}
					else if(bcType[2] == DIRICHLET){
						if(fscanf(f1, "%s", inp) > 0 && !strcmp(inp, "value")){
							if(fscanf(f1, "%lf %lf", &ub[2], &vb[2]) < 2) err++;
						}
						else err++;
					}
				}
				else if(!strcmp(inp, "topbc")){
					if(fscanf(f1, "%d", &bcType[3]) < 1) err++;
					else if(bcType[3] > 1){
						cout << "Invalid boundary condition type!\n";
						err++;
					}
					else if(bcType[3] == DIRICHLET){
						if(fscanf(f1, "%s", inp) > 0 && !strcmp(inp, "value")){
							if(fscanf(f1, "%lf %lf", &ub[3], &vb[3]) < 2) err++;
						}
						else err++;
					}
				}
				else if(!strcmp(inp, "dt")){
					if(fscanf(f1, "%lf", &dt) < 1) err++;
				}
				else if(!strcmp(inp, "final_time")){
					if(fscanf(f1, "%lf", &finalTime) < 1) err++;
				}
				else if(!strcmp(inp, "re")){
					if(fscanf(f1, "%lf", &re) < 1) err++;
				}
				else if(!strcmp(inp, "saveIter"))
					fscanf(f1, "%d", &saveIter);
			}
		}
		if(err > 0) cout << "Invalid data file format!\n";
	}
	if(err > 0) return 0;
	else return 1;
}