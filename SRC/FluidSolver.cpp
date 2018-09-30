#include <iostream>
#include <fstream>
#include <string.h>
#include "Grid.h"
#include "FluidSolver.h"
#include "petsc.h"

using namespace std;

FluidSolver::FluidSolver(char *fname, Grid *grid){
	this->grid = grid;

	FILE *f1 = fopen(fname,"r");

	if(ParseDataFile(f1) && SolverInitialize())
		SolverSetup();
}

void FluidSolver::CreateFluidField(FluidField *fField){
	fField = new FluidField;
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[0], &(fField->u));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[1], &(fField->v));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(fField->p));
	VecCreateSeq(PETSC_COMM_SELF, grid->nPoints[2], &(fField->phi));
	VecSet(fField->u,0.0);
	VecSet(fField->v,0.0);
	VecSet(fField->p,0.0);
	VecSet(fField->phi,0.0);
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
				if(pts[i][j].type != GHOST_CORNER)
					MatSetValues(bc_phi,1,&(pts[i][j].id),2,stencil,(PetscScalar *)weights,INSERT_VALUES);
				else
					MatSetValues(bc_phi,1,&(pts[i][j].id),1,&(pts[i][j].id),(PetscScalar *)unit,INSERT_VALUES);
			}
		}
	}

	MatAXPY(LHS_u, dt, dt_u);
	MatAXPY(LHS_u, -dt/(2*re), lap_u);
	MatAXPY(LHS_u, 1.0, bc_u);
	MatAXPY(LHS_v, dt, dt_v);
	MatAXPY(LHS_v, -dt/(2*re), lap_v);
	MatAXPY(LHS_v, 1.0, bc_v);
	MatAXPY(LHS_phi, 1.0, lap_phi);
	MatAXPY(LHS_phi, 1.0, bc_phi);

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
	// PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF, PETSC_VIEWER_ASCII_DENSE);
	// MatView(bc_phi, PETSC_VIEWER_STDOUT_SELF);
	// MatView(bc_u, PETSC_VIEWER_STDOUT_SELF);
	// MatView(dpdy, PETSC_VIEWER_STDOUT_SELF);

}

void FluidSolver::ConstructRHS(){

}

void FluidSolver::ConstructRHS_u(){
	VecSet(RHS_u, 0.0);
	Vec uLap, dudt, bc, temp, convectiveDer;
	double tmp;
	int hx = grid->hx;
	int hy = grid->hy;
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
	double urr, url, ulr, ull, v_U, v_D;

	Point **pts = grid->ugrid;

	for(int i = 0; i < ny+2; i++){
		for(int j = 0; j < nx+3; j++){
			if(ApplyGoverningEquation(i, j, "u")){
				url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "x")*hx/2;
				ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "x")*hx/2;
				ull = u[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, u, "x")*hx/2;
				urr = u[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, u, "x")*hx/2;
				E_R = 0.5*(pow(urr,2) + pow(url,2) - abs(urr + url)*(urr - url));
				E_L = 0.5*(pow(ulr,2) + pow(ull,2) - abs(ulr + ull)*(ulr - ull));

				url = u[pts[i][j].id] + SlopeLimiter(i, j, pts, u, "y")*hy/2;
				ulr = u[pts[i][j].id] - SlopeLimiter(i, j, pts, u, "y")*hy/2;
				ull = u[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, u, "y")*hy/2;
				urr = u[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, u, "y")*hy/2;
				v_U = 0.5*(v[grid->vgrid[i+1][j].id] + v[grid->vgrid[i+1][j-1].id]);
				v_D = 0.5*(v[grid->vgrid[i][j].id] + v[grid->vgrid[i][j-1].id]);
				E_U = 0.5*(v_U*(urr + url) - abs(v_U)*(urr - url));
				E_D = 0.5*(v_D*(ulr + ull) - abs(v_D)*(ulr - ull));

				tmp = (E_R - E_L)/hx + (E_U - E_D)/hy;
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
}

void FluidSolver::ConstructRHS_v(){
	VecSet(RHS_v, 0.0);
	Vec vLap, dvdt, bc, temp, convectiveDer;
	double tmp;
	int hx = grid->hx;
	int hy = grid->hy;
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
	double vrr, vrl, vlr, vll, u_R, u_L;

	Point **pts = grid->vgrid;

	for(int i = 0; i < ny+3; i++){
		for(int j = 0; j < nx+2; j++){
			if(ApplyGoverningEquation(i, j, "v")){
				vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "x")*hx/2;
				vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "x")*hx/2;
				vll = v[pts[i][j-1].id] + SlopeLimiter(i, j-1, pts, v, "x")*hx/2;
				vrr = v[pts[i][j+1].id] - SlopeLimiter(i, j+1, pts, v, "x")*hx/2;
				u_R = 0.5*(u[grid->ugrid[i][j+1].id] + u[grid->ugrid[i-1][j+1].id]);
				u_L = 0.5*(u[grid->ugrid[i][j].id] + u[grid->ugrid[i-1][j].id]);
				E_R = 0.5*(u_R*(vrr + vrl) - abs(u_R)*(vrr - vrl));
				E_L = 0.5*(u_L*(vlr + vll) - abs(u_L)*(vlr - vll));

				vrl = v[pts[i][j].id] + SlopeLimiter(i, j, pts, v, "y")*hy/2;
				vlr = v[pts[i][j].id] - SlopeLimiter(i, j, pts, v, "y")*hy/2;
				vll = v[pts[i-1][j].id] + SlopeLimiter(i-1, j, pts, v, "y")*hy/2;
				vrr = v[pts[i+1][j].id] - SlopeLimiter(i+1, j, pts, v, "y")*hy/2;
				E_U = 0.5*(pow(vrr,2) + pow(vrl,2) - abs(vrr + vrl)*(vrr - vrl));
				E_D = 0.5*(pow(vlr,2) + pow(vll,2) - abs(vlr + vll)*(vlr - vll));

				tmp = (E_R - E_L)/hx + (E_U - E_D)/hy;
				VecSetValues(convectiveDer, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
			}
			else{
				if(pts[i][j].type > INTERNAL && pts[i][j].type < GHOST_LEFT)
					VecSetValues(bc, 1, &(pts[i][j].id), &ub[pts[i][j].type-1], INSERT_VALUES);
				else if(pts[i][j].type == GHOST_BOTTOM && bcType[2] == NEUMANN){
					tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i+2][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_TOP && bcType[3] == NEUMANN){
					tmp = dt*(dphidy[pts[i][j].id] - dphidy[pts[i-2][j].id]);
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_LEFT && bcType[0] == DIRICHLET){
					tmp = 2*(ub[0] + 0.5*dt*(dphidy[pts[i][j].id] + dphidy[pts[i][j+1].id]));
					VecSetValues(bc, 1, &(pts[i][j].id), &tmp, INSERT_VALUES);
				}
				else if(pts[i][j].type == GHOST_RIGHT && bcType[1] == DIRICHLET){
					tmp = 2*(ub[1] + 0.5*dt*(dphidx[pts[i][j].id] + dphidx[pts[i][j-1].id]));
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
	else{
		CreateFluidField(prevField);
		CreateFluidField(nextField);
		return 1;
	}
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
	int hx = grid->hx;
	int hy = grid->hy;
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
					if(fscanf(f1, "%d", &re) < 1) err++;
				}
			}
		}
		if(err > 0) cout << "Invalid data file format!\n";
	}
	if(err > 0) return 0;
	else return 1;
}