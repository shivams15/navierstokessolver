#include <iostream>
#include <fstream>
#include <string.h>
#include "Grid.h"

 using namespace std;

 Grid::Grid(char* fname){
	FILE *f1 = fopen(fname,"r");
	if(ParseDataFile(f1)) SetupGrid();
 }

 void Grid::SetupGrid(){
 	if(xrange[0] >= xrange[1] || yrange[0] >= yrange[1]){
 		cout << "Invalid domain range!\n";
 		return;
 	}

 	hx = (xrange[1] - xrange[0])/nx;
 	hy = (yrange[1] - yrange[0])/ny;
 	pgrid = new Point*[ny+2];
 	ugrid = new Point*[ny+2];
 	vgrid = new Point*[ny+3];

 	for(int i = 0; i < ny + 2; i++){
 		pgrid[i] = new Point[nx+2];
 		ugrid[i] = new Point[nx+3];
 		vgrid[i] = new Point[nx+2];
 	}
 	vgrid[ny+2] = new Point[nx+2];

 	for(int i = 0; i < ny + 2; i++){
 		for(int j = 0; j < nx + 2; j++){
 			pgrid[i][j].x = xrange[0] -hx/2 + j*hx;
 			pgrid[i][j].y = yrange[0] - hy/2 + i*hy;
 			if (i%(ny+1) + j%(nx+1) == 0) 
 				pgrid[i][j].type = GHOST_CORNER;
 			else{
 				if(i == 0) pgrid[i][j].type = GHOST_BOTTOM;
 				else if(i == ny+1) pgrid[i][j].type = GHOST_TOP;
 				else if(j == 0) pgrid[i][j].type = GHOST_LEFT;
 				else if(j == nx+1) pgrid[i][j].type = GHOST_RIGHT;
 			}
 		}
 	}

 	for(int i = 0; i < ny + 2; i++){
 		for(int j = 0; j < nx + 3; j++){
 			ugrid[i][j].x = xrange[0] + (j-1)*hx;
 			ugrid[i][j].y = yrange[0] - hy/2 + i*hy;
 			if (i%(ny+1) + j%(nx+2) == 0) 
 				ugrid[i][j].type = GHOST_CORNER;
 			else{
 				if(i == 0) ugrid[i][j].type = GHOST_BOTTOM;
 				else if(i == ny+1) ugrid[i][j].type = GHOST_TOP;
 				else if(j == 0) ugrid[i][j].type = GHOST_LEFT;
 				else if(j == nx+2) ugrid[i][j].type = GHOST_RIGHT;
 				else if(j == 1) ugrid[i][j].type = BOUNDARY_LEFT;
 				else if(j == nx+1) ugrid[i][j].type = BOUNDARY_RIGHT;
 			}
 		}
 	}

 	for(int i = 0; i < ny + 3; i++){
 		for(int j = 0; j < nx + 2; j++){
 			vgrid[i][j].x = xrange[0] -hx/2 + j*hx;
 			vgrid[i][j].y = yrange[0] + (i-1)*hy;
 			if (i%(ny+2) + j%(nx+1) == 0) 
 				vgrid[i][j].type = GHOST_CORNER;
 			else{
 				if(i == 0) vgrid[i][j].type = GHOST_BOTTOM;
 				else if(i == ny+2) vgrid[i][j].type = GHOST_TOP;
 				else if(j == 0) vgrid[i][j].type = GHOST_LEFT;
 				else if(j == nx+1) vgrid[i][j].type = GHOST_RIGHT;
 				else if(i == 1) vgrid[i][j].type = BOUNDARY_BOTTOM;
 				else if(i == ny+1) vgrid[i][j].type = BOUNDARY_TOP;
 			}
 		}
 	}
 	setup = true;
 }

int Grid::ParseDataFile(FILE *f1){
	char inp[255];
	int err = 0;

	if(!f1){
		cout << "Grid data file not found!\n";
		err++;
	}
	else{
		while(!feof(f1)){
			if(fscanf(f1, "%s", inp) > 0){
				if(!strcmp(inp, "domain")){
					if(fscanf(f1, "%lf %lf", &xrange[0], &xrange[1]) < 2) err++;
					if(fscanf(f1, "%lf %lf", &yrange[0], &yrange[1]) < 2) err++;
				}
				else if(!strcmp(inp, "n")){
					if(fscanf(f1, "%d %d", &nx, &ny) < 2) err++;
				}
				else err++;
			}
		}
		if(err > 0) cout << "Invalid grid data file format!\n";
	}
	if(err > 0)	return 0;
	else return 1;
}

