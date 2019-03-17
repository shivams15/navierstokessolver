#include <iostream>
#include <string>
#include "Grid.h"

 using namespace std;

 Grid::Grid(char* fname){
	ifstream fs{fname};
	if(ParseDataFile(fs)) SetupGrid();
	fs.close();
 }

/*
Generates the grid points
A staggered grid is used for u and v
*/
 void Grid::SetupGrid(){
 	if(xrange[0] >= xrange[1] || yrange[0] >= yrange[1]){
 		cout << "Invalid domain range!\n";
 		return;
 	}

 	if(nx <= 0 || ny <= 0){
 		cout<<"Number of cells must be set to a positive number!\n";
 		return;
 	}

 	hx = (xrange[1] - xrange[0])/nx;
 	hy = (yrange[1] - yrange[0])/ny;
 	
 	pgrid = Points(ny+2, vector<Point>(nx+2));
 	ugrid = Points(ny+2, vector<Point>(nx+3));
 	vgrid = Points(ny+3, vector<Point>(nx+2));

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

//Reads user-provided parameters from the grid data file
bool Grid::ParseDataFile(ifstream& fs){
	string inp{};
	bool err = false;

	if(!fs){
		cout << "Grid data file not found!\n";
		return false;
	}

	while(!fs.eof()){
		fs>>inp;
		if(fs.eof()) break;
		if(inp == "domain") fs>>xrange[0]>>xrange[1]>>yrange[0]>>yrange[1];
		else if(inp == "n") fs>>nx>>ny;
		else {err = true; break;}

		if(!fs.good()) {err = true; break;}
	}

	if(err){
		cout<<"Invalid data file format!\n";
		return false;
	}
	return true;
}

