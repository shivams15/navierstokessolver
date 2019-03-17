#ifndef __GRID_H
#define __GRID_H

#include <fstream>
#include <vector>

using namespace std; 

enum bcTypes: int8_t{DIRICHLET, NEUMANN};
enum pointTypes: int8_t{
	INTERNAL, BOUNDARY_LEFT, BOUNDARY_RIGHT, BOUNDARY_BOTTOM, BOUNDARY_TOP,
	GHOST_LEFT, GHOST_RIGHT, GHOST_BOTTOM, GHOST_TOP, GHOST_CORNER
};

struct Point{
	int id = -1;
	double x;
	double y;
	int type = INTERNAL;
};

typedef vector<vector<Point>> Points;

class Grid{
private:
	double xrange[2];
	double yrange[2];
	void SetupGrid();
	bool ParseDataFile(ifstream& fs);
public:
	int nx = -1;	//number of cells in the x-direction for the u-grid
	int ny = -1;	//number of cells in the y-direction for the v-grid
	double hx, hy;	//grid spacing in x and y directions
	bool setup = false;
	Points pgrid;
	Points ugrid;
	Points vgrid;
	int nPoints[3];
	Grid(char* fname);
};


#endif