#ifndef __GRID_H
#define __GRID_H
#define DIRICHLET 0
#define NEUMANN 1
#define INTERNAL 0
#define BOUNDARY_LEFT 1
#define BOUNDARY_RIGHT 2
#define BOUNDARY_BOTTOM 3
#define BOUNDARY_TOP 4
#define GHOST_LEFT 5
#define GHOST_RIGHT 6
#define GHOST_BOTTOM 7
#define GHOST_TOP 8
#define GHOST_CORNER 9

struct Point{
	int id = -1;
	double x;
	double y;
	int type = INTERNAL;
};

class Grid{
private:
	double xrange[2];
	double yrange[2];
	void SetupGrid();
	int ParseDataFile(FILE *f1);
public:
	int nx;	//number of cells in the x-direction for the u-grid
	int ny;	//number of cells in the y-direction for the v-grid
	double hx, hy;	//grid spacing in x and y directions
	bool setup = false;
	Point **pgrid;
	Point **ugrid;
	Point **vgrid;
	int nPoints[3];
	Grid(char* fname);
};


#endif