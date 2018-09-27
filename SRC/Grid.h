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

struct Point{
	int id = -1;
	int x;
	int y;
	int type = INTERNAL;
};

class Grid{
private:
	double xrange[2];
	double yrange[2];
	void SetupGrid();
public:
	int nx, ny;
	double hx, hy;
	bool setup = false;
	Point **pgrid;
	Point **ugrid;
	Point **vgrid;
	Grid(char* fname);
};


#endif