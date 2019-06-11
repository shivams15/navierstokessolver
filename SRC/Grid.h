#ifndef __GRID_H
#define __GRID_H

#include <fstream>
#include <vector>
#include <set>

using namespace std; 

enum bcTypes: int8_t{DIRICHLET, NEUMANN};

struct Edge{
	int nx,ny; //Information on orientation of the boundaries
	vector<double> loc {0,0,0}; //Information on the boundary location
	int bcType;	//Type of boundary condition
	vector<double> bcInfo;	//Information required for implementing the boundary condition
};

typedef vector<Edge> EdgeList;

struct Cell{
	int id {-1};	//Number used to identify the cell
	double x,y;	//Coordinates of the cell center
	vector<double> X {0,0}; //Coordinates of the face centers
	vector<double> Y {0,0};
	set<int> edges;	//List of boundaries adjacent to the cell
};

typedef vector<vector<Cell>> CellList;

class Grid{
private:
	const double TOL = 1E-8;
	vector<double> xrange {1E15,-1E15};	//Extent of the computational domain
	vector<double> yrange {1E15,-1E15};
	vector<double> AR {1E15,-1E15};	//Minimum and maximum aspect ratios
	vector<vector<double>> vertices; //Vertices of the computational domain
	void SetupGrid();
	bool GenerateEdges();
	bool GenerateFaces();
	void GenerateCells();
	void Cleanup();
	bool Interior(Cell&);
	void Show();
	void ShowEdges();
	void ExportData();
	bool ParseDataFile(ifstream& fs);
	bool equals(double, double);
public:
	int N; //Total number of cells
	vector<vector<double>> Nx;	//Number of cells in the x-direction
	vector<vector<double>> Ny;	//Number of cells in the y-direction
	vector<double> X,Y;	//Location of the cell faces
	vector<double> hx, hy;	//Grid spacing in x and y directions
	CellList cells;	//List of cells in the grid
	EdgeList edges;	//List of boundaries
	bool setup = false;
	Grid(char* fname);
};


#endif