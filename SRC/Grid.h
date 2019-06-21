#ifndef __GRID_H
#define __GRID_H

#include <fstream>
#include <vector>
#include <set>

using namespace std; 

enum bcTypes: int8_t{INLET_UNI, INLET_PARABOLIC, WALL, PRESSURE, NEUMANN};
const double TOL = 1E-8;

struct Stencil
{
	vector<double> weights;
	vector<vector<int>> support;
	vector<double> constant;	
};

typedef vector<Stencil> StencilList;

struct Edge{
	int nx,ny; //Information on orientation of the boundaries
	vector<double> loc {0,0,0}; //Information on the boundary location
	int bcType {-1};	//Type of boundary condition
	double bcInfo {1.0};	//Information required for implementing the boundary condition
	StencilList ghost; //Ghost point implicitly represented in terms of the interior points
};

typedef vector<Edge> EdgeList;

struct Cell{
	int id {-1};	//Number used to identify the cell
	double x,y;	//Coordinates of the cell center
	vector<double> X {0,0}; //Coordinates of the face centers
	vector<double> Y {0,0};
	vector<int> edges {-1,-1,-1,-1};	//List of boundaries adjacent to the cell
};

typedef vector<vector<Cell>> CellList;

class Grid{
private:
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
	void ExportData();
	bool ParseDataFile(ifstream& fs);
public:
	int N; //Total number of cells
	vector<vector<double>> Nx;	//Number of cells in the x-direction
	vector<vector<double>> Ny;	//Number of cells in the y-direction
	vector<double> X,Y;	//Location of the cell faces
	vector<double> hx, hy;	//Grid spacing in x and y directions
	CellList cells;	//List of cells in the grid
	EdgeList edges;	//List of boundaries
	bool setup = false;
	void ShowEdges(bool BC = false);
	bool inDomain(int, int);
	Grid(char* fname);
};

bool equals(double, double);


#endif