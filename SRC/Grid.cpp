#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include "Grid.h"

using namespace std;

Grid::Grid(char* fname){
	ifstream fs{fname};
	if(ParseDataFile(fs)) SetupGrid();
	fs.close();
	// ExportData();
}

void Grid::SetupGrid(){
 	if(GenerateEdges() && GenerateFaces()) {
 		GenerateCells(); 
 		Cleanup();
 		Show();
 		setup = true;
 	}
}

//Generates edge information from vertices
bool Grid::GenerateEdges(){
 	cout<<"Generating Edges...\n";

 	vertices.push_back(vertices[0]);

 	for(int i = 1; i < vertices.size(); i++){
 		edges.push_back({});
 		if(vertices[i][0] == vertices[i-1][0]){
 			edges[i-1].loc[0] = vertices[i][0];
 			if(vertices[i][1] > vertices[i-1][1]){
 				edges[i-1].loc[1] = vertices[i-1][1];
 				edges[i-1].loc[2] = vertices[i][1];
 				edges[i-1].nx = -1;
 				xrange[0] = (vertices[i][0] < xrange[0])?vertices[i][0]:xrange[0];
 			}
 			else{
 				edges[i-1].loc[2] = vertices[i-1][1];
 				edges[i-1].loc[1] = vertices[i][1];
 				edges[i-1].nx = 1;
 				xrange[1] = (vertices[i][0] > xrange[1])?vertices[i][0]:xrange[1];
 			}
 		}
 		else if(vertices[i][1] == vertices[i-1][1]){
 			edges[i-1].loc[0] = vertices[i][1];
 			if(vertices[i][0] > vertices[i-1][0]){
 				edges[i-1].loc[1] = vertices[i-1][0];
 				edges[i-1].loc[2] = vertices[i][0];
 				edges[i-1].ny = 1;
 				yrange[1] = (vertices[i][1] > yrange[1])?vertices[i][1]:yrange[1];
 			}
 			else{
 				edges[i-1].loc[2] = vertices[i-1][0];
 				edges[i-1].loc[1] = vertices[i][0];
 				edges[i-1].ny = -1;
 				yrange[0] = (vertices[i][1] < yrange[0])?vertices[i][1]:yrange[0];
 			}
 		}
 		else{
 			cout<<"Edges should be parallel to the x-axis or y-axis\n";
 			return false;
 		}
 	}

 	return true;
}

//Generates cell faces
bool Grid::GenerateFaces(){
 	cout<<"Generating Faces...\n";

 	X.push_back(xrange[0]);
 	Y.push_back(yrange[0]);

 	double h {0.0};
 	for(auto& i: Nx){
 		if(!equals(i[0], X.back()) || (X.size() == 1 && i[2] <= 0)) {
 			cout<<"Invalid specification for number of cells\n"; 
 			return false;
 		}
 		if(i[3] >0){
 			if(i[2] <= 0) i[2] = ceil(log((i[1]-i[0])*(i[3]-1)/h + 1)/log(i[3]));
 			h = (i[1]-i[0])*(i[3]-1)/(pow(i[3],i[2])-1);
 			double x = i[0]; 	
 			for(int j = 0; j<i[2]; j++) {X.push_back(x+=h); hx.push_back(h); h*=i[3];}		
 		}
 		else if(i[3] == -1){
 			if(i[2] <= 0) i[2] = ceil((i[1]-i[0])/h);
 			h = (i[1]-i[0])/i[2];
 			double x = i[0];
 			for(int j = 0; j<i[2]; j++) {X.push_back(x+=h); hx.push_back(h);}
 		}
 		else {cout<<"Invalid specification for number of cells\n"; return false;}
 	}

 	for(auto& i: Ny){
 		if(!equals(i[0], Y.back()) || (Y.size() == 1 && i[2] <= 0)) {
 			cout<<"Invalid specification for number of cells\n"; 
 			return false;
 		}
 		if(i[3] >0){
 			if(i[2] <= 0) i[2] = ceil(log((i[1]-i[0])*(i[3]-1)/h + 1)/log(i[3]));
 			h = (i[1]-i[0])*(i[3]-1)/(pow(i[3],i[2])-1);
 			double x = i[0]; 	
 			for(int j = 0; j<i[2]; j++) {Y.push_back(x+=h); hy.push_back(h); h*=i[3];}		
 		}
 		else if(i[3] == -1){
 			if(i[2] <= 0) i[2] = ceil((i[1]-i[0])/h);
 			h = (i[1]-i[0])/i[2];
 			double x = i[0];
 			for(int j = 0; j<i[2]; j++) {Y.push_back(x+=h); hy.push_back(h);}
 		}
 		else {cout<<"Invalid specification for number of cells\n"; return false;}
 	}

 	if(!equals(xrange[1], X.back()) || !equals(yrange[1], Y.back())) {
 		cout<<"Invalid specification for number of cells\n"; 
 		return false;
 	}

 	return true;
}

//Constructs the discrete cells
void Grid::GenerateCells(){
 	cout<<"Generating Cells...\n";

 	for(int i = 1; i < X.size(); i++){
 		cells.push_back(vector<Cell>(Y.size()-1));
 		for(int j = 1; j < Y.size(); j++){
 			Cell& cell = cells[i-1][j-1];
 			cell.X[0] = X[i-1];
 			cell.X[1] = X[i];
 			cell.Y[0] = Y[j-1];
 			cell.Y[1] = Y[j];
 			cell.x = 0.5*(cell.X[0]+cell.X[1]);
 			cell.y = 0.5*(cell.Y[0]+cell.Y[1]);
 		}
 	}
}

//Removes unnecessary cells
void Grid::Cleanup(){
 	int n = 0;
 	for(auto& i: cells){
 		for(auto& j: i){
 			if(Interior(j)) {
 				j.id = n++;
 				double ar = (j.X[1]-j.X[0])/(j.Y[1]-j.Y[0]);
 				AR[0] = (AR[0] > ar)?ar:AR[0];
 				AR[1] = (AR[1] < ar)?ar:AR[1];
 			}
 		}
 	}
 	N = n;
}

//Checks whether a given cell lies inside the computational domain
bool Grid::Interior(Cell& cell){
 	int intersect = 0;
 	for(int i = 0; i < edges.size(); i++){
 		Edge e = edges[i];
 		if(e.nx != 0){
 			if(cell.y > e.loc[1] && cell.y < e.loc[2]){
 				if(e.loc[0] > cell.x) intersect++;
 				if(e.nx == -1 && equals(cell.X[0], e.loc[0])) cell.edges[0] = i;
 				if(e.nx == 1 && equals(cell.X[1], e.loc[0])) cell.edges[1] = i;
 			} 
 		}
 		else{
 			if(cell.x > e.loc[1] && cell.x < e.loc[2]){
 				if(e.ny == -1 && equals(cell.Y[0], e.loc[0])) cell.edges[2] = i;
 				if(e.ny == 1 && equals(cell.Y[1], e.loc[0])) cell.edges[3] = i;
 			} 
 		}
 	}
 	if(intersect%2) return true;
 	return false;
}

//Outputs important information to the terminal
void Grid::Show(){
 	cout<<setprecision(2);
 	ShowEdges();
 	cout<<endl;
 	cout<<"X:\t"<<xrange[0]<<"\t"<<xrange[1]<<endl;
 	cout<<"Y:\t"<<yrange[0]<<"\t"<<yrange[1]<<endl;
 	cout<<"AR:\t"<<AR[0]<<"\t"<<AR[1]<<endl;
 	cout<<"hx:\t"<<*min_element(hx.begin(),hx.end())<<"\t"
 					<<*max_element(hx.begin(),hx.end())<<endl;
 	cout<<"hy:\t"<<*min_element(hy.begin(),hy.end())<<"\t"
 					<<*max_element(hy.begin(),hy.end())<<endl<<endl;
}

//Prints information about the boundaries
void Grid::ShowEdges(bool BC){
 	cout<<endl;
 	for(int i = 0; i < edges.size(); i++){
 		Edge e = edges[i];
 		cout<<"Edge\t"<<i<<":\t"<<"n\t"<<e.nx<<"\t"<<e.ny<<"\t";
 		if(e.nx != 0) cout<<"x\t"<<e.loc[0]<<"\ty\t"<<e.loc[1]<<"\t"<<e.loc[2]<<endl;
 		else cout<<"y\t"<<e.loc[0]<<"\tx\t"<<e.loc[1]<<"\t"<<e.loc[2]<<endl;
 		if(BC) cout<<"BC\t"<<e.bcType<<"\t"<<e.bcInfo<<endl;
 	}
}

//
bool Grid::inDomain(int i, int j){
	if(i < 0 || i >= hx.size() || j < 0 || j >= hy.size()) return false;
	if(cells[i][j].id < 0) return false;
	return true;
}

// //Saves cell center information to a data file
// void Grid::ExportData(){
//  	ofstream fs{"CellCenters.csv"};
//  	for(auto& i: cells){
//  		for(auto& j: i){
//  			if(j.id != -1){
//  				fs<<j.x<<","<<j.y;
//  				for(int k = 0; k < edges.size(); k++){
//  					if(j.edges.find(k) != j.edges.end()) fs<<","<<1;
//  					else fs<<","<<0;
//  				}
//  				fs<<endl;
//  			}
//  		}
//  	}
//  	fs.close();
// }

//Reads user-provided parameters from the grid data file
bool Grid::ParseDataFile(ifstream& fs){
	string inp{};
	stringstream s;
	bool err = false;

	if(!fs){
		cout << "Grid data file not found!\n";
		return false;
	}

	while(!fs.eof()){
		fs>>inp;
		if(fs.eof()) break;
		if(inp == "Vertices") {
			if((fs>>ws).get() != '{') {err = true; break;}
			while(fs.good()) {
				if((fs>>ws).peek() == '}') {fs.ignore(); break;}
				vertices.push_back({0,0});
				fs>>vertices.back()[0]>>vertices.back()[1];
				if(fs.eof()) {err = true; break;}
			}
		}
		else if(inp == "Nx") {
			if((fs>>ws).get() != '{') {err = true; break;}
			while(fs.good()){
				if((fs>>ws).peek() == '}') {fs.ignore(); break;}
				Nx.push_back(vector<double>(4,-1));
				getline(fs, inp);
				s.str(inp);
				s>>Nx.back()[0]>>Nx.back()[1]>>Nx.back()[2]>>Nx.back()[3];
				s.clear();
				if(fs.eof()) {err = true; break;}
			}
		}
		else if(inp == "Ny") {
			if((fs>>ws).get() != '{') {err = true; break;}
			while(fs.good()){
				if((fs>>ws).peek() == '}') {fs.ignore(); break;}
				Ny.push_back(vector<double>(4,-1));
				getline(fs, inp);
				s.str(inp);
				s>>Ny.back()[0]>>Ny.back()[1]>>Ny.back()[2]>>Ny.back()[3];
				s.clear();
				if(fs.eof()) {err = true; break;}
			}
		}
		else {err = true; break;}

		if(!fs.good()) {err = true; break;}
	}

	if(err){
		cout<<"Invalid data file format!\n";
		return false;
	}
	return true;
}

bool Grid::equals(double a, double b){
	return (abs(a-b) > TOL)?false:true;
}
