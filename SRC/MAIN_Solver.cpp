#include <iostream>
#include <fstream>
#include <string.h>
#include "Grid.h"

using namespace std;

int main(int argc, char const *argv[])
{
	if(argc < 3)
		cout << "Grid data file or simulation data file not provided!\n";
	else{
		Grid *grid = new Grid((char *)argv[1]);
		if(!grid->setup) cout << "Grid setup failed!\n";
	}

	return 0;
}