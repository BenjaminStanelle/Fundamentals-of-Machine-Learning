#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <direct.h>
#include <limits.h>
#include <iomanip>
#include <unordered_map>
#include <cmath>
//#include <filesystem>

using namespace std;
//Function Prototypes and simplications:

/* Inputs: UCI Dataset file, reference to vector of vectorFloats used to store data, reference to  class labels vector of floats
   Output: updates outputData vector and classLabels with attributes from file(first 16 columns), and class label (column 17)*/
float LoadData(string fileName, vector<vector<float>>& outputData, vector<float>& classLabels);

template<typename T, typename A>
unordered_map<T, int> MapClassLabelsToInts(vector<T, A> const& classLabels);

void NormalizeVectorByFloat(vector<vector<float>>& trainFileData, float largestFloat);

int main(int argc, char* argv[]) {
	/* Checking Valid Command Line Arguments ====================================================== */

	string trainingFileName = argv[1];
	string testFileName = argv[2];

	const int layers = stoi(argv[3]);
	int rounds;
	int unitsPerlayer;

	// Specific command line argument parameters for this assignment.
	if (layers < 2) {
		cout << "Layers must be 2 or larger, please enter a larger value" << endl;
		return 0;
	}
	else if (layers == 2) {
		rounds = stoi(argv[4]);
	}
	else {
		unitsPerlayer = stoi(argv[4]);
		rounds = stoi(argv[5]);
	}

	/* Training Stage ====================================================== */
	vector<vector<float>> trainFileData;
	vector<float> classLabels;
	float largestFloat = LoadData(trainingFileName, trainFileData, classLabels);
	
	int count1 = 1;
	auto uniqueClassLabels = MapClassLabelsToInts(classLabels);
	
	//Normalize the datainput vector, this helps prevent attributes with initially large ranges from outweighing attributes with smaller rangers.
	//all columns will be in a common scale from 0-1;
	NormalizeVectorByFloat(trainFileData, largestFloat);

		//Uniform the weights from the UCI dataset, this helps the neural network not get stuck in a local minimum during gradient descent
	// if we didn't normalize, then all perceptrons start at the same point
	
	return 0;
}



float LoadData(string fileName, vector<vector<float>>& outputData, vector<float> &classLabels) {
	//getting the current working directory
	char buff[255];
	_getcwd(buff, 255);
	string current_working_dir(buff);
	current_working_dir = current_working_dir + "\\" + fileName; //file path to training data set


	float largestFloat = -1;
	ifstream FILE(current_working_dir, ios::in);
	if (FILE.is_open()) {
		string line;
		float inputFloat;
		int innerIDX = 1, outerIDX = 1; //inner index of input file which is the columns number, outter index which is the rows number
		
		outputData.emplace_back(vector<float>()); //creating the first vector in the vector of vectorFloats, create new vector of floats directly in outputData, no temporary.
		while (FILE >> inputFloat) {
			
			if (innerIDX % 17 != 0) {	//When looking at first 1-16 columns, everything but last column.
				outputData[outerIDX - 1].push_back(inputFloat); //
				
				if (inputFloat > largestFloat) { //finding the largest value in the input data file
					largestFloat = inputFloat;
				}
			}
			else {		//last column
				classLabels.push_back(inputFloat); //class labels stored in their own vector for ease of use
				outputData.emplace_back(vector<float>()); //push a vector<float> to vector of vectors
				outerIDX++;	//increment to know we've reached the next line in file
			}
			innerIDX++; //increment to get to next column in file
		}

	}
	FILE.close(); //safety first

	return largestFloat;
}

	// typename A is the allocator, we need this so our function will accept vectors with alternative allocators.
template<typename T, typename A>
unordered_map<T, int> MapClassLabelsToInts(vector<T, A> const& classLabels) {	
	unordered_map<T, int> uniqueClassLabels; //map of all unique class labels with numbering.
	int count1 = 1;
	for (auto& line : classLabels) {
		if (!(uniqueClassLabels.find(line) != uniqueClassLabels.end())) { //if the key is not found in the map already
			uniqueClassLabels[line] = count1;		//put it in the map and increment count
			count1++;
		}
	}
	return uniqueClassLabels;
}


void NormalizeVectorByFloat(vector<vector<float>>& trainFileData, float largestFloat) {
	int i, j;

	for (i = 0; i < trainFileData.size(); i++) {
		for (j = 0; j < trainFileData[i].size(); j++) {
			trainFileData[i][j] = abs(trainFileData[i][j] / largestFloat);
		}
	}

}