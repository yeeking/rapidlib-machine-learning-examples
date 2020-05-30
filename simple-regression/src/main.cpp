#include <vector>
#include <iostream>
#include <cassert>
#include <random>
#include <algorithm>

#include "../../lib/rapidLib.h"

#include <iostream>

using namespace rapidLib;

int main(int argc, const char * argv[]) 
{
    std::cout << "Here we go " << std::endl;
    rapidLib::regression myNN_nodes;
    myNN_nodes.setNumHiddenNodes(10);
  
    std::vector<rapidLib::trainingExample> trainingSet;
    rapidLib::trainingExample  tempExample;
    tempExample.input = { 0.2, 0.7 };
    tempExample.output = { 3.0, 0.0, 127 };
    trainingSet.push_back(tempExample);

    tempExample.input = { 2.0, 44.2 };
    tempExample.output = { 20.14, 64, 87 };
    trainingSet.push_back(tempExample);

    myNN_nodes.train(trainingSet);
  
    // write the trained network to disk
    //std::string filepath = "./NN_test.json";
    //myNN_nodes.writeJSON(filepath);
    //  setting up a network from a json string
    //rapidLib::regression myNNfromString;
    //myNNfromString.putJSON(myNN_nodes.getJSON());

    // setting up a network from a file
    //rapidLib::regression myNNfromFile;
    //myNNfromFile.readJSON(filepath);
    
    std::vector<double> inputVec = { 2.0, 44.2 };
    std::cout << "Known input: " << myNN_nodes.run(inputVec)[0] << std::endl;

    std::vector<double> inputVec2 = { 2.0, 40.2 };
    std::cout << "Unkown, close input: " << myNN_nodes.run(inputVec2)[0] << std::endl;
      
   std::vector<double> inputVec3 = { 0.1, 2.2 };
    std::cout << "Unkown, far input: " << myNN_nodes.run(inputVec3)[0] << std::endl;
    
    
    return 0;
}
