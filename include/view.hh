#ifndef VIEW_HH
#define VIEW_HH

#include <blob.hh>
#include <iostream>

// Print detailed information about all layers in the network
void viewNetwork( caffe::NetParameter & net, int from, int to );
void viewNetwork( caffe::NetParameter & net );

// Print detailed information of a single layer
void viewLayer( caffe::LayerParameter * layer );

// Print SolverParameter information
void listSolverObjects( caffe::SolverParameter & solver );

std::ostream & operator<<( std::ostream &, std::vector<int> );
#endif