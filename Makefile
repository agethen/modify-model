CXX := g++ -std=c++11 -g -Wall
CAFFE := /home/teq/Desktop/caffe/caffe
LIB := -L$(CAFFE)/build/lib
INCLUDE := -I$(CAFFE)/build/src/ -Iinclude/
all:
	$(CXX) -c src/blob.cc $(INCLUDE)
	$(CXX) -c src/view.cc $(INCLUDE)
	$(CXX) -c src/modify.cc $(INCLUDE)
	$(CXX) -o modify modify.o blob.o view.o $(LIB) -lcaffe -lprotobuf