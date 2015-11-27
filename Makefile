CXX := g++ -std=c++11 -g
CAFFE := /home/teq/Desktop/caffe/caffe
LIB := $(CAFFE)/build/lib
INCLUDE := $(CAFFE)/build/src/
all:
	$(CXX) -c blob.cc -I$(INCLUDE) -I./
	$(CXX) -c modify.cc -I$(INCLUDE) -I./
	$(CXX) -o modify modify.o blob.o -L$(LIB) -lcaffe -lprotobuf