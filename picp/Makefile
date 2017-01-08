APP_NAME=icp

OBJS=icp.o

default: $(APP_NAME)

IDIR = ./library/eigen
CXX = g++ -m64 -std=c++11
CXXFLAGS = -I$(IDIR) -O3 -Wall -fopenmp -Wno-unknown-pragmas

GXX = nvcc
GXXFLAGS = -I$(IDIR) -O3 -m64 -std=c++11 -arch=compute_20 ./src/icp.cu -o icpgpu
# Compilation Rules
$(APP_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

cpu: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(APP_NAME) $(OBJS)
gpu: $(OBJS)
	$(GXX) $(GXXFLAGS)

%.o: ./src/%.cpp
	$(CXX) $< $(CXXFLAGS) -c -o $@

clean:
	/bin/rm -rf *.o $(APP_NAME)
