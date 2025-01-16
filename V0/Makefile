NVCC := $(CUDA_PATH)/bin/nvcc

TARGET := V0.exe
SRC := V0.cu
CU_SRC := $(SRC:.cu=.o)
CPP_SRC := utilities.cpp
OBJ := $(CU_SRC:.cu=.o) $(CPP_SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -o $@ $^

%.o: %.cu
    $(NVCC) -c $< -o $@

%.o: %.cpp
    $(CXX) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET)