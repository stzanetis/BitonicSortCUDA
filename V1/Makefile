NVCC = nvcc

UTILS_DIR = ../Utilities
UTILS = $(UTILS_DIR)/utilities.cpp
TARGET = V1.exe
SRC := V1.cu
OBJ := utilities.o

all: $(TARGET)

$(TARGET): $(OBJ)
	$(NVCC) -I$(UTILS_DIR) $^ -o $@

$(OBJ): $(SRC)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

V1.o: $(UTILS)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

run: $(TARGET)
	./$(TARGET) 20 7