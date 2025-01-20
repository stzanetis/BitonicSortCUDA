NVCC = nvcc

UTILS_DIR = ../Utilities
UTILS = $(UTILS_DIR)/utilities.cpp
TARGET = V1.exe
SRC := V1.cu
OBJS := utilities.obj V1.obj

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@

utilities.obj: $(UTILS)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

V1.obj: $(SRC)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	del /Q $(OBJS) $(TARGET) *.exp *.lib
else
	rm -f $(OBJS) $(TARGET) *.exp *.lib
endif

run: $(TARGET)
	./$(TARGET) 27