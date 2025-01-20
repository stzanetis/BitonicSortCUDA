NVCC = nvcc

UTILS_DIR = ../Utilities
UTILS = $(UTILS_DIR)/utilities.cpp
TARGET = V2.exe
SRC := V2.cu
OBJS := utilities.obj V2.obj

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@

utilities.obj: $(UTILS)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

V2.obj: $(SRC)
	$(NVCC) -I$(UTILS_DIR) -c $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	del /Q $(OBJS) $(TARGET) *.exp *.lib
else
	rm -f $(OBJS) $(TARGET) *.exp *.lib
endif

run: $(TARGET)
	./$(TARGET) 27