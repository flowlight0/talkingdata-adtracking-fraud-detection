CXX = g++ -std=c++11
CXXFLAGS = -g -Wall -O3 -I/opt/conda/include
SRCS = ${shell find . -type f  | grep \.cc | grep -v "_main" }
OBJS = $(addprefix build/, $(SRCS:.cc=.o))
MAIN_PROG = ${shell find . -type f  | grep "_main.cc" | sed s/\.cc//g } 

all: ${MAIN_PROG}

%_main: ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ $@.cc $^ -L/opt/conda/lib -larrow

build/%.o : %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $^ -o $@

.PHONY: clean
.SECONDARY: $(OBJS)

clean:
	rm -rf build ${MAIN_PROG}

# brew install lz4
# brew install zstd
# brew install brotli
