CXX = g++ -std=c++11
CXXFLAGS = -g -Wall -O3
SRCS = ${shell find . -type f  | grep \.cc}
OBJS = $(addprefix build/, $(SRCS:.cc=.o))

all: interval_count

interval_count: ${OBJS}
	${CXX} ${CXXFLAGS} -o $@ $^  -larrow

build/%.o : %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $^ -o $@


clean:
	rm ${OBJS}

# brew install lz4
# brew install zstd
# brew install brotli
