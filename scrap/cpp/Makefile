flags= -std=c++11 -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual

all: test

test: Deme.o Individual.o
	g++ $(flags) Deme.o Individual.o -o test

Individual.o:	Individual.cpp
	g++ $(flags) -c Individual.cpp

Deme.o: Individual.h Deme.cpp
	g++ $(flags) -c Deme.cpp Individual.h

clean:
	rm -rf *o
