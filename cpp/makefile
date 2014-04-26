flags= -std=c++11 -Wall -Wextra -Wfloat-equal -Wundef -Wcast-align -Wwrite-strings -Wlogical-op -Wmissing-declarations -Wredundant-decls -Wshadow -Woverloaded-virtual

all:	main

main: main.o random_cython.o
	g++ $(flags) main.o random_cython.o -o main

main.o: main.cpp
	g++ $(flags) -c main.cpp

random_cython.o: random_cython.cpp random_cython.h
	g++ $(flags) -c random_cython.cpp

clean: 
	rm *.o
