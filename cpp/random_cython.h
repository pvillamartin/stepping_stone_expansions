#include <random>

class uniform_random {
    const int _low;
    const int _high;
    const double _seed;

    std::mt19937 _generator;
    std::uniform_int_distribution<int> _distribution;

 public:
    uniform_random(int low, int high, double seed);
    int get_random();
};
