#include <iostream>
#include <random>
#include <chrono>
#include "random_cython.h"

uniform_random::uniform_random(int low, int high, double seed):
  _low(low),
  _high(high),
  _seed(seed),
  _generator(seed),
  _distribution(low, high)
{};

int uniform_random::get_random(){
  return _distribution(_generator);
};
