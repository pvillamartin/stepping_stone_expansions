#include "random_cython.h"
#include <iostream>
#include <random>
#include <chrono>

int main()
{
  
  typedef std::chrono::high_resolution_clock myclock;
  auto since_epoch = myclock::now().time_since_epoch();
  double seed = since_epoch.count();
  uniform_random derp = uniform_random(0, 5, seed);
  
  std::cout << derp.get_random() << std::endl;
  std::cout << derp.get_random() << std::endl;
  std::cout << derp.get_random() << std::endl;
  std::cout << derp.get_random() << std::endl;
  std::cout << derp.get_random() << std::endl;
};
