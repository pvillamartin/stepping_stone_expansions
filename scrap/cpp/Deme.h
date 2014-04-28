#include "Individual.h"

class Deme{
  const int _num_alleles;
  int _num_members;
  
  std::vector<Individual> _members;
  std::vector<int> _binned_alleles;

 public:
  Deme(int num_alleles, std::vector<Individual> members);
  //void reproduce();
  //void get_alleles();
  template <int> std::vector<int> bin_alleles();
};
