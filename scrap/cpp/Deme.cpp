#include "Deme.h"

Deme::Deme(int num_alleles, std::vector<Individual> members):
  _num_alleles(num_alleles),
  _num_members(members.size()),
  _members(members)
{
  _binned_alleles = bin_alleles();
}

std::vector<int> template <int> std::vector<int> Deme::bin_alleles(){
  //Iterate over the vector and count the number of each type
  //Create a vector of the length of the number of alleles
  std::vector<int> allele_vec (_num_alleles,0);

for(std::vector<T>::iterator it = allele_vec.begin(); it != allele_vec.end(); ++it){
  int id = it.get_allele_id();
  allele_vec[id] += 1;
}

 return allele_vec
};
