#include "Individual.h"

Individual::Individual(int allele_id){
  _allele_id = allele_id;
}

int Individual::get_allele_id(){
  return _allele_id;
}
