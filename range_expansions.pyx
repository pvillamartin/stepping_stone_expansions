__author__ = 'bryan'

cimport numpy as np

from libc.stdlib cimport rand, RAND_MAX

cdef class Individual:

    cdef readonly long allele_id

    def __init__(Individual self, long allele_id):
        self.allele_id = allele_id

cdef class Deme:

    cdef readonly Individual[:] members
    cdef readonly long num_alleles
    cdef readonly long[:] binned_alleles
    cdef readonly long num_members

    def __init__(Deme self,  long num_alleles, Individual[:] members):
        self.members = members
        self.num_members = len(members)
        self.num_alleles = num_alleles
        self.binned_alleles = self.bin_alleles()

    cdef reproduce(Deme self):
        cdef int to_reproduce
        cdef int to_die

        to_reproduce = <int>(self.num_members*rand()/RAND_MAX)
        to_die = <int>(self.num_members*rand()/RAND_MAX)
        # Update allele array
        self.binned_alleles[self.members[to_die].allele_id] -= 1
        self.binned_alleles[self.members[to_reproduce].allele_id] += 1
        # Update the members
        self.members[to_die] = self.members[to_reproduce]

    cdef get_alleles(Deme self):
        return [individual.allele_id for individual in self.members]

    cdef bin_alleles(Deme self):
        return np.bincount(self.get_alleles(), minlength=self.num_alleles)

def simulate_deme(Deme deme, long numAlleles, long numIterations=100):
    cdef long[:,:] history

    history = np.empty((numIterations, numAlleles), dtype=np.long)
    cdef long i
    for i in range(numIterations):
        history[i, :] = deme.binned_alleles
        deme.reproduce()
    return history