
import unittest

import numpy as np
from scipy import signal

from ..sift import sift

class test_cycles(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 1000
        self.seconds = 2
        self.time_vect = np.linspace(0,self.seconds,self.sample_rate*self.seconds)
        self.signal = np.sin( 2*np.pi*10*self.time_vect )[:,None]

    def cycle_generator(self,f,phase=np.pi,distort=None):
        from ..utils import get_cycle_inds
        from ..spectra import frequency_stats

        x = np.sin( 2*np.pi*f*self.time_vect + phase )[:,None]

        # Add a wobble
        if distort is not None:
            x[distort-25:distort+25,0] += np.linspace(-.1,.1,50)

        # This is a perfect sin so we can use normal hilbert
        IP = np.angle(signal.hilbert(x,axis=0)) + np.pi
        # Find good cycles
        cycles = get_cycle_inds(IP)[:,0]

        return cycles

    def test_simple_cycle_counting(self):

        # Test basic cycle detection
        uni_cycles = np.unique(self.cycle_generator( 4,phase=1.5*np.pi ))
        assert( np.all(uni_cycles==np.arange(9)) )

        uni_cycles = np.unique(self.cycle_generator( 5,phase=1.5*np.pi ))
        assert( np.all(uni_cycles==np.arange(11)) )

    def test_cycle_count_with_bad_start_and_end(self):

        # Test basic cycle detection
        cycles = self.cycle_generator( 4,phase=1.5 )
        uni_cycles = np.unique(cycles)
        assert( np.all(uni_cycles==np.arange(8)) )
        assert( cycles[50] == 0 )
        assert( cycles[1950] == 0 )

        cycles = self.cycle_generator( 5,phase=1.5 )
        uni_cycles = np.unique(cycles)
        assert( np.all(uni_cycles==np.arange(10)) )
        assert( cycles[50] == 0 )
        assert( cycles[1950] == 0 )

    def test_cycle_count_with_bad_in_middle(self):

        cycles = self.cycle_generator( 4,phase=1.5*np.pi,distort=1100 )
        uni_cycles = np.unique(cycles)
        assert( np.all(uni_cycles==np.arange(8)) )
        assert( cycles[1100] == 0 )
