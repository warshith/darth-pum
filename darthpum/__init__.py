"""A reproduction of DARTH-PUM (Wong, Feinberg & Ghose, ASPLOS '26).

The authors' simulator was not released. Their two upstream components were:
MASTODON (digital PUM, their own lab, MIT licensed) and CrossSim (analog
crossbar, Sandia National Labs, BSD-3). Both are public. What is missing -- and
what this package implements -- is the hybrid coordination layer that is the
paper's actual contribution.
"""

__version__ = "0.1.0"
