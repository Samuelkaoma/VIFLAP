"""Experiment scripts for training and evaluating the acoustic stack.

Deliberately outside the ``viflap`` package. These are the harness that drives
the system, not part of it: they read corpora from disk, spawn worker processes
and write reports, none of which belongs behind the layering the architecture
tests enforce. Keeping them separate means the library stays free of a
dependency on any particular corpus layout.
"""
