"""Delivery mechanisms.

The outermost layer. Nothing in the system depends on this package; it depends
on everything else. Adding a second delivery mechanism — a command line, a batch
processor — requires no change anywhere below.
"""
