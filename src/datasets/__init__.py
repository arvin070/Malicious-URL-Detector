"""Multi-source URL dataset pipeline for URLGuard.

sources  -> declarative registry of feeds, with licence and trust metadata
fetch    -> cached downloads that record provenance
normalize-> per-source parsers that emit one common schema
"""
