"""The model-agnostic layer.

Everything here works on ``[n_vectors, dim]`` float arrays and knows nothing about
pages, images or a particular encoder. See ``docs/package/UNIVERSAL_PACKAGE_DESIGN.md``.

A model-agnostic interface is not model-independent compression quality: the same
pipeline can keep retrieval quality on one encoder and lose it on another, or in
another coordinate basis of the same encoder. The selection layer measures; it may
abstain.
"""
