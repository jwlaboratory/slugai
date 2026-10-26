"""slugai — slow distributed inference.

Throughput-first, pipeline-parallel LLM decoding across a ring of MacBooks
connected over Wi-Fi. Each Mac holds a contiguous block of layers; only hidden
states cross the network, and micro-batches circulate the ring so that
communication overlaps compute.
"""

__version__ = "0.1.0"
