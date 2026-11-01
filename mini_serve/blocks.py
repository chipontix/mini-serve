"""KV block pool with reference counts and a prefix cache.

The pool hands out physical block ids; what is stored in a block is the engine's business. A block
holds `page_size` tokens of K and V for every layer and KV head.

Prefix cache. A full block is identified by a hash of its tokens chained with the hash of the
block before it, so equal hashes mean equal token prefixes, not just equal blocks. Once a block is
full and computed it is registered under that hash. A new sequence whose prompt starts with the
same tokens takes those blocks instead of recomputing them: the block's reference count goes up,
and it is shared read-only. Only full blocks are ever shared, and a sequence only writes into its
own last, partly filled block, so no copy-on-write is needed.

A block whose reference count drops to zero goes to the free list but keeps its hash: it can
still be hit until it is reused. The free list is LRU-ordered, so the blocks freed longest ago
are reused (and lose their cached content) first.
"""

from collections import OrderedDict


def block_hashes(tokens, page_size, start=0, prev=None):
    """Chained hashes of the full blocks of `tokens`, from block index `start` on.
    `prev` is the hash of block start-1 (None for the first block)."""
    out = []
    for i in range(start, len(tokens) // page_size):
        prev = hash((prev, tuple(tokens[i * page_size:(i + 1) * page_size])))
        out.append(prev)
    return out


class BlockPool:
    def __init__(self, num_blocks, page_size, prefix_cache=True):
        self.num_blocks = num_blocks
        self.page_size = page_size
        self.prefix_cache = prefix_cache
        self.ref = [0] * num_blocks
        self.free = OrderedDict((b, None) for b in range(num_blocks))  # LRU first
        self.hash_of = {}   # block -> hash, for registered blocks
        self.by_hash = {}   # hash -> block
        self.hits = 0       # blocks taken from the cache instead of computed

    def num_free(self):
        return len(self.free)

    def allocate(self):
        """A fresh block with refcount 1. Evicts the least recently freed block's cached content."""
        if not self.free:
            raise RuntimeError("block pool exhausted")
        b, _ = self.free.popitem(last=False)
        h = self.hash_of.pop(b, None)
        if h is not None:
            del self.by_hash[h]
        self.ref[b] = 1
        return b

    def release(self, blocks):
        for b in blocks:
            assert self.ref[b] > 0, f"double free of block {b}"
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free[b] = None  # most recently freed: evicted last

    def lookup(self, hashes):
        """Blocks of the longest cached prefix of `hashes`, without taking them."""
        out = []
        if self.prefix_cache:
            for h in hashes:
                b = self.by_hash.get(h)
                if b is None:
                    break
                out.append(b)
        return out

    def take(self, blocks):
        """Take a reference on cached blocks returned by lookup()."""
        for b in blocks:
            if self.ref[b] == 0:
                del self.free[b]
            self.ref[b] += 1
        self.hits += len(blocks)

    def register(self, block, h):
        """Mark a full, computed block as holding the prefix with hash `h`. If another block
        already holds the same prefix (two sequences computed it at the same time), that one
        stays the cached copy and this block stays private."""
        if self.prefix_cache and h not in self.by_hash:
            self.by_hash[h] = block
            self.hash_of[block] = h
