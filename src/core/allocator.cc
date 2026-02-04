#include "core/allocator.h"
#include <map>
#include <unordered_map>
#include <utility>

namespace infini
{
    using FreeList = std::map<size_t, size_t>; // offset -> size
    static std::unordered_map<Allocator *, FreeList> g_free_lists;

    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }

        // Cleanup bookkeeping for this Allocator instance.
        g_free_lists.erase(this);
    }

    size_t Allocator::alloc(size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // Strategy: a simple first-fit allocator over a free-list.
        // - Try to reuse an existing free block.
        // - If there is a free block at the end (offset+size==peak), allow
        //   "extend-in-place" for larger requests (no gap left).
        // - Otherwise, bump `peak`.

        auto &free_list = g_free_lists[this];

        // 1) First-fit on free blocks.
        for (auto it = free_list.begin(); it != free_list.end(); ++it)
        {
            const size_t off = it->first;
            const size_t blk = it->second;
            if (blk >= size)
            {
                const size_t addr = off;
                free_list.erase(it);
                if (blk > size)
                {
                    free_list.emplace(off + size, blk - size);
                }
                used += size;
                return addr;
            }
        }

        // 2) Special case: free block at the end can be extended.
        if (!free_list.empty())
        {
            auto it = std::prev(free_list.end());
            const size_t off = it->first;
            const size_t blk = it->second;
            if (off + blk == peak && blk < size)
            {
                const size_t addr = off;
                free_list.erase(it);
                peak += (size - blk);
                used += size;
                return addr;
            }
        }

        // 3) Bump allocation.
        const size_t addr = peak;
        peak += size;
        used += size;
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        IT_ASSERT(used >= size);
        used -= size;

        auto &free_list = g_free_lists[this];

        // Insert and coalesce with neighbors.
        auto it_next = free_list.lower_bound(addr);
        size_t new_off = addr;
        size_t new_sz = size;

        // Merge with previous block if adjacent.
        if (it_next != free_list.begin())
        {
            auto it_prev = std::prev(it_next);
            if (it_prev->first + it_prev->second == addr)
            {
                new_off = it_prev->first;
                new_sz += it_prev->second;
                free_list.erase(it_prev);
            }
        }

        // Merge with next block if adjacent.
        if (it_next != free_list.end())
        {
            if (new_off + new_sz == it_next->first)
            {
                new_sz += it_next->second;
                free_list.erase(it_next);
            }
        }

        // If this free range reaches the end, shrink peak and avoid keeping
        // a tail free block.
        if (new_off + new_sz == peak)
        {
            peak = new_off;
            return;
        }

        free_list.emplace(new_off, new_sz);
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
