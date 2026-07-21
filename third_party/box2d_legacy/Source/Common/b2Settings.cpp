/*
* Copyright (c) 2006-2007 Erin Catto http://www.gphysics.com
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/

#include "b2Settings.h"
#include <cstddef>
#include <cstdlib>
#include <limits>

std::atomic<int32> b2_byteCount{0};

namespace
{
// IRISU SAFETY PATCH: r58's four-byte prefix misaligned every allocation on
// 64-bit targets. A max-aligned header keeps the returned payload aligned.
struct alignas(std::max_align_t) b2AllocationHeader
{
	int32 size;
};
}

// Memory allocators. Modify these to use your own allocator.
void* b2Alloc(int32 size)
{
	b2Assert(size >= 0);
	b2Assert(size <= std::numeric_limits<int32>::max() - (int32)sizeof(b2AllocationHeader));
	const int32 allocationSize = size + (int32)sizeof(b2AllocationHeader);
	b2AllocationHeader* header =
		(b2AllocationHeader*)malloc((size_t)allocationSize);
	b2Assert(header != NULL);
	header->size = allocationSize;
	b2_byteCount.fetch_add(allocationSize, std::memory_order_relaxed);
	return header + 1;
}

void b2Free(void* mem)
{
	if (mem == NULL)
	{
		return;
	}

	b2AllocationHeader* header = (b2AllocationHeader*)mem - 1;
	const int32 size = header->size;
	const int32 previous = b2_byteCount.fetch_sub(size, std::memory_order_relaxed);
	b2Assert(previous >= size);
	free(header);
}
