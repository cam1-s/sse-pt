// pseudo-random number generator

#ifndef prng_hh_
#define prng_hh_

#include <cstdint>
#include <cstdio>
#include <thread>

#include "inline.hh"

namespace prng
{

// + 1 to ensure it isn't zero
thread_local uint32_t s0 = std::hash<std::thread::id>()(std::this_thread::get_id()) + 1;

__always_inline uint32_t next()
{
	// xorshift32

	uint32_t s = s0;
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 15;
	s0 = s;
	return s;
}

// converts the top 23 bits of an integer to a positive float between 0 and 1.
__always_inline float itof(uint32_t f)
{
	union { float f; uint32_t u; } u = { .u = 0x3F800000U | (f >> 9) };
	return 2.f - u.f;
}

}

#endif
