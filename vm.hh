// vector maths

#ifndef vm_hh_
#define vm_hh_

#include <cstdint>
#include <cmath>

//#define VM_DISABLE_SIMD

#if (defined(__i386__) || defined(__x86_64__)) && !defined(VM_DISABLE_SIMD)
#include <x86intrin.h>
#endif

#include "inline.hh"

#define _pi 3.14159265356f
#define _hpi 1.57079632678f
#define _2pi 6.28318530712f

typedef float float_type;

namespace vm
{

template <class T> __always_inline T max(T const &a, T const &b) { return (b < a) ? a : b; }
template <class T> __always_inline T min(T const &a, T const &b) { return (a < b) ? a : b; }
template <class T> __always_inline T clamp(T const &i, T const &a, T const &b) { return min(max(i, a), b); }
template <class T> __always_inline T mix(T const &a, T const &b, T const &fac) { return fac * b + (1.f - fac) * a; }

__always_inline float rsqrt(float n)
{
	const float x2 = n * .5f;
	union { float f; uint32_t i; } c = { .f = n };
	c.i = 0x5f3759df - (c.i >> 1);
	c.f *= 1.5f - (x2 * c.f * c.f);
	//c.f *= 1.5f - (x2 * c.f * c.f);
	return c.f;
}

// my compiler won't inline std::sin
#ifdef __GNUC__

__always_inline float fast_sin (float x) { return __builtin_sinf (x); }
__always_inline float fast_cos (float x) { return __builtin_cosf (x); }
__always_inline float fast_sqrt(float x) { return __builtin_sqrtf(x); }

__always_inline double fast_sin (double x) { return __builtin_sin (x); }
__always_inline double fast_cos (double x) { return __builtin_cos (x); }
__always_inline double fast_sqrt(double x) { return __builtin_sqrt(x); }

__always_inline long double fast_sin (long double x) { return __builtin_sinl (x); }
__always_inline long double fast_cos (long double x) { return __builtin_cosl (x); }
__always_inline long double fast_sqrt(long double x) { return __builtin_sqrtl(x); }

#else 

template <class T> __always_inline T fast_sin (T x) { return std::sin (x); }
template <class T> __always_inline T fast_cos (T x) { return std::cos (x); }
template <class T> __always_inline T fast_sqrt(T x) { return std::sqrt(x); }

#endif

// VEC2

// todo: redesign tvec2 to be more like tvec3

template <class T> class tvec2
{
public:
	__always_inline tvec2() : x(0), y(0) {  }
	__always_inline tvec2(tvec2<T> const &i) : x(i.x), y(i.y) {  }
	__always_inline tvec2(T ix, T iy) : x(ix), y(iy) {  }
	__always_inline tvec2(T i) : x(i), y(i) {  }

	__always_inline tvec2<T> &operator=(tvec2<T> const &i) { x = i.x; y = i.y; return *this; }

	union { T x, u, s; };
	union { T y, v, t; };

	__always_inline tvec2<T>  operator-() const { return tvec2<T>(-x, -y); }

	__always_inline tvec2<T> &operator-=(tvec2<T> const &i) { x -= i.x; y -= i.y; return *this; }
	__always_inline tvec2<T> &operator-=(	   T  const &i) { x -= i;   y -= i;   return *this; }

	__always_inline tvec2<T> &operator+=(tvec2<T> const &i) { x += i.x; y += i.y; return *this; }
	__always_inline tvec2<T> &operator+=(	   T  const &i) { x += i;   y += i;   return *this; }

	__always_inline tvec2<T> &operator*=(tvec2<T> const &i) { x *= i.x; y *= i.y; return *this; }
	__always_inline tvec2<T> &operator*=(	   T  const &i) { x *= i;   y *= i;   return *this; }

	__always_inline tvec2<T> &operator/=(tvec2<T> const &i) { x /= i.x; y /= i.y; return *this; }
	__always_inline tvec2<T> &operator/=(	   T  const &i) { x /= i;   y /= i;   return *this; }
};

template <class T> tvec2<T> __always_inline operator-(tvec2<T> const &a, tvec2<T> const &b) { return tvec2<T>(a.x - b.x, a.y - b.y); }
template <class T> tvec2<T> __always_inline operator-(tvec2<T> const &a,       T  const &b) { return tvec2<T>(a.x - b  , a.y - b  ); }
template <class T> tvec2<T> __always_inline operator-(      T  const &b, tvec2<T> const &a) { return tvec2<T>(a.x - b  , a.y - b  ); }

template <class T> tvec2<T> __always_inline operator+(tvec2<T> const &a, tvec2<T> const &b) { return tvec2<T>(a.x + b.x, a.y + b.y); }
template <class T> tvec2<T> __always_inline operator+(tvec2<T> const &a,       T  const &b) { return tvec2<T>(a.x + b  , a.y + b  ); }
template <class T> tvec2<T> __always_inline operator+(      T  const &b, tvec2<T> const &a) { return tvec2<T>(a.x + b  , a.y + b  ); }

template <class T> tvec2<T> __always_inline operator*(tvec2<T> const &a, tvec2<T> const &b) { return tvec2<T>(a.x * b.x, a.y * b.y); }
template <class T> tvec2<T> __always_inline operator*(tvec2<T> const &a,       T  const &b) { return tvec2<T>(a.x * b  , a.y * b  ); }
template <class T> tvec2<T> __always_inline operator*(      T  const &b, tvec2<T> const &a) { return tvec2<T>(a.x * b  , a.y * b  ); }

template <class T> tvec2<T> __always_inline operator/(tvec2<T> const &a, tvec2<T> const &b) { return tvec2<T>(a.x / b.x, a.y / b.y); }
template <class T> tvec2<T> __always_inline operator/(tvec2<T> const &a,       T  const &b) { return tvec2<T>(a.x / b  , a.y / b  ); }
template <class T> tvec2<T> __always_inline operator/(      T  const &b, tvec2<T> const &a) { return tvec2<T>(a.x / b  , a.y / b  ); }

typedef tvec2<float_type> vec2;


// VEC3

template <class T> class tvec3
{
public:
	__always_inline tvec3() : _x(0), _y(0), _z(0) {  }
	__always_inline tvec3(tvec3<T> const &i) : _x(i._x), _y(i._y), _z(i._z) {  }
	__always_inline tvec3(T ix, T iy, T iz) : _x(ix), _y(iy), _z(iz) {  }
	__always_inline tvec3(T i) : _x(i), _y(i), _z(i) {  }

	__always_inline T x() const { return _x; }
	__always_inline T y() const { return _y; }
	__always_inline T z() const { return _z; }

	__always_inline T x(T i) { return _x = i; }
	__always_inline T y(T i) { return _y = i; }
	__always_inline T z(T i) { return _z = i; }

	union { T _x, _r; };
	union { T _y, _g; };
	union { T _z, _b; };

	__always_inline float       &operator[](size_t const &i)       { switch (i) { case 0: return _x; case 1: return _y; case 2: return _z; } }
	__always_inline float const &operator[](size_t const &i) const { switch (i) { case 0: return _x; case 1: return _y; case 2: return _z; } }

	__always_inline tvec3<T> &operator=(tvec3<T> const &i) { _x = i._x; _y = i._y; _z = i._z; return *this; }

	__always_inline tvec3<T>  operator-() const { return tvec3<T>(-_x, -_y, -_z); }

	__always_inline tvec3<T> &operator-=(tvec3<T> const &i) { _x -= i._x; _y -= i._y; _z -= i._z; return *this; }
	__always_inline tvec3<T> &operator-=(	   T  const &i) { _x -= i;    _y -= i;    _z -= i;    return *this; }

	__always_inline tvec3<T> &operator+=(tvec3<T> const &i) { _x += i._x; _y += i._y; _z += i._z; return *this; }
	__always_inline tvec3<T> &operator+=(	   T  const &i) { _x += i;    _y += i;    _z += i;    return *this; }

	__always_inline tvec3<T> &operator*=(tvec3<T> const &i) { _x *= i._x; _y *= i._y; _z *= i._z; return *this; }
	__always_inline tvec3<T> &operator*=(	   T  const &i) { _x *= i;    _y *= i;    _z *= i;    return *this; }

	__always_inline tvec3<T> &operator/=(tvec3<T> const &i) { _x /= i._x; _y /= i._y; _z /= i._z; return *this; }
	__always_inline tvec3<T> &operator/=(	   T  const &i) { _x /= i;    _y /= i;    _z /= i;    return *this; }
};

template <class T> __always_inline tvec3<T> operator-(tvec3<T> const &a, tvec3<T> const &b) { return tvec3<T>(a._x - b._x, a._y - b._y, a._z - b._z); }
template <class T> __always_inline tvec3<T> operator-(tvec3<T> const &a,       T  const &b) { return tvec3<T>(a._x - b   , a._y - b   , a._z - b   ); }
template <class T> __always_inline tvec3<T> operator-(      T  const &b, tvec3<T> const &a) { return tvec3<T>(a._x - b   , a._y - b   , a._z - b   ); }

template <class T> __always_inline tvec3<T> operator+(tvec3<T> const &a, tvec3<T> const &b) { return tvec3<T>(a._x + b._x, a._y + b._y, a._z + b._z); }
template <class T> __always_inline tvec3<T> operator+(tvec3<T> const &a,       T  const &b) { return tvec3<T>(a._x + b   , a._y + b   , a._z + b   ); }
template <class T> __always_inline tvec3<T> operator+(      T  const &b, tvec3<T> const &a) { return tvec3<T>(a._x + b   , a._y + b   , a._z + b   ); }

template <class T> __always_inline tvec3<T> operator*(tvec3<T> const &a, tvec3<T> const &b) { return tvec3<T>(a._x * b._x, a._y * b._y, a._z * b._z); }
template <class T> __always_inline tvec3<T> operator*(tvec3<T> const &a,       T  const &b) { return tvec3<T>(a._x * b   , a._y * b   , a._z * b   ); }
template <class T> __always_inline tvec3<T> operator*(      T  const &b, tvec3<T> const &a) { return tvec3<T>(a._x * b   , a._y * b   , a._z * b   ); }

template <class T> __always_inline tvec3<T> operator/(tvec3<T> const &a, tvec3<T> const &b) { return tvec3<T>(a._x / b._x, a._y / b._y, a._z / b._z); }
template <class T> __always_inline tvec3<T> operator/(tvec3<T> const &a,       T  const &b) { return tvec3<T>(a._x / b   , a._y / b   , a._z / b   ); }
template <class T> __always_inline tvec3<T> operator/(      T  const &b, tvec3<T> const &a) { return tvec3<T>(a._x / b   , a._y / b   , a._z / b   ); }

template <class T> __always_inline T dot(tvec3<T> const &a, tvec3<T> const &b)
{
	return a._x * b._x + a._y * b._y + a._z * b._z;
}

template <class T> __always_inline tvec3<T> cross(tvec3<T> const &a, tvec3<T> const &b)
{
	return tvec3<T>(a._y * b._z - a._z * b._y, -(a._x * b._z - a._z * b._x), a._x * b._y - a._y * b._x);
}

template <class T> __always_inline T length(tvec3<T> const &a)
{
	return fast_sqrt(dot(a, a));
}

template <class T> __always_inline T rlength(tvec3<T> const &a)
{
	return rsqrt(dot(a, a));
}

template <class T> __always_inline T length2(tvec3<T> const &a)
{
	return dot(a, a);
}

template <class T> __always_inline tvec3<T> normalize(tvec3<T> const &a)
{
	return a * rlength(a);
}

template <class T> __always_inline tvec3<T> mix(tvec3<T> const &a, tvec3<T> const &b, T fac)
{
	return a + (b - a) * fac;
}

template <class T> __always_inline tvec3<T> reflect(tvec3<T> const &a, tvec3<T> const &b)
{
	return a - 2 * dot(a, b) * b;
}

template <class T> __always_inline tvec3<T> refract(tvec3<T> const &v, tvec3<T> const &n, T const &e)
{
	float c = vm::clamp(-1.f, 1.f, vm::dot(v, n));
	float k = 1.f - e * e * (1.f - c * c);
	return k < 0.f ? tvec3<T>(0.f) : e * v + (e * c - fast_sqrt(k)) * n;
}

template <class T> __always_inline tvec3<T> max(tvec3<T> const &a, tvec3<T> const &b)
{
	return tvec3<T>(max(a._x, b._x), max(a._y, b._y), max(a._z, b._z));
}

template <class T> __always_inline tvec3<T> max(tvec3<T> const &a, T const &b)
{
	return tvec3<T>(max(a._x, b), max(a._y, b), max(a._z, b));
}

template <class T> __always_inline tvec3<T> min(tvec3<T> const &a, tvec3<T> const &b)
{
	return tvec3<T>(min(a._x, b._x), min(a._y, b._y), min(a._z, b._z));
}

template <class T> __always_inline tvec3<T> min(tvec3<T> const &a, T const &b)
{
	return tvec3<T>(min(a._x, b), min(a._y, b), min(a._z, b));
}

template <class T> __always_inline tvec3<T> clamp(tvec3<T> const &i, tvec3<T> const &a, tvec3<T> const &b)
{
	return min(max(i, a), b);
}

template <class T> __always_inline tvec3<T> clamp(tvec3<T> const &i, T const &a, T const &b)
{
	return min(max(i, a), b);
}

template <class T> __always_inline tvec3<T> pow(tvec3<T> const &a, tvec3<T> const &b)
{
	return tvec3<T>(std::pow(a.x(), b.x()), std::pow(a.y(), b.y()), std::pow(a.z(), b.z()));
}

typedef tvec3<float>    fvec3;
typedef tvec3<double>   dvec3;
typedef tvec3<int32_t>  ivec3;
typedef tvec3<uint32_t> uvec3;

#if !(defined(__i386__) || defined(__x86_64__)) || defined(VM_DISABLE_SIMD)

#warning "not using SIMD vector types."
typedef tvec3<float_type> vec3;

#else

// assuming SSE is supported because __i386__ is defined
// vec3 will use 128bit simd mathematics

class vec3
{
public:
	__always_inline vec3() { d = _mm_set_ps(0.f, 0.f, 0.f, 0.f); }
	__always_inline vec3(vec3 const &i) { d = i.d; }
	__always_inline vec3(__m128 const &i) { d = i; }
	__always_inline vec3(vm::fvec3 const &i) { d = _mm_set_ps(i.z(), i.z(), i.y(), i.x()); }
	__always_inline vec3(float x, float y, float z) { d = _mm_set_ps(z, z, y, x); }

	__always_inline void store(float *i) const { _mm_store_ps(i, d); }

	__always_inline float x() const { return _mm_cvtss_f32(d); }
	__always_inline float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(d, d, _MM_SHUFFLE(1, 1, 1, 1))); }
	__always_inline float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(d, d, _MM_SHUFFLE(2, 2, 2, 2))); }

	__always_inline void x(float i) { d = _mm_move_ss(d, _mm_set_ss(i)); }
	__always_inline void y(float i) { __m128 t = _mm_move_ss(d, _mm_set_ss(i));  t = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 2, 0, 0)); d = _mm_move_ss(t, d); }
	__always_inline void z(float i) { __m128 t = _mm_move_ss(d, _mm_set_ss(i));  t = _mm_shuffle_ps(t, t, _MM_SHUFFLE(3, 0, 1, 0)); d = _mm_move_ss(t, d); }

	__always_inline void s(size_t a, float i) { switch (a) { case 0: x(i); return; break; case 1: y(i); return; break; case 2: z(i); return; break; default: return; } }

	__always_inline vm::fvec3 to_vec3() const { return vm::fvec3(x(), y(), z()); }

	__always_inline __m128 shuffle(size_t x, size_t y, size_t z) const { return _mm_shuffle_ps(d, d, _MM_SHUFFLE(z, z, y, x)); }

	__always_inline float operator[](size_t const &i) const { switch (i) { case 0: return x(); break; case 1: return y(); break; case 2: return z(); break; default: return z(); } }

	__always_inline vec3 &operator=(vec3 const &i) { d = i.d; return *this; }

	__always_inline vec3 operator-() const { return vec3(_mm_sub_ps(_mm_set1_ps(0.0), d)); }

	__always_inline vec3 &operator*=(vec3  const &i) { d = _mm_mul_ps(d, i.d); return *this; }
	__always_inline vec3 &operator*=(float const &i) { d = _mm_mul_ps(d, _mm_load_ps1(&i)); return *this; }

	__always_inline vec3 &operator/=(vec3  const &i) { d = _mm_div_ps(d, i.d); return *this; }
	__always_inline vec3 &operator/=(float const &i) { d = _mm_div_ps(d, _mm_load_ps1(&i)); return *this; }

	__always_inline vec3 &operator+=(vec3  const &i) { d = _mm_add_ps(d, i.d); return *this; }
	__always_inline vec3 &operator+=(float const &i) { d = _mm_add_ps(d, _mm_load_ps1(&i)); return *this; }

	__always_inline vec3 &operator-=(vec3  const &i) { d = _mm_sub_ps(d, i.d); return *this; }
	__always_inline vec3 &operator-=(float const &i) { d = _mm_sub_ps(d, _mm_load_ps1(&i)); return *this; }

	__m128 d;
};

__always_inline vec3 operator*(vec3  const &a,  vec3 const &b) { return vec3(_mm_mul_ps(a.d, b.d)); }
__always_inline vec3 operator*(float const &a,  vec3 const &b) { return vec3(_mm_mul_ps(b.d, _mm_load_ps1(&a))); }
__always_inline vec3 operator*(vec3  const &a, float const &b) { return b * a; }

__always_inline vec3 operator/(vec3  const &a,  vec3 const &b) { return vec3(_mm_div_ps(a.d, b.d)); }
__always_inline vec3 operator/(float const &a,  vec3 const &b) { return vec3(_mm_div_ps(_mm_load_ps1(&a), b.d)); }
__always_inline vec3 operator/(vec3  const &a, float const &b) { return vec3(_mm_div_ps(a.d, _mm_load_ps1(&b))); }

__always_inline vec3 operator+(vec3  const &a,  vec3 const &b) { return vec3(_mm_add_ps(a.d, b.d)); }
__always_inline vec3 operator+(float const &a,  vec3 const &b) { return vec3(_mm_add_ps(b.d, _mm_load_ps1(&a))); }
__always_inline vec3 operator+(vec3  const &a, float const &b) { return b + a; }

__always_inline vec3 operator-(vec3  const &a,  vec3 const &b) { return vec3(_mm_sub_ps(a.d, b.d)); }
__always_inline vec3 operator-(float const &a,  vec3 const &b) { return vec3(_mm_sub_ps(_mm_load_ps1(&a), b.d)); }
__always_inline vec3 operator-(vec3  const &a, float const &b) { return vec3(_mm_sub_ps(a.d, _mm_load_ps1(&b))); }

__always_inline vec3 cross(vec3 const &a, vec3 const &b)
{
	return vec3(_mm_sub_ps(_mm_mul_ps(_mm_shuffle_ps(a.d, a.d, _MM_SHUFFLE(3, 0, 2, 1)),
	                                  _mm_shuffle_ps(b.d, b.d, _MM_SHUFFLE(3, 1, 0, 2))),
	                       _mm_mul_ps(_mm_shuffle_ps(a.d, a.d, _MM_SHUFFLE(3, 1, 0, 2)),
	                                  _mm_shuffle_ps(b.d, b.d, _MM_SHUFFLE(3, 0, 2, 1)))));
}

__always_inline float length(vec3 const &a)
{
	return _mm_cvtss_f32(_mm_sqrt_ss(_mm_dp_ps(a.d, a.d, 0x71)));
}

__always_inline vec3 normalize(vec3 const &a)
{
	return vec3(_mm_mul_ps(a.d, _mm_rsqrt_ps(_mm_dp_ps(a.d, a.d, 0x77))));
}

__always_inline float dot(vec3 const &a, vec3 const &b)
{
	return _mm_cvtss_f32(_mm_dp_ps(a.d, b.d, 0x71));
}

__always_inline vec3 max(vec3 const &a, vec3 const &b)
{
	return vec3(_mm_max_ps(a.d, b.d));
}

__always_inline vec3 min(vec3 const &a, vec3 const &b)
{
	return vec3(_mm_min_ps(a.d, b.d));
}

//TODO: simd versions of everything below
__always_inline vec3 reflect(vec3 const &v, vec3 const &n)
{
	return v - 2.f * vm::dot(v, n) * n;
}

__always_inline vec3 refract(vec3 const &v, vec3 const &n, float e)
{
	float c = vm::clamp(-1.f, 1.f, vm::dot(v, n));
	float k = 1.f - e * e * (1.f - c * c);
	return k < 0.f ? vm::vec3(0.f) : e * v + (e * c - fast_sqrt(k)) * n;
}

__always_inline vec3 mix(vec3 const &v, vec3 const &n, float fac)
{
	return v * (1.f - fac) + n * fac;
}

__always_inline vec3 clamp(vec3 const &v, vec3 const &b, vec3 const&t)
{
	return vm::min(vm::max(v, b), t);
}

__always_inline vec3 pow(vec3 const &a, vec3 const &b)
{
	return vec3(std::pow(a.x(), b.x()), std::pow(a.y(), b.y()), std::pow(a.z(), b.z()));
}

#endif



// VEC4

// TODO: tvec4
template <class T> class tvec4
{
public:
	__always_inline tvec4() : x(0), y(0), z(0), w(0) {  }
	__always_inline tvec4(tvec4<T> const &i) : x(i.x), y(i.y), z(i.z), w(i.w) {  }
	__always_inline tvec4(T ix, T iy, T iz, T iw) : x(ix), y(iy), z(iz), w(iw) {  }
	__always_inline tvec4(T i) : x(i), y(i), z(i), w(i) {  }

	__always_inline tvec2<T> &operator=(tvec2<T> const &i) { x = i.x; y = i.y; z = i.z; w = i.w; return *this; }

	union { T x, r; };
	union { T y, g; };
	union { T z, b; };
	union { T w, a; };

};

typedef tvec4<float_type> vec4;


}

#endif
