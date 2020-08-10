#include "vm.hh"
#include "prng.hh"

#include <thread>
#include <vector>
#include <functional>
#include <fstream>

// hack to remove "undefined reference to WinMain" on MINGW
#define SDL_main_h_
#include <SDL2/SDL.h>

#define USE_RAND_SPHERE_TABLE
#define USE_RAND_HEMISPHERE_TABLE

// increase the size of these tables for extremely high sample renders.
std::vector<vm::vec3> _rand_sphere_table(0x10000);
std::vector<vm::vec3> _rand_hemisphere_table(0x8000);

unsigned const _num_threads = std::thread::hardware_concurrency();

unsigned const _num_bounces = 4;
unsigned const _num_samples = 32;

unsigned const _width = 960;
unsigned const _height = 720;

vm::vec2 const _pixel_size(1.f / static_cast<float_type>(_width), 1.f / static_cast<float_type>(_height));

float_type const _eps = .001f;
float_type const _huge = 1000000000000.f;

vm::vec3 const _camera_pos(0.f, 1.5f, -5);
vm::vec3 const _camera_facing(0.f, 0.f, 1.f);

std::vector<uint32_t> _image(_width * _height);

void generate_rand_sphere_table()
{
#ifdef USE_RAND_SPHERE_TABLE
	for (auto &i : _rand_sphere_table)
	{
		float_type a = prng::itof(prng::next()) * _2pi, b = prng::itof(prng::next()) * _2pi;
		float_type s = vm::fast_sin(a);
		i = vm::vec3(s * vm::fast_sin(b), s * vm::fast_cos(b), vm::fast_cos(a));

		// TODO: test bias for the sampling technique below
		//i = vm::normalize(vm::vec3(prng::itof(prng::next()) - .5f, prng::itof(prng::next()) - .5f, prng::itof(prng::next()) - .5f));
	}
#endif
}

void generate_rand_hemisphere_table()
{
#ifdef USE_RAND_HEMISPHERE_TABLE
	for (auto &i : _rand_hemisphere_table)
	{
		vm::vec3 n(0.f, 1.f, 0.f);
		
		float r1 = 2.f * _pi * prng::itof(prng::next());
		float r2 = prng::itof(prng::next());
		float r2s = vm::fast_sqrt(r2);

		vm::vec3 w = n;
		vm::vec3 u;

		if (fabs(w.x()) > .1f) { u = vm::normalize(vm::cross(vm::vec3(0.f,1.f,0.f), w)); }
		else { u = vm::normalize(vm::cross(vm::vec3(1.f,0.f,0.f), w)); }

		vm::vec3 v = vm::cross(w, u);

		i = vm::normalize(u * cos(r1) * r2s + v* sin(r1) * r2s + w * sqrt(1 - r2));
	}
#endif
}

// random sphere direction
__always_inline static vm::vec3 rand_sphere()
{
#ifdef USE_RAND_SPHERE_TABLE
	return _rand_sphere_table[prng::next() % _rand_sphere_table.size()];
#else
	float_type a = prng::itof(prng::next()) * _2pi, b = prng::itof(prng::next()) * _2pi;
	float_type s = vm::fast_sin(a);
	return vm::vec3(s * vm::fast_sin(b), s * vm::fast_cos(b), vm::fast_cos(a));
#endif
}

// random hemisphere direction
__always_inline static vm::vec3 rand_hemi(vm::vec3 const &n)
{
#ifdef USE_RAND_HEMISPHERE_TABLE
	vm::vec3 r = _rand_hemisphere_table[prng::next() % _rand_hemisphere_table.size()];
	return vm::dot(r, n) * r;
#else
	// cosine weighted
	float r1 = 2.f * _pi * prng::itof(prng::next());
	float r2 = prng::itof(prng::next());
	float r2s = vm::fast_sqrt(r2);

	vm::vec3 w = n;
	vm::vec3 u;

	if (fabs(w.x()) > .1f) { u = vm::normalize(vm::cross(vm::vec3(0.f,1.f,0.f), w)); }
	else { u = vm::normalize(vm::cross(vm::vec3(1.f,0.f,0.f), w)); }

	vm::vec3 v = vm::cross(w, u);

	return vm::normalize(u * cos(r1) * r2s + v* sin(r1) * r2s + w * sqrt(1 - r2));
#endif
}

class ray
{
public:
	ray() : l(0.f), o(), d() { calc(); }
	ray(vm::vec3 const &origin, vm::vec3 const &direction) : l(0.f), o(origin), d(direction) { calc(); }
	ray &operator=(ray const &i) { o = i.o, d = i.d, r = i.r; return *this; }

	/** calculates the reciprocal direction vector.
	 * floating point division is very slow, so the reciprocal direction is calculated once and then used multiple times. */
	void calc() { r = 1.f / d; }

	float_type l; // length

	vm::vec3 o; // origin
	vm::vec3 d; // direction
	vm::vec3 r; // reciprocal direction todo: use
};

// material abstract
class mat
{
public:
	mat(vm::vec3 const &colour) : c(colour), e(false) {  }

	/** @brief calculates new ray direction from material surface
	 * @param n surface normal
	 * @param r the ray that intersected with the surface
	 * @param b enable bidirectional sample */
	virtual vm::vec3 brdf(vm::vec3 const &n, ray const &r) = 0;

	/** @brief ratio between sampling directions.
	 * 0.f = pure forward sampling
	 * 1.f = pure backward sampling */
	virtual float_type ratio() = 0;

	/** @brief colour sample. this will be changed when i add textures. */
	vm::vec3 sample() { return c; }

	bool e; // true for emission material, false for others.
	vm::vec3 c;
};

class mat_emission : public mat
{
public:
	mat_emission(vm::vec3 const &colour) : mat(colour) { e = true; }	
	vm::vec3 brdf(vm::vec3 const &n, ray const &r) { return vm::vec3(); }
	float_type ratio() { return 0.f; }
};

class mat_diffuse : public mat
{
public:
	mat_diffuse(vm::vec3 const &colour, float_type roughness) : mat(colour), d(roughness) {  }	

	vm::vec3 brdf(vm::vec3 const &n, ray const &r)
	{
		return vm::mix(vm::reflect(r.d, n), rand_hemi(n), d);
	}

	float_type ratio() { return 1.f; }

	float_type d;
};

class mat_glossy : public mat
{
public:
	mat_glossy(vm::vec3 const &colour, float_type roughness) : mat(colour), d(roughness) {  }	

	vm::vec3 brdf(vm::vec3 const &n, ray const &r)
	{
		return vm::mix(vm::reflect(r.d, n), rand_hemi(n), d);
	}

	float_type ratio() { /*return (d * .8f) + .1f;*/ return d; }

	float_type d;
};

class mat_glass : public mat
{
public:
	mat_glass(vm::vec3 const &colour, float_type ior) : mat(colour), i(ior) {  }	

	vm::vec3 brdf(vm::vec3 const &_n, ray const &r)
	{
		float_type i0, i1; // ior values
		float_type ndr = dot(r.d, _n);

		vm::vec3 n = _n;

		// assuming that the ior of air is 1.0

		if(ndr > 0.f) i0 = 1.f, i1 = i, n = -_n;
		else i0 = i, i1 = 1.f;

		float_type f0 = (i0 - i1) / (i0 + i1); f0 *= f0;
		float_type fresnel = f0 + (1.f - f0) * powf(1.f - std::abs(ndr), 5);

		if (prng::itof(prng::next()) < fresnel) return reflect(r.d, n);
		else return refract(r.d, n, i1 / i0);
	}

	float_type ratio() { return 0.f; }

	float_type i;
};

// object abstract
class obj
{
public:
	obj(mat &material, vm::vec3 const &position) : m(material), p(position) {  }

	/** @brief calculate surface normal.
	 * @param v position on the object surface.  */
	virtual vm::vec3 normal(vm::vec3 const &v) = 0;
	
	/** @brief tests if a ray intersects with the object. */
	virtual bool isect(ray &r) = 0;

	/** @brief tests if a ray intersects with the object and calculates normal if it does.
	 * @return tuple with: first = true if ray intersects object; second = the new ray origin; third = the normal of the object at the point where the intersection occurs. */
	virtual std::tuple<bool, vm::vec3, vm::vec3> isectn(ray &r) = 0;

	mat &m;
	vm::vec3 p;
};

class obj_sphere : public obj
{
public:
	obj_sphere(vm::vec3 const &position, float_type radius, mat &material) : r(radius), obj(material, position) {  }

	vm::vec3 normal(vm::vec3 const &v) { return (v - p) / r; }

	__always_inline std::tuple<bool, bool> is(ray &_r)
	{
		vm::vec3 l = _r.o - p;
		float_type a = vm::dot(_r.d, _r.d);
		float_type b = 2.f * vm::dot(l, _r.d);
		float_type c = vm::dot(l, l) - r * r;
		float_type di = b * b - 4.f * a * c;

		if (di < 0) return { false, false };

		float_type s = vm::fast_sqrt(di);
		float_type t0 = -b - s;
		bool inside = t0 < 0.f;
		_r.l = inside ? -b + s : t0;
		_r.l /= 2.f * a;
		return { inside, true };
	}

	bool isect(ray &_r)
	{
		return std::get<1>(is(_r));
	}

	std::tuple<bool, vm::vec3, vm::vec3> isectn(ray &_r)
	{
		auto res = is(_r);
		if (std::get<1>(res)) { vm::vec3 t = _r.o + _r.d * (_r.l - _eps); return { true, t, std::get<0>(res) ? -normal(t) : normal(t) }; };
		return { false, vm::vec3(), vm::vec3() };
	}

	float_type r;  // radius
};

class obj_plane : public obj
{
public:
	obj_plane(vm::vec3 const &normal, float_type length, mat &material) : l(length), obj(material, normal) {  }

	vm::vec3 normal(vm::vec3 const &v) { return p; }

	bool isect(ray &_r)
	{
		_r.l = (-l - vm::dot(p, _r.o)) / vm::dot(p, _r.d);

		return _r.l > _eps;
	}

	std::tuple<bool, vm::vec3, vm::vec3> isectn(ray &_r)
	{
		if (isect(_r)) { return { true, _r.o + _r.d * _r.l, p }; };
		return { false, vm::vec3(), vm::vec3() };
	}

	float_type l;
};

class obj_tri : public obj
{
public:
	obj_tri(vm::vec3 const &_0, vm::vec3 const &_1, vm::vec3 const &_2, mat &material) : obj(material, _0), p1(_1), p2(_2) {  }

	vm::vec3 normal(vm::vec3 const &v)
	{
		return n0; // TODO:
	}

	bool isect(ray &_r)
	{
		vm::vec3 t = _r.o - p0;
		vm::vec3 p = vm::cross(_r.d, p2 - p0);
		float d = 1.f / vm::dot(p1 - p0, p);

		float u = vm::dot(t, p) * d;
		if (u < 0.f || u > 1.f) return false;

		vm::vec3 q = vm::cross(t, p1 - p0);
		float v = vm::dot(_r.d, q) * d;

		if (v < 0.f || (u + v) > 1.f) return false;
		
		return true;
	}

	std::tuple<bool, vm::vec3, vm::vec3> isectn(ray &_r)
	{
		vm::vec3 t = _r.o - p0;
		vm::vec3 p = vm::cross(_r.d, p2 - p0);
		float d = 1.f / vm::dot(p1 - p0, p);

		float u = vm::dot(t, p) * d;
		if (u < 0.f || u > 1.f) return { false, vm::vec3(), vm::vec3() };

		vm::vec3 q = vm::cross(t, p1 - p0);
		float v = vm::dot(_r.d, q) * d;

		if (v < 0.f || (u + v) > 1.f) return { false, vm::vec3(), vm::vec3() };

		return { true, _r.o + _r.d * (vm::dot(p2 - p0, q) * d), vm::normalize((1.f - u - v) * n0 + u * n1 + v * n2) };
	}

	vm::vec3 &p0 = p; // i'm hoping this is optimized to nothing.
	vm::vec3  p1;
	vm::vec3  p2;

	vm::vec3 n0;
	vm::vec3 n1;
	vm::vec3 n2;
};
/*
class obj_mesh
{
public:
	obj_mesh()
};*/

std::vector<obj*> _objects;

/** @brief intersect entire scene
 * @returns tuple with first = iterator to either position of hit object or _objects.end(); second = new ray origin; third = surface normal. */
std::tuple<std::vector<obj*>::iterator, vm::vec3, vm::vec3> isectn(ray &r)
{
	r.l = _huge;
	std::vector<obj*>::iterator hit = _objects.end();
	vm::vec3 normal;
	vm::vec3 ro;

	for (auto i = _objects.begin(); i < _objects.end(); i++)
	{
		float_type l = r.l;
		auto tu = (*i)->isectn(r);
		if (std::get<0>(tu) && r.l < l && r.l > _eps) { hit = i; ro = std::get<1>(tu); normal = std::get<2>(tu); }
		else r.l = l;
	}

	return { hit, ro, normal };
}

vm::vec3 trace(ray &r)
{
	vm::vec3 acc = vm::vec3(1); // colour accumulator
	vm::vec3 col = vm::vec3(0); // colour (return value)

	float_type ratio = 0.f;

	vm::vec3 brdf;
	vm::vec3 normal = _camera_facing;

	for (unsigned i = 0; i < _num_bounces + 1; i++)
	{
		auto result = isectn(r);
		auto &it = std::get<0>(result);

		// if nothing is intersected, return ambient colour
		if (it == _objects.end()) { return col; }

		obj &ob = **it;
		vm::vec3 albedo = ob.m.sample();

		//return (std::get<2>(result) + 1.f) * .5f;

		// forward sample
		if (ob.m.e)
		{
			col /= (i + 1);

			if (ratio < .99f)
			{
				col += acc * albedo * (1.f - ratio);
			}
			
			return col;
		}

		normal = std::get<2>(result);
		brdf = ob.m.brdf(normal, r);

		ratio = ob.m.ratio();

		r.d = brdf;
		r.o = std::get<1>(result);

		acc *= albedo;

		// backward sample
		if (ratio > .01f)
		{
			for (auto const &o : _objects)
			{
				if (o->m.e)
				{
					obj_sphere &light = dynamic_cast<obj_sphere&>(*o);

					vm::vec3 rand = (rand_sphere() * light.r + light.p) - r.o;
					ray shadow_ray(r.o, vm::normalize(rand));
					shadow_ray.o += normal * _eps;
					shadow_ray.l = vm::length(rand) - _eps;
					
					vm::vec3 d = r.o - light.p;

					bool hit = false;

					float_type l = shadow_ray.l;

					for (auto &oo : _objects)
					{
						if (oo == o) continue;
						if (oo->isect(shadow_ray) && shadow_ray.l < l && shadow_ray.l > _eps) { hit = true; break; }
						shadow_ray.l = l;
					}

					if (!hit)
					{
						float_type c = vm::fast_sqrt(1.f - light.r * light.r / vm::dot(d, d));
						float_type w = 2.f * (1.f - c);

						float_type density = w * vm::clamp(vm::dot(shadow_ray.d, normal), 0.f, 1.f);

						col += ratio * acc * light.m.sample() * density;
					}
				}
			}
		}
	}

	return col / (_num_bounces + 1.f);
}

vm::vec3 sample(vm::vec2 const &coord)
{
	// convert pixel space to absolute space
	vm::vec2 pos = -1.f + 2.f * coord * _pixel_size;

	// correct aspect ratio
	pos.x *= static_cast<float_type>(_width) / static_cast<float_type>(_height);

	vm::vec3 cross = vm::normalize(vm::cross(_camera_facing, vm::vec3(0.f, 1.f, 0.f)));
	vm::vec3 up    = vm::normalize(vm::cross(cross, _camera_facing));

	vm::vec3 ret = vm::vec3(0.f);

	for (unsigned i = 0; i < _num_samples; i++)
	{
		vm::vec2 offset = (vm::vec2(prng::itof(prng::next()), prng::itof(prng::next())) - .5f) * _pixel_size * 3.f;

		ray r(_camera_pos, vm::normalize((pos.x + offset.x) * cross + (pos.y + offset.y) * up + 3.f * _camera_facing));

		ret += trace(r);
	}

	ret /= static_cast<float_type>(_num_samples);

	return vm::pow(vm::clamp(ret, vm::vec3(0.f), vm::vec3(1.f)), vm::vec3(.6f));
}

void render(unsigned x0, unsigned y0, unsigned x1, unsigned y1)
{
	for (unsigned i = x0; i < x1; i++)
	{
		for (unsigned j = y0 + 1; j <= y1; j++)
		{
			vm::vec3 s = sample(vm::vec2(i, j));

			uint8_t a = 0xFF;
			uint8_t r = s.x() * 0xFF;
			uint8_t g = s.y() * 0xFF;
			uint8_t b = s.z() * 0xFF;

			_image[((_height - j) * _width) + i] = (a << 24) | (r << 16) | (g << 8) | b;
		}
	}
}

int main(int argc, char *argv[])
{
	mat_emission em(vm::vec3(64.f, 60.f, 55.f));
	obj_sphere light(vm::vec3(-.3f, 1.5f, 2.f), .31f, em);
	_objects.push_back(&light);

	mat_emission em1(vm::vec3(30.f, 25.f, 20.f));
	obj_sphere light1(vm::vec3(.1f, .23f, .23f), .23f, em1);
	_objects.push_back(&light1);

	mat_diffuse dif_white(vm::vec3(.8f), .84f);
	mat_diffuse dif_red(vm::vec3(1.f, 0.f, 0.f), .84f);
	mat_diffuse dif_green(vm::vec3(0.f, 1.f, 0.f), .84f);

	mat_glossy glossy(vm::vec3(.8f), .3f);
	obj_sphere glossy_sphere(vm::vec3(.8f, .34f, 1.f), .34f, glossy);
	_objects.push_back(&glossy_sphere);

	mat_glass glass(vm::vec3(.8f), 1.5f);
	obj_sphere glass_sphere(vm::vec3(-.8f, .8f, 0.f), .4f, glass);
	_objects.push_back(&glass_sphere);

	mat_glossy glossy1(vm::vec3(.8f), .04f);
	obj_sphere glossy_sphere1(vm::vec3(.4f, 2.4f, 2.3f), .38f, glossy1);
	_objects.push_back(&glossy_sphere1);

	obj_sphere red_sphere(vm::vec3(-1.f, .2f, -0.3f), .2f, dif_red);
	_objects.push_back(&red_sphere);

	obj_sphere green_sphere(vm::vec3(.2f, .3f, 2.f), .3f, dif_green);
	_objects.push_back(&green_sphere);

	//mat_diffuse floor_mat(vm::vec3(.8f), .04f);
	obj_plane floor(vm::vec3(0.f, 1.f, 0.f), 0.f, dif_white);
	_objects.push_back(&floor);

	obj_plane ceil(vm::vec3(0.f, -1.f, 0.f), 3.f, dif_white);
	_objects.push_back(&ceil);

	obj_plane back_wall(vm::vec3(0.f, 0.f, -1.f), 3.f, dif_white);
	_objects.push_back(&back_wall);

	obj_plane right_wall(vm::vec3(1.f, 0.f, 0.f), 1.5f, dif_red);
	_objects.push_back(&right_wall);

	obj_plane left_wall(vm::vec3(-1.f, 0.f, 0.f), 1.5f, dif_green);
	_objects.push_back(&left_wall);

	std::thread sdl_thread([]()
	{
		SDL_Event event;
		SDL_Window *sdl_window;
		SDL_Renderer *sdl_renderer;

		SDL_Init(SDL_INIT_VIDEO);
		SDL_CreateWindowAndRenderer(_width, _height, 0, &sdl_window, &sdl_renderer);
		SDL_Texture *sdl_framebuffer = SDL_CreateTexture(sdl_renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, _width, _height);
		SDL_SetWindowTitle(sdl_window, "pt");

		bool _exit = false;

		while (!_exit)
		{
			while (SDL_PollEvent(&event)) if (event.type == SDL_QUIT) _exit = true;

			SDL_UpdateTexture(sdl_framebuffer, 0, _image.data(), _width * 4);
			SDL_SetRenderDrawColor(sdl_renderer, 0, 0, 0, 0);
			SDL_RenderClear(sdl_renderer);
			SDL_RenderCopy(sdl_renderer, sdl_framebuffer, 0, 0);
			SDL_RenderPresent(sdl_renderer);

			std::this_thread::sleep_for(std::chrono::milliseconds(33));
		}

		SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, _width, _height, 32, SDL_PIXELFORMAT_ARGB8888);
		SDL_RenderReadPixels(sdl_renderer, 0, SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch);
		auto t = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
		SDL_SaveBMP(surface, ("screenshots/" + std::to_string(t) + ".bmp").c_str());

		SDL_DestroyTexture(sdl_framebuffer);
		SDL_DestroyRenderer(sdl_renderer);
		SDL_DestroyWindow(sdl_window);
		SDL_Quit();
	});

	generate_rand_sphere_table();
	generate_rand_hemisphere_table();

	std::vector<std::thread> threads(_num_threads);

	unsigned x = 0;
	unsigned rpt = _width / _num_threads;

	auto _begin = std::chrono::high_resolution_clock::now();

	for (unsigned t = 0; t < _num_threads; t++)
	{
		unsigned x1 = t == _num_threads - 1 ? _width : x + rpt;
		threads[t] = std::thread(render, x, 0, x1, _height);
		x = x1;
	}

	for (auto &i : threads) { i.join(); }

	auto _end = std::chrono::high_resolution_clock::now();

	int64_t time = std::chrono::duration_cast<std::chrono::microseconds>(_end - _begin).count();
	double dtime = static_cast<double>(time) / 1000000.;

	std::printf("rendering complete. %.08llf seconds\n", dtime);

	// i wrote this whole bmp file tool and now i dont even need it.

	/*// write to a bitmap image
	std::ofstream bmp("im.bmp", std::ios::binary | std::ios::out);

	unsigned bmp_header_size = 0xE;
	unsigned bmp_info_size = 0x28;
	unsigned bmp_data_size = _width * _height * 3;

	// unions for binary conversions of integers to c-strings
	union strq { uint64_t u; char c[8]; };
	union strl { uint32_t u; char c[4]; };
	union strw { uint16_t u; char c[2]; };
	union strb { uint8_t  u; char c[1]; };

	// header
	bmp.write("BM"                                                           , 2); // bmp 2-byte signiture "BM"
	bmp.write(strl { .u = bmp_header_size + bmp_info_size + bmp_data_size }.c, 4); // file length in bytes
	bmp.write(strl { .u = 0                                               }.c, 4); // reserved
	bmp.write(strl { .u = bmp_header_size + bmp_info_size                 }.c, 4); // offset of image data in file

	// dib header (BITMAPINFOHEADER)
	bmp.write(strl { .u = bmp_info_size }.c, 4); // size of the dib header
	bmp.write(strl { .u = _width        }.c, 4); // image width
	bmp.write(strl { .u = _height       }.c, 4); // image height
	bmp.write(strw { .u = 1             }.c, 2); // number of colour planes
	bmp.write(strw { .u = 24            }.c, 2); // bits per pixel (24 for R8 G8 B8)
	bmp.write(strl { .u = 0             }.c, 4); // compression method (BI_RGB = no compression)
	bmp.write(strl { .u = 0             }.c, 4); // image size (0 can be used for uncompressed images)
	bmp.write(strl { .u = 2829          }.c, 4); // horizontal pixels per metre
	bmp.write(strl { .u = 2829          }.c, 4); // vertical pixels per metre
	bmp.write(strl { .u = 0             }.c, 4); // number of colours in the colour palette
	bmp.write(strl { .u = 0             }.c, 4); // number of important colours used, or 0

	for (unsigned i = 0; i < _width; i++)
	{
		for (unsigned j = 0; j < _height; j++)
		{
			uint32_t p = _image[(i * _height) + j];

			uint8_t a = (p >> 24) & 0xFF;
			uint8_t r = (p >> 16) & 0xFF;
			uint8_t g = (p >> 8) & 0xFF;
			uint8_t b = p & 0xFF;

			bmp.write(strb { .u = b }.c, 1); // B
			bmp.write(strb { .u = g }.c, 1); // G
			bmp.write(strb { .u = r }.c, 1); // R
		}
	}
	
	bmp.close();*/

	sdl_thread.join();

	return 0;
}
