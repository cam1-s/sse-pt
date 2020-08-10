g++ -O3 -flto -s -ffast-math -fno-math-errno -march=native -mtune=native pt.cc -o pt -static-libgcc -static-libstdc++ -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic -lsdl2
