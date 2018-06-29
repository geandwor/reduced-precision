#C code for matrix multiplication in float32 and bfloat16 - mmmul_libxsmm.c


#gcc compile the libxsmm program with #include libxsmm.h



gcc prac1.c -I $HOME/libxsmm/include -lpthread -o 1


#needs lapack and blas to run the libxsmm
#running environment:
#ubuntu 16.04,gcc 5.4.0
#install lapack, blas
#make the libxsmm as described at https://github.com/hfp/libxsmm
#cd libxsmm-folde
#make STATIC=0
#make

#gcc mmmul_libxsmm.c -I $HOME/libxsmm/include -o 1 -lpthread -L -llapack -lblas
