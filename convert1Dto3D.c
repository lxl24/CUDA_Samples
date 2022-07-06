#include <stdio.h>

void convert1DTo3D(const int sizeX, const int sizeXY, const int n, int *nx, int *ny, int *nz);

int main()
{
    int sizeX = 10;
    int sizeY = 20;
    int sizeZ = 30;
    int sizeXY = sizeX * sizeY;
    int n = 1234;
    int nx, ny, nz;
    convert1DTo3D(sizeX, sizeXY, n, &nx, &ny, &nz);
    printf("nx=%d, ny=%d, nz=%d\n", nx, ny, nz);
}

void convert1DTo3D(const int sizeX, const int sizeXY, const int n, int *nx, int *ny, int *nz)
{
    // your code
}