#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define grid 10
#define block 1024
bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    if (count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    int i;
    for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }
    cudaSetDevice(i);
    return true;
}

struct complex{
float r;
float i;
};

__device__ float mod(struct complex a){
return sqrt(a.r*a.r+a.i*a.i);
}

__device__ struct complex add(struct complex a,struct complex b){
struct complex c;
c.r=a.r+b.r;
c.i=a.i+b.i;
return c;
}
__device__ struct complex square(struct complex a){
struct complex b;
b.r=a.r*a.r-a.i*a.i;
b.i=2*a.r*a.i;
return b;
}
#define maxx 1000
__global__ void man(struct complex *p,int *o,int tol,int pixels)
{
int id=blockIdx.x*blockDim.x+threadIdx.x;
int i,j,k;
struct complex z,c;
	for(i=0;i*tol+id<pixels;i++){
		z.r=0;
		z.i=0;
		c=p[i*tol+id];
		k=maxx;
		for(j=1;j<maxx;j++){
			z=add(square(z),c);
			if((k==maxx)&&(mod(z)>2.0f)){
				k=j;
			}

		}
		o[i*tol+id]=k;
	}
}



#define height 2000
#define width 3000
#define xstart -2.0f
#define xend 1.0f
#define ystart -1.0f
#define yend 1.0f
int main()
{
    if (!InitCUDA()) {
        return 0;
    }
	struct complex* pic=(struct complex*)malloc(sizeof(struct complex)*width*height);
	int i,j;
	for(i=0;i<width;i++){
		for(j=0;j<height;j++){
			pic[i*height+j].r=xstart+(xend-xstart)*i/width;
			pic[i*height+j].i=ystart+(yend-ystart)*j/height;
		}
	}
	struct complex* picd;
	int *o=(int*)malloc(sizeof(int)*width*height); 
	int *od;
	cudaMalloc(&picd,sizeof(struct complex)*width*height);
	cudaMemcpy(picd,pic,sizeof(struct complex)*width*height,cudaMemcpyHostToDevice);
	
	cudaMalloc(&od,sizeof(int)*width*height);
	man<<<grid,block>>>(picd,od,grid*block,height*width);
	cudaMemcpy(o,od,sizeof(int)*width*height,cudaMemcpyDeviceToHost);
	FILE *fp=fopen("./out.ppm","w");
	fprintf(fp,"P3\n%d %d\n255\n",width,height);
	int r,g,b,col;
	for(i=0;i<height;i++){
		for(j=0;j<width;j++){
			col=o[j*height+i];
			if(col==maxx){
			r=0;
			g=0;
			b=0;
			}else{
			b=(int)max(min((0+(int)(512.0f/sqrt(o[j*height+i]))),255),0);
			g=0;
			r=0;
			}
			fprintf(fp,"%d %d %d ",r,g,b);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	
	free(pic);
	free(o);
	cudaFree(picd);
	cudaFree(od);
	cudaDeviceReset();
    return 0;
}
