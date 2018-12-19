
#include "gpu_nms_poly.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <stdio.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


#define maxn 510
#define eps 1E-5

typedef struct Point{
    float x,y;
}Point;

__device__ inline int sig(float d){
    return (d>eps)-(d<-eps);
}

__device__ inline int cmp(const Point* ps1, const Point* ps2) {
    return sig(ps1->x - ps2->x)==0 && sig(ps1->y - ps2->y)==0;
}
__device__ inline void swap(Point* ps1, Point* ps2) {
    Point tmp = *ps1;
    *ps1 = *ps2;
    *ps2 = tmp;
}
__device__ inline void reverse(Point*ps, int n) {
    //swap
    for(int i=n/2;i<n;i++) {
        Point tmp = ps[i];
        ps[i] = ps[n-i];
        ps[n-i] = tmp;
    }
}

__device__ inline float cross(Point o,Point a,Point b){  //叉积
    return (a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}

__device__ inline float area(Point* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(Point a,Point b,Point c,Point d,Point* p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p->x=(c.x*s2-d.x*s1)/(s2-s1);
    p->y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}
//多边形切割
//用直线ab切割多边形p，切割后的在向量(a,b)的左侧，并原地保存切割结果
//如果退化为一个点，也会返回去,此时n为1
__device__ inline void polygon_cut(Point*p,int* n_io,Point a,Point b){
    Point pp[maxn]; // 这里不能使用 static 局部变量, 结果会错误, 因为第二次执行时不会被初始化,
    // 另外多线程环境 static 局部变量可能会有问题.
    memset(pp,0,sizeof(Point)*maxn);
    int n = *n_io;
    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp+m++);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!cmp(pp + i,pp + i-1))
            p[n++]=pp[i];
    while(n>1&&cmp(p+n-1,p))n--;

    *n_io = n;
}
//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float triangleIntersectArea(Point a,Point b,Point c,Point d){
    Point o={0,0};
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    if(s1==-1) swap(&a, &b);
    if(s2==-1) swap(&c, &d);
    Point p[10]={o,a,b};
    int n=3;
    polygon_cut(p,&n,o,c);
    polygon_cut(p,&n,c,d);
    polygon_cut(p,&n,d,o);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;
    return res;
}

//求两多边形的交面积
__device__ inline float intersectArea(Point*ps1,int n1,Point*ps2,int n2){
    if(area(ps1,n1)<0) reverse(ps1,n1);
    if(area(ps2,n2)<0) reverse(ps2,n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=triangleIntersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assume res is positive!
}

__device__ inline float devRotateIoU(float const * const p, float const * const q) {
    Point ps1[maxn],ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = inter_area / union_area;
    //printf("gpu calc iou = %.2f score [%.3f vs %.3f] inter_area=%.2f, union_area=%.2f\n", iou, p[8], q[8], inter_area, union_area);
    return iou;
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}

__global__ void overlaps_kernel(const int N, const int K, const int boxes_dim, const float* dev_boxes,
                           const float * dev_query_boxes, float* dev_overlaps) {

  const int col_start = blockIdx.y;
  const int row_start = blockIdx.x;

  const int row_size =
        min(N - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(K - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 8];
  __shared__ float block_query_boxes[threadsPerBlock * 8];
  if (threadIdx.x < col_size) {
    for (int dim=0; dim<boxes_dim; dim++){
        block_query_boxes[threadIdx.x * boxes_dim + dim] =
            dev_query_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + dim];
    }
  }

  if (threadIdx.x < row_size) {
    for (int dim=0; dim<boxes_dim; dim++){
        block_boxes[threadIdx.x * boxes_dim + dim] =
            dev_boxes[(threadsPerBlock * row_start + threadIdx.x) * boxes_dim + dim];
    }
  }

  __syncthreads();

  if (threadIdx.x < row_size) {

    for(int i = 0;i < col_size; i++) {
      int offset = row_start*threadsPerBlock * K + col_start*threadsPerBlock + threadIdx.x*K+ i ;
      dev_overlaps[offset] = devRotateIoU(block_boxes + threadIdx.x * boxes_dim, block_query_boxes + i * boxes_dim);
    }

  }
}

void _poly_overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id) {

  _set_device(device_id);

  float* overlaps_dev = NULL;
  float* boxes_dev = NULL;
  float* query_boxes_dev = NULL;
  int boxes_dim = 8;

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        n * boxes_dim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes,
                        n * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&query_boxes_dev,
                        k * boxes_dim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(query_boxes_dev,
                        query_boxes,
                        k * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&overlaps_dev,
                        n * k * sizeof(float)));

  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(k, threadsPerBlock));

  dim3 threads(threadsPerBlock);

  overlaps_kernel<<<blocks, threads>>>(n, k, boxes_dim,
                                    boxes_dev,
                                    query_boxes_dev,
                                    overlaps_dev);

  CUDA_CHECK(cudaMemcpy(overlaps,
                        overlaps_dev,
                        n * k * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(overlaps_dev));
  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(query_boxes_dev));

}

__global__ void rotate_nms_kernel(const int n_boxes, const int boxes_dim, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // boxes_dim = 9
  __shared__ float block_boxes[threadsPerBlock * 9];
  if (threadIdx.x < col_size) {
    for (int dim=0; dim<boxes_dim; dim++){
        block_boxes[threadIdx.x * boxes_dim + dim] =
            dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * boxes_dim + dim];
    }
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * boxes_dim;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devRotateIoU(cur_box, block_boxes + i * boxes_dim) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _poly_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
          int boxes_dim, float nms_overlap_thresh, int device_id) {
  _set_device(device_id);

  float* boxes_dev = NULL;
  unsigned long long* mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&boxes_dev,
                        boxes_num * boxes_dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(boxes_dev,
                        boxes_host,
                        boxes_num * boxes_dim * sizeof(float),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  rotate_nms_kernel<<<blocks, threads>>>(boxes_num, boxes_dim,
                                  nms_overlap_thresh,
                                  boxes_dev,
                                  mask_dev);

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(boxes_dev));
  CUDA_CHECK(cudaFree(mask_dev));
}

