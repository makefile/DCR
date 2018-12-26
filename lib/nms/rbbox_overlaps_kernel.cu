//@deprecated('wrong result in some cases')
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


__device__ inline float trangle_area(float const * a, float const * b, float const * c) {
  return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))/2.0;
}

__device__ inline float area(float const * int_pts, int num_of_inter) {

  float area = 0.0;
  for(int i = 0;i < num_of_inter - 2;i++) {
    area += fabs(trangle_area(int_pts, int_pts + 2 * i + 2, int_pts + 2 * i + 4));
  }
  return area;
}

__device__ inline void reorder_pts(float * int_pts, int num_of_inter) {



  if(num_of_inter > 0) {

    float center[2];

    center[0] = 0.0;
    center[1] = 0.0;

    for(int i = 0;i < num_of_inter;i++) {
      center[0] += int_pts[2 * i];
      center[1] += int_pts[2 * i + 1];
    }
    center[0] /= num_of_inter;
    center[1] /= num_of_inter;

    float vs[16];
    float v[2];
    float d;
    for(int i = 0;i < num_of_inter;i++) {
      v[0] = int_pts[2 * i]-center[0];
      v[1] = int_pts[2 * i + 1]-center[1];
      d = sqrt(v[0] * v[0] + v[1] * v[1]);
      v[0] = v[0] / d;
      v[1] = v[1] / d;
      if(v[1] < 0) {
        v[0]= - 2 - v[0];
      }
      vs[i] = v[0];
    }

    float temp,tx,ty;
    int j;
    for(int i=1;i<num_of_inter;++i){
      if(vs[i-1]>vs[i]){
        temp = vs[i];
        tx = int_pts[2*i];
        ty = int_pts[2*i+1];
        j=i;
        while(j>0&&vs[j-1]>temp){
          vs[j] = vs[j-1];
          int_pts[j*2] = int_pts[j*2-2];
          int_pts[j*2+1] = int_pts[j*2-1];
          j--;
        }
        vs[j] = temp;
        int_pts[j*2] = tx;
        int_pts[j*2+1] = ty;
      }
    }
  }

}
__device__ inline bool inter2line(float const * pts1, float const *pts2, int i, int j, float * temp_pts) {

  float a[2];
  float b[2];
  float c[2];
  float d[2];

  float area_abc, area_abd, area_cda, area_cdb;

  a[0] = pts1[2 * i];
  a[1] = pts1[2 * i + 1];

  b[0] = pts1[2 * ((i + 1) % 4)];
  b[1] = pts1[2 * ((i + 1) % 4) + 1];

  c[0] = pts2[2 * j];
  c[1] = pts2[2 * j + 1];

  d[0] = pts2[2 * ((j + 1) % 4)];
  d[1] = pts2[2 * ((j + 1) % 4) + 1];

  area_abc = trangle_area(a, b, c);
  area_abd = trangle_area(a, b, d);

  if(area_abc * area_abd >= -1e-5) {
    return false;
  }

  area_cda = trangle_area(c, d, a);
  area_cdb = area_cda + area_abc - area_abd;

  if (area_cda * area_cdb >= -1e-5) {
    return false;
  }
  float t = area_cda / (area_abd - area_abc);

  float dx = t * (b[0] - a[0]);
  float dy = t * (b[1] - a[1]);
  temp_pts[0] = a[0] + dx;
  temp_pts[1] = a[1] + dy;

  return true;
}

// DO NOT USE THIS FUNCTION
__device__ inline bool in_rect_only_suitable_for_integer(float pt_x, float pt_y, float const * pts) {

  double ab[2];
  double ad[2];
  double ap[2];

  double abab;
  double abap;
  double adad;
  double adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];
  // BUG!! fyk: there has bug for coordinate such as judge (1,1) in [0.5, 0.5, 1.5, 0.5, 1.5, 1.5, 0.5, 1.5]
  // maybe the value -1 is bad
  bool result = (abab - abap >=  -1) and (abap >= -1) and (adad - adap >= -1) and (adap >= -1);
  return result;
}
__device__ inline bool in_rect_only_suitable_for_convex_quadrangle(float pt_x, float pt_y, float const * pts) {
    // https://blog.csdn.net/San_Junipero/article/details/79172260
    // https://blog.csdn.net/laukaka/article/details/45168439
    // float a = (B.x - A.x)*(y - A.y) - (B.y - A.y)*(x - A.x);
    // float b = (C.x - B.x)*(y - B.y) - (C.y - B.y)*(x - B.x);
    // float c = (D.x - C.x)*(y - C.y) - (D.y - C.y)*(x - C.x);
    // float d = (A.x - D.x)*(y - D.y) - (A.y - D.y)*(x - D.x);
//    float a = (pts[2] - pts[0])*(pt_y - pts[1]) - (pts[3] - pts[1])*(pt_x - pts[0]);
//    float b = (pts[4] - pts[2])*(pt_y - pts[3]) - (pts[5] - pts[3])*(pt_x - pts[2]);
//    float c = (pts[6] - pts[4])*(pt_y - pts[5]) - (pts[7] - pts[5])*(pt_x - pts[4]);
//    float d = (pts[0] - pts[6])*(pt_y - pts[7]) - (pts[1] - pts[7])*(pt_x - pts[6]);
//    if((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0)) {
//        return true;
//    }
//    return false;

// from https://stackoverflow.com/questions/2752725/finding-whether-a-point-lies-inside-a-rectangle-or-not
//# Corners in ax,ay,bx,by,dx,dy
//# Point in x, y
//
//bax = bx - ax
//bay = by - ay
//dax = dx - ax
//day = dy - ay
//
//if ((x - ax) * bax + (y - ay) * bay < 0.0) return false
//if ((x - bx) * bax + (y - by) * bay > 0.0) return false
//if ((x - ax) * dax + (y - ay) * day < 0.0) return false
//if ((x - dx) * dax + (y - dy) * day > 0.0) return false
//
//return true
    float bax = pts[2] - pts[0];
    float bay = pts[3] - pts[1];
    float dax = pts[6] - pts[0];
    float day = pts[7] - pts[1];

    if ((pt_x - pts[0]) * bax + (pt_y - pts[1]) * bay < 0.0) return false;
    if ((pt_x - pts[2]) * bax + (pt_y - pts[3]) * bay > 0.0) return false;
    if ((pt_x - pts[0]) * dax + (pt_y - pts[1]) * day < 0.0) return false;
    if ((pt_x - pts[6]) * dax + (pt_y - pts[7]) * day > 0.0) return false;

    return true;

}
__device__ inline float cross_mul(float *pt1, float * pt2, float *pt3){
    /* 叉积是判断多边形凹凸性以及点是否在凸多边形内部的利器。
    叉积 https://blog.csdn.net/Mikchy/article/details/81490908
    float cross_mul(const point_t &a, const point_t &b) {
        return a.x * b.y - a.y * b.x;
    }
    float cross_mul(const point_t &a, const point_t &b, const point_t &c) {
        return cross_mul(a - c, b - c);
    }*/
    return pt2[0]*pt3[1]+pt3[0]*pt1[1]+pt1[0]*pt2[1]-pt2[0]*pt1[1]-pt3[0]*pt2[1]-pt1[0]*pt3[1];
}
__device__ inline bool in_rect_convex(float pt_x, float pt_y, float * pts) {
// also only suitable for convex quadrangle, not for non-convex quadrangle
// https://blog.csdn.net/laukaka/article/details/45168439
  bool flag = true;
  int cur_sign;
  float pt[2];
  pt[0] = pt_x;
  pt[1] = pt_y;
  int sign;
  for(int i = 0 ;i<4;i++){
     float val = cross_mul(pts+i*2,pts+((i+1)%4*2),pt);
     if(val<0.0f){
        cur_sign = -1;
     }else if(val>0.0f){
        cur_sign = 1;
     }else{
        cur_sign =0;
     }
     if(cur_sign !=0){
        if(flag){
            flag = false;
            sign = cur_sign;
        }else{
            if(sign!=cur_sign) return false; // not same sign, then not in convex rect
        }
     }
  }
  return true;
}
/* maybe the best answer for check point in polygon
__device__ inline bool pointIsInPoly(Point p, Point polygon[], int n) {
    bool isInside = false;
    float minX = polygon[0].x, maxX = polygon[0].x;
    float minY = polygon[0].y, maxY = polygon[0].y;
    for (int i = 1; i < n; i++) {
        Point q = polygon[n];
        minX = Math.min(q.x, minX);
        maxX = Math.max(q.x, maxX);
        minY = Math.min(q.y, minY);
        maxY = Math.max(q.y, maxY);
    }
    if (p.x < minX || p.x > maxX || p.y < minY || p.y > maxY) {
        return false;
    }
    // the previous bounding box check can remove false-negatives in edge-cases
    // remove the previous check code to speed up if you don't care of edge-cases
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if ( (polygon[i].y > p.y) != (polygon[j].y > p.y) &&
                p.x < (polygon[j].x - polygon[i].x) * (p.y - polygon[i].y) / (polygon[j].y - polygon[i].y) + polygon[i].x ) {
            isInside = !isInside;
        }
    }

    return isInside;
} */

__device__ inline bool in_rect(float pt_x, float pt_y, float const * pts) {
    // https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon
    float minX = fmin(fmin(pts[0],pts[2]),fmin(pts[4],pts[6]));
    float maxX = fmax(fmax(pts[0],pts[2]),fmax(pts[4],pts[6]));
    float minY = fmin(fmin(pts[1],pts[3]),fmin(pts[5],pts[7]));
    float maxY = fmax(fmax(pts[1],pts[3]),fmax(pts[5],pts[7]));
    if (pt_x < minX || pt_x > maxX || pt_y < minY || pt_y > maxY) {
        return false;
    }
    bool isInside = false;
    // the previous bounding box check can remove false-negatives in edge-cases
    // remove the previous check code to speed up if you don't care of edge-cases
    int n = 4; // point num
    for (int i = 0, j = n - 1; i < n; j = i++) {
        float ix = pts[i * 2], iy = pts[i * 2 + 1];
        float jx = pts[j * 2], jy = pts[j * 2 + 1];
        if ( (iy > pt_y) != (jy > pt_y) &&
                pt_x < (jx - ix) * (pt_y - iy) / (jy - iy) + ix ) {
            isInside = !isInside;
        }
    }

    return isInside;
}

__device__ inline int inter_pts(float const * pts1, float const * pts2, float * int_pts) {

  int num_of_inter = 0;

  for(int i = 0;i < 4;i++) {
    if(in_rect(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
     if(in_rect(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];

  for(int i = 0;i < 4;i++) {
    for(int j = 0;j < 4;j++) {
      bool has_pts = inter2line(pts1, pts2, i, j, temp_pts);
      if(has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }


  return num_of_inter;
}

/*__device__ inline float inter(float const * const region1, float const * const region2) {

  float int_pts[16];
  int num_of_inter;

  num_of_inter = inter_pts(region1, region2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  return area(int_pts, num_of_inter);

}*/
/*
__device__ inline void convert_region(float * pts , float const * const region) {

  float angle = region[4];
  float a_cos = cos(angle/180.0*3.1415926535);
  float a_sin = -sin(angle/180.0*3.1415926535);// anti clock-wise

  float ctr_x = region[0];
  float ctr_y = region[1];
  float h = region[2];
  float w = region[3];



  float pts_x[4];
  float pts_y[4];

  pts_x[0] = - w / 2;
  pts_x[1] = - w / 2;
  pts_x[2] = w / 2;
  pts_x[3] = w / 2;

  pts_y[0] = - h / 2;
  pts_y[1] = h / 2;
  pts_y[2] = h / 2;
  pts_y[3] = - h / 2;

  for(int i = 0;i < 4;i++) {
    pts[2 * i] = a_cos * pts_x[i] - a_sin * pts_y[i] + ctr_x;
    pts[2 * i + 1] = a_sin * pts_x[i] + a_cos * pts_y[i] + ctr_y;

  }

}
*/

__device__ inline float devRotateIoU(float const * const region1, float const * const region2) {
  // enlarge to decrease the edge cases
  const float pts1[] = {region1[0] * 100, region1[1] * 100, region1[2] * 100, region1[3] * 100,
                         region1[4] * 100, region1[5] * 100, region1[6] * 100, region1[7] * 100};
  const float pts2[] = {region2[0] * 100, region2[1] * 100, region2[2] * 100, region2[3] * 100,
                         region2[4] * 100, region2[5] * 100, region2[6] * 100, region2[7] * 100};

  float area1 = area(pts1, 4);
  float area2 = area(pts2, 4);
  //float area_inter = inter(pts1, pts2);
  float int_pts[16];
  int num_of_inter;

  num_of_inter = inter_pts(pts1, pts2, int_pts);

  reorder_pts(int_pts, num_of_inter);

  float area_inter = area(int_pts, num_of_inter);

  float result = area_inter / (area1 + area2 - area_inter + 1e-5);

//  if(result < 0) {
//    result = 0.0;
//  }
  return result;

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


void _rotate_overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id) {

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

  // boxes_dim + 1 = 9
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

void _rotate_nms(int* keep_out, int* num_out, const float* boxes_host, int boxes_num,
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