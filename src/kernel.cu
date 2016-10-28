#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "svd3.h"
#include "kdtree.hpp"
#include "kernel.h"
#include "device_launch_parameters.h"


// Controls for ICP implementation
#define KD_TREE_SEARCH 1
#define INITIAL_ROT 1


// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef clamp
#define clamp(x, lo, hi) (x < lo) ? lo : (x > hi) ? hi : x
#endif

#ifndef wrap
#define wrap(x, lo, hi) (x < lo) ? x + (hi - lo) : (x > hi) ? x - (hi - lo) : x
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 256
#define sharedMemorySize 65536

/*! Size of the starting area in simulation space. */
#define scene_scale 50.0f


/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int sizeTarget;
int sizeScene;
int numObjects;
dim3 threadsPerBlock(blockSize);

glm::vec4 *dev_pos;
glm::vec3 *dev_color;
int *dev_dist;
int *dev_pair;
KDTree::Node *dev_kd;

glm::vec4 *host_pos;
int *host_dist;
int *host_pair;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

__host__ __device__ bool sortFuncX(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.x < p2.x;
}
__host__ __device__ bool sortFuncY(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.y < p2.y;
}
__host__ __device__ bool sortFuncZ(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.z < p2.z;
}

__global__ void transformPoint(int N, glm::vec4 *points, glm::mat4 transform) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	points[index] = glm::vec4(glm::vec3(transform * glm::vec4(glm::vec3(points[index]), 1)), 1);
}

__global__ void kernResetVec3Buffer(int N, glm::vec3 *intBuffer, glm::vec3 value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}

__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		intBuffer[index] = value;
	}
}


/**
* Initialize memory, update some globals
*/
void ICP::initSimulation(std::vector<glm::vec4>	scene, std::vector<glm::vec4>	target) {
	sizeScene = scene.size();
	sizeTarget = target.size();
	numObjects = sizeScene + sizeTarget;

	cudaMalloc((void**)&dev_pos, numObjects * sizeof(glm::vec4));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_color, numObjects * sizeof(glm::vec4));
	checkCUDAErrorWithLine("cudaMalloc dev_color failed!");

	cudaMalloc((void**)&dev_dist, sizeTarget * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_dist failed!");

	cudaMalloc((void**)&dev_pair, sizeTarget * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_pair failed!");

	cudaMalloc((void**)&dev_kd, sizeScene * sizeof(KDTree::Node));
	checkCUDAErrorWithLine("cudaMalloc dev_kd failed!");
		
	int checksum = 0;
	for (int i = 0; i < sizeScene; i++)
		checksum += scene[i].w;

	KDTree::Node *kd = new KDTree::Node[sizeScene];
	KDTree::Create(scene, kd);

	cudaMemcpy(dev_kd, kd, sizeScene*sizeof(KDTree::Node), cudaMemcpyHostToDevice);

	int testsum = 0;
	for (int i = 0; i < sizeScene; i++) {
		testsum += kd[i].value.w;
	}
	printf("kd size: %i\n", sizeScene*sizeof(KDTree::Node));

	//verify all items are in the kd tree
	assert(checksum == testsum);
	
	// copy both scene and target to output points
	cudaMemcpy(dev_pos, &scene[0], scene.size()*sizeof(glm::vec4), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_pos[scene.size()], &target[0], target.size()*sizeof(glm::vec4), cudaMemcpyHostToDevice);

#if INITIAL_ROT
	//add rotation and translation to target for test;
	glm::vec3 t(80, -22, 100);
	glm::vec3 r(-.5, .6, .8);
	glm::vec3 s(1, 1, 1);
	glm::mat4 initial_rot = utilityCore::buildTransformationMatrix(t, r, s);
	transformPoint << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> >(target.size(), &dev_pos[scene.size()], initial_rot);
#endif

	//set colors for points
	kernResetVec3Buffer << <dim3((scene.size() + blockSize - 1) / blockSize), blockSize >> >(scene.size(), dev_color, glm::vec3(1, 1, 1));
	kernResetVec3Buffer << <dim3((target.size() + blockSize - 1) / blockSize), blockSize >> >(target.size(), &dev_color[scene.size()], glm::vec3(0, 1, 0));

	
	cudaThreadSynchronize();

	host_pos = (glm::vec4*) malloc(numObjects * sizeof(glm::vec4));
	host_pair = (int*)malloc(target.size() * sizeof(int));

	cudaMemcpy(host_pos, dev_pos, numObjects * sizeof(glm::vec4), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
}

void ICP::endSimulation() {
	cudaFree(dev_pos);
	cudaFree(dev_color);
	cudaFree(dev_dist);
	cudaFree(dev_pair);
	cudaFree(dev_kd);

	free(host_pos);
	free(host_pair);
}

/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec4 *pos, float *vbo, float s_scale, int start) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x) + start;

	float c_scale = -1.0f / s_scale;

	if (index - start < N) {
		vbo[4 * index + 0] = pos[index].x * c_scale;
		vbo[4 * index + 1] = pos[index].y * c_scale;
		vbo[4 * index + 2] = pos[index].z * c_scale;
		vbo[4 * index + 3] = 1.0f;
	}
}

__global__ void kernCopyColorsToVBO(int N, glm::vec3 *color, float *vbo, float s_scale, int start) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x) + start;

	if (index - start < N) {
		vbo[4 * index + 0] = color[index].x + 0.3f;
		vbo[4 * index + 1] = color[index].y + 0.3f;
		vbo[4 * index + 2] = color[index].z + 0.3f;
		vbo[4 * index + 3] = 1.0f;
	}
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void ICP::copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {

  //batch copies to prevent memory errors
  int batchSize = 1 << 16;
  for (int i = 0; i <= numObjects; i += batchSize) {
	  int n = imin(batchSize, numObjects - i);
	  dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

	  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(n, dev_pos, vbodptr_positions, scene_scale, i);
	  kernCopyColorsToVBO << <fullBlocksPerGrid, blockSize >> >(n, dev_color, vbodptr_velocities, scene_scale, i);
  }

  checkCUDAErrorWithLine("copyBoidsToVBO color failed!");

  cudaThreadSynchronize();
}


/******************
* stepSimulation *
******************/

__global__ void findCorrespondence(int N, int sizeScene, glm::vec4 *cor, const glm::vec4 *points)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i >= N) {
		return;
	}

	glm::vec4 pt = points[i + sizeScene];
	float best = glm::distance(glm::vec3(points[0]), glm::vec3(pt));
	cor[i] = points[0];

	for (int j = 1; j < sizeScene; j++) {
		float d = glm::distance(glm::vec3(points[j]), glm::vec3(pt));

		if (d < best) {
			cor[i] = points[j];
			best = d;
		}
	}
}


__device__ float getHyperplaneDist(const glm::vec4 *pt1, const glm::vec4 *pt2, int axis, bool *branch)
{
	if (axis == 0) {
		*branch = sortFuncX(*pt1, *pt2);
		return abs(pt1->x - pt2->x);
	}
	if (axis == 1) {
		*branch = sortFuncY(*pt1, *pt2);
		return abs(pt1->y - pt2->y);
	}
	if (axis == 2) {
		*branch = sortFuncZ(*pt1, *pt2);
		return abs(pt1->z - pt2->z);
	}
}

__global__ void findCorrespondenceKD(int N, int sizeScene, glm::vec4 *cor, const glm::vec4 *points, const KDTree::Node* tree)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i >= N) {
		return;
	}

	glm::vec4 pt = points[i + sizeScene];
	float bestDist = glm::distance(glm::vec3(pt), glm::vec3(tree[0].value));
	int bestIdx = 0;
	int head = 0;
	bool done = false;
	bool branch = false;
	bool nodeFullyExplored = false;

	while (!done) {
		// depth first on current branch
		while (head >= 0) {
			// check the current node
			const KDTree::Node test = tree[head];
			float d = glm::distance(glm::vec3(pt), glm::vec3(test.value));
			if (d < bestDist) {
				bestDist = d;
				bestIdx = head;
				nodeFullyExplored = false;
			}

			// find branch path
			getHyperplaneDist(&pt, &test.value, test.axis, &branch);
			head = branch ? test.left : test.right;
		}

		if (nodeFullyExplored) {
			done = true;
		}
		else {
			// check if parent of best node could have better values on other branch
			const KDTree::Node parent = tree[tree[bestIdx].parent];
			if (getHyperplaneDist(&pt, &parent.value, parent.axis, &branch) < bestDist) {
				head = !branch ? parent.left : parent.right;
				nodeFullyExplored = true;
			}
			else
				done = true;
		}
	}
	
	cor[i] = tree[bestIdx].value;
}

__global__ void outerProduct(int N, const glm::vec4 *vec1, const glm::vec4 *vec2, glm::mat3 *out)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	if (i >= N) {
		return;
	}

	out[i] = glm::mat3(	glm::vec3(vec1[i]) * vec2[i].x, 
						glm::vec3(vec1[i]) * vec2[i].y, 
						glm::vec3(vec1[i] * vec2[i].z));
}


/**
* Step the ICP algorithm.
*/
void ICP::stepCPU() {
	// find the closest point in the scene for each point in the target
	for (int i = 0; i < sizeTarget; i++) {
		float best = glm::distance(glm::vec3(host_pos[0]), glm::vec3(host_pos[i + sizeScene]));
		host_pair[i] = 0;

		for (int j = 1; j < sizeScene; j++) {
			float d = glm::distance(glm::vec3(host_pos[j]), glm::vec3(host_pos[i + sizeScene]));

			if (d < best) {
				host_pair[i] = j;
				best = d;
			}
		}
	}
	

	// Calculate mean centered correspondenses 
	glm::vec3 mu_tar(0, 0, 0), mu_cor(0, 0, 0);
	std::vector<glm::vec3> tar_c; 
	std::vector<glm::vec3> cor_c;

	for (int i = 0; i < sizeTarget; i++) {
		mu_tar += glm::vec3(host_pos[i + sizeScene]);
		mu_cor += glm::vec3(host_pos[host_pair[i]]);
	}
	mu_tar /= sizeTarget;
	mu_cor /= sizeTarget;

	for (int i = 0; i < sizeTarget; i++) {
		tar_c.push_back(glm::vec3(host_pos[i + sizeScene]) - mu_tar);
		cor_c.push_back(glm::vec3(host_pos[host_pair[i]]) - mu_cor);
	}


	// Calculate W
	float W[3][3] = {0};

	for (int i = 0; i < sizeTarget; i++) {
		W[0][0] += tar_c[i].x * cor_c[i].x;
		W[0][1] += tar_c[i].y * cor_c[i].x;
		W[0][2] += tar_c[i].z * cor_c[i].x;
		W[1][0] += tar_c[i].x * cor_c[i].y;
		W[1][1] += tar_c[i].y * cor_c[i].y;
		W[1][2] += tar_c[i].z * cor_c[i].y;
		W[2][0] += tar_c[i].x * cor_c[i].z;
		W[2][1] += tar_c[i].y * cor_c[i].z;
		W[2][2] += tar_c[i].z * cor_c[i].z;
	}

	// calculate SVD of W
	float U[3][3] = { 0 };
	float S[3][3] = { 0 };
	float V[3][3] = { 0 };

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
		);

	
	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R =g_U * g_Vt;
	glm::vec3 t = mu_cor - R*mu_tar;	

	// update target points
	for (int i = 0; i < sizeTarget; i++) {
		host_pos[i + sizeScene] = glm::vec4(R*glm::vec3(host_pos[i + sizeScene]) + t, host_pos[i + sizeScene].w);
	}

	cudaMemcpy(&dev_pos[sizeScene], &host_pos[sizeScene], sizeTarget * sizeof(glm::vec4), cudaMemcpyHostToDevice);
}

/**
* Step the ICP algorithm.
*/
void ICP::stepGPU() {
	dim3 fullBlocksPerGrid((sizeTarget + blockSize - 1) / blockSize);
	// find the closest point in the scene for each point in the target

	glm::vec4 *dev_cor, *tar_c, *cor_c;
	glm::mat3 *dev_W;
	cudaMalloc((void**)&dev_cor, sizeTarget*sizeof(glm::vec4));
	cudaMalloc((void**)&tar_c, sizeTarget*sizeof(glm::vec4));
	cudaMalloc((void**)&cor_c, sizeTarget*sizeof(glm::vec4));
	cudaMalloc((void**)&dev_W, sizeTarget * sizeof(glm::mat3));
	cudaMemset(dev_W, 0, sizeTarget * sizeof(glm::mat3));

#if KD_TREE_SEARCH
	findCorrespondenceKD << <fullBlocksPerGrid, blockSize >> >(sizeTarget, sizeScene, dev_cor, dev_pos, dev_kd);
#else
	findCorrespondence << <fullBlocksPerGrid, blockSize >> >(sizeTarget, sizeScene, dev_cor, dev_pos);
#endif
	cudaThreadSynchronize();

	// Calculate mean centered correspondenses 
	glm::vec3 mu_tar(0, 0, 0), mu_cor(0, 0, 0);


	thrust::device_ptr<glm::vec4> ptr_target(&dev_pos[sizeScene]);
	thrust::device_ptr<glm::vec4> ptr_scene(dev_pos);
	thrust::device_ptr<glm::vec4> ptr_cor(dev_cor);

	mu_tar = glm::vec3(thrust::reduce(ptr_target, ptr_target + sizeTarget, glm::vec4(0, 0, 0, 0)));
	mu_cor = glm::vec3(thrust::reduce(ptr_cor, ptr_cor + sizeTarget, glm::vec4(0, 0, 0, 0)));

	mu_tar /= sizeTarget;
	mu_cor /= sizeTarget;

	cudaMemcpy(tar_c, &dev_pos[sizeScene], sizeTarget*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);
	cudaMemcpy(cor_c, dev_cor, sizeTarget*sizeof(glm::vec4), cudaMemcpyDeviceToDevice);

	glm::vec3 r(0, 0, 0);
	glm::vec3 s(1, 1, 1);
	glm::mat4 center_tar = utilityCore::buildTransformationMatrix(-mu_tar, r, s);
	glm::mat4 center_cor = utilityCore::buildTransformationMatrix(-mu_cor, r, s);

	transformPoint << <fullBlocksPerGrid, blockSize >> >(sizeTarget, tar_c, center_tar);
	transformPoint << <fullBlocksPerGrid, blockSize >> >(sizeTarget, cor_c, center_cor);

	checkCUDAErrorWithLine("mean centered transformation failed!");
	cudaThreadSynchronize();

	// Calculate W

	outerProduct << <fullBlocksPerGrid, blockSize >> >(sizeTarget, tar_c, cor_c, dev_W);
	thrust::device_ptr<glm::mat3> ptr_W(dev_W);
	glm::mat3 W = thrust::reduce(ptr_W, ptr_W + sizeTarget, glm::mat3(0));

	checkCUDAErrorWithLine("outer product failed!");
	cudaThreadSynchronize();

	// calculate SVD of W
	glm::mat3 U, S, V;

	svd(W[0][0], W[0][1], W[0][2], W[1][0], W[1][1], W[1][2], W[2][0], W[2][1], W[2][2],
		U[0][0], U[0][1], U[0][2], U[1][0], U[1][1], U[1][2], U[2][0], U[2][1], U[2][2],
		S[0][0], S[0][1], S[0][2], S[1][0], S[1][1], S[1][2], S[2][0], S[2][1], S[2][2],
		V[0][0], V[0][1], V[0][2], V[1][0], V[1][1], V[1][2], V[2][0], V[2][1], V[2][2]
		);


	glm::mat3 g_U(glm::vec3(U[0][0], U[1][0], U[2][0]), glm::vec3(U[0][1], U[1][1], U[2][1]), glm::vec3(U[0][2], U[1][2], U[2][2]));
	glm::mat3 g_Vt(glm::vec3(V[0][0], V[0][1], V[0][2]), glm::vec3(V[1][0], V[1][1], V[1][2]), glm::vec3(V[2][0], V[2][1], V[2][2]));

	// Get transformation from SVD
	glm::mat3 R = g_U * g_Vt;
	glm::vec3 t = glm::vec3(mu_cor) - R*glm::vec3(mu_tar);

	// update target points
	glm::mat4 transform = glm::translate(glm::mat4(), t) * glm::mat4(R);
	transformPoint << <fullBlocksPerGrid, blockSize >> >(sizeTarget, &dev_pos[sizeScene], transform);

	cudaFree(dev_cor);
	cudaFree(tar_c);
	cudaFree(cor_c);
	cudaFree(dev_W);
}


void ICP::checkConvergence(int thresh) {
	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
}


void ICP::unitTest() {

	std::vector<glm::vec4> test;
	for (int i = 0; i < 16; i++) 
		test.push_back(glm::vec4((6 - 2 * i) % 3, -i % 4, i, i));
	
	KDTree::Node *nd = new KDTree::Node[16];
	KDTree::Create(test, nd);

	printf("nodes: \n");
	for (int i = 0; i < 16; i++)
		printf("  %i: parent(%i), axis(%i), children(%i %i), val (%f %f %f)\n", i,
			nd[i].parent, nd[i].axis, nd[i].left, nd[i].right, nd[i].value.x, nd[i].value.y, nd[i].value.z);


	glm::vec4 a(.1, .2, .3, 1234);
	glm::vec3 b = glm::vec3(a);

	//printf("\nb: %f %f %f (%f)\n", b.x, b.y, b.z, a.w);
}