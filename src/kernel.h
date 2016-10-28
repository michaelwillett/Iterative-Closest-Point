#pragma once

#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <cmath>
#include <vector>

namespace ICP {
	void initSimulation(std::vector<glm::vec4> scene, std::vector<glm::vec4> target);
	void endSimulation();

	// Base ICP implementation obtained from http://ais.informatik.uni-freiburg.de/teaching/ss12/robotics/slides/17-icp.pdf
	void stepCPU();
	void stepGPU();
	void checkConvergence(int thresh);
    
	void copyPointsToVBO(float *vbodptr_positions, float *vbodptr_velocities);

    void unitTest();
}
