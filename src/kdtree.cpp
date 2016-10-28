/**
 * @file      kdtree.cpp
 * @brief     intializer for 3-axis kd-tree
 * @authors   Michael Willett
 * @date      2016
 * @copyright Michael Willett
 */

#include <algorithm>
#include "kdtree.hpp"

bool sortX(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.x < p2.x;
}
bool sortY(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.y < p2.y;
}
bool sortZ(const glm::vec4 &p1, const glm::vec4 &p2)
{
	return p1.z < p2.z;
}

void KDTree::Create(std::vector<glm::vec4> input, Node *list)
{
	std::sort(input.begin(), input.end(), sortX);
	InsertList(input, list, 0, -1);
}


void KDTree::InsertList(std::vector<glm::vec4> input, Node *list, int idx, int parent)
{
	int axis = (parent == -1) ? 0 : (list[parent].axis + 1) % 3;
	if (axis == 0)
		std::sort(input.begin(), input.end(), sortX);
	if (axis == 1)
		std::sort(input.begin(), input.end(), sortY);
	if (axis == 2 )
		std::sort(input.begin(), input.end(), sortZ);

	// set current node
	int mid = (int) input.size() / 2;
	list[idx] = Node(input[mid], axis, parent);

	if (mid > 0) {
		list[idx].left = idx + 1;
		std::vector<glm::vec4> left(input.begin(), input.begin() + mid);
		InsertList(left, list, list[idx].left, idx);
	}

	if (mid < input.size() - 1) {
		list[idx].right = idx + mid + 1;
		std::vector<glm::vec4> right(input.begin() + mid + 1, input.end());
		InsertList(right, list, list[idx].right, idx);
	}
}


KDTree::Node::Node() {
	left = -1;
	right = -1;
	parent = -1;
	value = glm::vec4(0, 0, 0, 0);
	axis = 0;
}

KDTree::Node::Node(glm::vec4  p, int state, int source) {
	left = -1;
	right = -1;
	parent = source;
	value = p;
	axis = state;
}
