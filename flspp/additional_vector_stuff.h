#pragma once
#include <vector>
#include <iterator>
#include "clustering.h"
#include "makros.h"

// overload comparison of vectors
// template<typename T>
// bool operator==(const std::vector<T>& lhs, const std::vector<T>& rhs);

// templates can not be compiled => They can not be included in a cpp file
// https://stackoverflow.com/questions/1639797/template-issue-causes-linker-error-c

template <typename T>
inline bool operator==(const std::vector<T> &lhs, const std::vector<T> &rhs)
{
	if (lhs.size() != rhs.size())
		return false;
	for (int i = 0; i < lhs.size(); i++)
	{
		if (lhs[i] != rhs[i])
			return false;
	}
	return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &vec)
{
	for (auto elem : vec)
	{
		os << elem << " ";
	}
	return os;
}

// printing vectors
// template <typename T>
// std::ostream& operator << (std::ostream& os, const std::vector<T>& vec);