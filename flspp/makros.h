#pragma once
#include <string>
#include <iostream>
// For Debugging Code set this following Keyword or comment it
// #define DEBUG

// small code to detect different operation systems
// #ifdef __unix__
//...
#if defined(_WIN32) || defined(_WIN64)

#define OS_Windows

#endif // __unix__

// For Printing stuff to stdout
#define DEBUG_PRINT

// void DEBUG_PRINTING(std::string out);
