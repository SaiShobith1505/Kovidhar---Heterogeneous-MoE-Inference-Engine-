#ifndef MOE_API_COMMON_H
#define MOE_API_COMMON_H

#ifndef MOE_API
#ifdef _WIN32
#ifdef BUILDING_MOE_ENGINE
#define MOE_API __declspec(dllexport)
#else
#define MOE_API __declspec(dllimport)
#endif
#else
#define MOE_API
#endif
#endif

#endif // MOE_API_COMMON_H
