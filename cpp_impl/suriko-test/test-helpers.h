#pragma once
#include <filesystem>

inline std::filesystem::path& TestDataRoot()
{
    static std::filesystem::path result;
    return result;
}

inline std::filesystem::path TestData(const std::filesystem::path& rel_path)
{
    return TestDataRoot() / rel_path;
}