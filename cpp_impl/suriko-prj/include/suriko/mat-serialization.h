#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include "suriko/rt-config.h"

namespace suriko
{
bool ReadMatrixFromFile(const boost::filesystem::path &file_path, char delimiter,
                        std::vector<Scalar> *data_by_row, size_t *rows, size_t *cols,
                        std::string *err_msg);
}