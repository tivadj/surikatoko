#include <iostream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream> // std::ifstream
#include <sstream> // std::stringstream
#include <string.h> // strtok
#include "suriko/mat-serialization.h"

namespace suriko
{
bool ReadMatrixFromFile(const std::filesystem::path &file_path, char delimiter,
                        std::vector<Scalar> *data_by_row, size_t *rows, size_t *cols,
                        std::string *err_msg)
{
    std::ifstream fs(file_path.c_str());
    if (!fs)
    {
        if (err_msg != nullptr)
        {
            std::stringstream ss;
            ss << "Can't open file " << file_path;
            *err_msg = ss.str();
        }
        return false;
    }

    char all_delims[] = {delimiter, 0};
    std::string line;
    std::string num_str;

    size_t num_rows = 0;
    ptrdiff_t num_cols = -1;

    while (std::getline(fs, line))
    {
        char* cur_token = const_cast<char *>(line.c_str());

        int cur_line_num_cols = 0;

        // split line into tokens
        bool strtok_first_call = true;
        while(true)
        {
            char* in_token = strtok_first_call ? cur_token : static_cast<char*>(nullptr);
            strtok_first_call = false;

            char *nxt_token = strtok(in_token, all_delims);
            if (nxt_token == nullptr) break;

            num_str.assign(nxt_token);

            std::istringstream iss(num_str);
            Scalar num;
            iss >> num;

            if (!iss.eof()) // the whole number must be read
            {
                if (err_msg != nullptr) {
                    std::stringstream buf;
                    buf << "Can't parse number (" <<num_str << ") on line " << num_rows;
                    *err_msg = buf.str();
                }
                return false;
            }

            cur_line_num_cols += 1;
            data_by_row->push_back(num);
        }
        if (num_cols == -1)
            num_cols = cur_line_num_cols;
        else if (num_cols != cur_line_num_cols)
        {
            if (err_msg != nullptr) {
                std::stringstream buf;
                buf << "Data has inconsistent number of columns, row(0).columns=" <<num_cols
                    << ", row(" << num_rows << ").columns=" <<cur_line_num_cols;
                *err_msg = buf.str();
            }
            return false;
        }
        num_rows += 1;
    }
    *rows = num_rows;
    *cols = num_cols == -1 ? (size_t)0 : static_cast<size_t>(num_cols);
    return true;
}
}