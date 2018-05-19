#pragma once
#include <gsl/span>
#include <Eigen/Dense>
#include <glog/logging.h>
#include "suriko/rt-config.h" // Scalar

namespace suriko
{
// Indices of columns/rows to be removed must be sorted in ascending order.
template <typename T>
void RemoveRowsAndColsInplace(gsl::span<const size_t> rows_to_remove, gsl::span<const size_t> cols_to_remove, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>* mat)
{
    for (ptrdiff_t i = 0; i < (ptrdiff_t)rows_to_remove.size() - 1; ++i)
        CHECK(rows_to_remove[i] < rows_to_remove[i + 1]) << "Require strictly ascending indices";
    for (ptrdiff_t i = 0; i < (ptrdiff_t)cols_to_remove.size() - 1; ++i)
        CHECK(cols_to_remove[i] < cols_to_remove[i + 1]) << "Require strictly ascending indices";

    // pad matrix with terminator row at the bottom and terminator column at the right
    // this will allow us to uniformally copy rows/columns after the last row/column to be removed

    size_t row_src = 0;
    size_t row_dst = 0;
    for (decltype(rows_to_remove)::index_type i = 0; i <= rows_to_remove.size(); ++i)
    {

        size_t rem_row_ind;
        if (i < rows_to_remove.size())
        {
            rem_row_ind = rows_to_remove[i];
            CHECK(rem_row_ind < (size_t)mat->rows()) << "row index: " << rem_row_ind << " is out of range of rows count: " << mat->rows();
        }
        else
            rem_row_ind = mat->rows(); // add terminator row

        size_t col_src = 0;
        size_t col_dst = 0;

        if (row_src < rem_row_ind)
        {
            size_t height = rem_row_ind - row_src;

            for (decltype(cols_to_remove)::index_type j = 0; j <= cols_to_remove.size(); ++j)
            {
                size_t rem_col_ind;
                if (j < cols_to_remove.size())
                {
                    rem_col_ind = cols_to_remove[j];
                    CHECK(rem_col_ind < (size_t)mat->cols()) << "column index: " << rem_col_ind << " is out of range of columns count: " << mat->cols();
                }
                else
                    rem_col_ind = mat->cols(); // add terminator column

                if (col_src < rem_col_ind)
                {
                    size_t width = rem_col_ind - col_src;
                    mat->block(row_dst, col_dst, height, width) = mat->block(row_src, col_src, height, width);

                    col_src += width;
                    col_dst += width;
                }
                col_src += 1; // removed column itself
            }
            SRK_ASSERT(col_src == (size_t)(mat->cols() + 1)) << "Not all columns are copied";

            // move to next horizontal line of blocks
            row_src += height;
            row_dst += height;
        }
        row_src += 1; // removed row itself
    }
    // +1 for terminator row
    SRK_ASSERT(row_src == (size_t)(mat->rows() + 1)) << "Not all rows are copied";

    size_t new_rows = mat->rows() - rows_to_remove.size();
    size_t new_cols = mat->cols() - cols_to_remove.size();
    
    // reset dimensions of empty matrix
    // matrix can't have dimension [0,3]; if one dimension is zero then the entire matrix is empty
    if (new_rows == 0) new_cols = 0;
    if (new_cols == 0) new_rows = 0;

    mat->conservativeResize(new_rows, new_cols); // resize means 'keep matrix elements intact'
}
}
