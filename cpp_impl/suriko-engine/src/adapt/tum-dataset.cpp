#include <array>
#include <tuple>
#include <vector>
#include <optional>
#include <fstream>
#include <string_view>
#include <gsl/pointers> // gsl::not_null
#include "suriko/adapt/tum-dataset.h"
#include "suriko/rt-config.h"
#include "suriko/quat.h"
#include "suriko/stat-helpers.h"
#include "suriko/approx-alg.h"

namespace suriko::adapt::tum
{
using namespace std::literals;
constexpr char TumCommentChar = '#';

bool ReadTumDatasetTimedRgb(const std::filesystem::path& rgb_file_path,
    std::vector<TumTimestampFilename>* infos, std::string* err_msg)
{
    std::ifstream fs(rgb_file_path.c_str());
    if (!fs)
    {
        if (err_msg != nullptr)
        {
            std::stringstream ss;
            ss << "Can't open file: " << rgb_file_path;
            *err_msg = ss.str();
        }
        return false;
    }
    std::string line;
    std::string num_str;

    // read line by line
    for (size_t line_ind = 0; std::getline(fs, line); ++line_ind)
    {
        if (line.size() >= 1 && line[0] == TumCommentChar) continue;

        // read line, format:
        // timestamp filename

        std::istringstream iss(line);
        TumTimestamp stamp;
        std::string filename;
        iss >> stamp >> filename;

        if (!iss.eof()) // the whole number must be read
        {
            if (err_msg != nullptr) {
                std::stringstream buf;
                buf << "Can't parse line: " << line_ind;
                *err_msg = buf.str();
            }
            return false;
        }

        TumTimestampFilename r;
        r.timestamp = stamp;
        r.filename = std::move(filename);
        infos->push_back(r);
    }
    return true;
}

auto TimestampPoseToSE3(const TumTimestampPose& r)->std::optional<SE3Transform>
{
    if (!r.pos.has_value()) return std::nullopt;
    SE3Transform se3;
    se3.T = r.pos.value();
    auto q = gsl::make_span(r.quat->data(), 4);
    RotMatFromQuat(q, &se3.R);
    return se3;
}

auto SE3ToTimestampPose(TumTimestamp stamp, std::optional<SE3Transform> rt_opt)->TumTimestampPose
{
    TumTimestampPose result;
    result.timestamp = stamp;
    if (rt_opt.has_value())
    {
        const auto& rt = rt_opt.value();
        result.pos = rt.T;

        Eigen::Matrix<Scalar, 4, 1 > q;
        QuatFromRotationMatNoRChecks(rt.R, gsl::make_span<Scalar>(q.data(), 4));
        result.quat = q;
    }
    return result;
}

bool ReadTumDatasetGroundTruth(const std::filesystem::path& file_path,
    std::vector<TumTimestampPose>* poses_gt, std::string* err_msg, bool check_timestamp_order_asc)
{
    std::ifstream fs(file_path.c_str());
    if (!fs)
    {
        if (err_msg != nullptr)
        {
            std::stringstream ss;
            ss << "Can't open file: " << file_path;
            *err_msg = ss.str();
        }
        return false;
    }
    char all_delims[] = { ' ', 0 };
    std::string line;
    std::string num_str;
    std::vector<std::optional<Scalar>> nums_per_line;

    // read line by line
    for (size_t line_ind = 0; std::getline(fs, line); ++line_ind)
    {
        if (line.size() >= 1 && line[0] == TumCommentChar) continue;

        bool got_err = false;
        char* in_token = line.data();
        auto read_scalar = [line_ind, &num_str, &err_msg, &got_err](char* token) -> std::optional<Scalar>
        {
            if (got_err) return std::nullopt;

            num_str.assign(token);
            if (num_str == "NaN"sv)  // groundtruth is unavailable for this frame
                return std::nullopt;

            std::istringstream iss(num_str);
            Scalar num;
            iss >> num;

            if (!iss.eof()) // the whole number must be read
            {
                got_err = true;
                if (err_msg != nullptr) {
                    std::stringstream buf;
                    buf << "Can't parse number (" << num_str << ") on line: " << line_ind;
                    *err_msg = buf.str();
                }
                return std::nullopt;
            }
            return num;
        };

        // read line, format:
        // timestamp tx ty tz qx qy qz qw
        // timestamp NaN NaN NaN NaN NaN NaN NaN
        nums_per_line.clear();
        for (int i = 0; i < 8; i++)  // 8=number of digits per line
        {
            char* nxt_token = strtok(i == 0 ? in_token : nullptr, all_delims);
            std::optional<Scalar> num = read_scalar(nxt_token);
            nums_per_line.push_back(num);
        }
        if (got_err) return false;
        if (nums_per_line.empty() || !nums_per_line[0].has_value())
        {
            if (err_msg != nullptr) {
                std::stringstream buf;
                buf << "No timestamp on line: " << line_ind;
                *err_msg = buf.str();
            }
            return false;
        }

        TumTimestamp stamp = nums_per_line[0].value();

        if (check_timestamp_order_asc && !poses_gt->empty())
        {
            // there may be records with the same timestamp but different poses
            // for simplicity, use only the first occurence, ignore the remaining records
            if (poses_gt->back().timestamp == stamp)
                continue;

            if (poses_gt->back().timestamp > stamp)
            {
                if (err_msg != nullptr) {
                    std::stringstream buf;
                    buf << "Broken ascending order of timestamps on line: " << line_ind;
                    *err_msg = buf.str();
                }
                return false;
            }
        }
        TumTimestampPose pose;
        pose.timestamp = stamp;

        if (nums_per_line[1].has_value() &&
            nums_per_line[2].has_value() &&
            nums_per_line[3].has_value())
            pose.pos = suriko::Point3{ nums_per_line[1].value(),nums_per_line[2].value(),nums_per_line[3].value() };

        if (nums_per_line[4].has_value() &&
            nums_per_line[5].has_value() &&
            nums_per_line[6].has_value() &&
            nums_per_line[7].has_value())
        {
            // input: [qx qy qz qw]
            pose.quat = NewQuat(
                nums_per_line[4].value(),
                nums_per_line[5].value(),
                nums_per_line[6].value(),
                nums_per_line[7].value());
        }
        poses_gt->push_back(pose);
    }
    return true;
}

bool SaveTumDatasetGroundTruth(const std::filesystem::path& file_path,
    std::vector<TumTimestampPose>* poses_gt, QuatLayout quat_layout, std::string_view header, std::string* err_msg)
{
    std::ofstream fs(file_path.c_str());
    if (!fs)
    {
        if (err_msg != nullptr)
        {
            std::stringstream ss;
            ss << "Can't open file: " << file_path;
            *err_msg = ss.str();
        }
        return false;
    }
    if (!header.empty())
        fs << "# " << header << std::endl;
    if (quat_layout == QuatLayout::XyzW)
        fs << "# timestamp tx ty tz qx qy qz qw" << std::endl;
    else
        fs << "# timestamp tx ty tz qw qx qy qz" << std::endl;

    for (size_t i = 0; i < poses_gt->size(); ++i)
    {
        const TumTimestampPose& pose = (*poses_gt)[i];
        fs.precision(6);
        fs <<std::fixed << pose.timestamp ;

        if (pose.pos.has_value() && pose.quat.has_value())
        {
            const auto& p = pose.pos.value();
            const auto& q = pose.quat.value();
            fs.precision(4);
            fs << " " << p[0] << " " << p[1] << " " << p[2] << " ";
            if (quat_layout == QuatLayout::XyzW)
                fs << q[1] << " " << q[2] << " " << q[3] << " " << q[0];  // [qx qy qz qw]
            else
                fs << q[0] << " " << q[1] << " " << q[2] << " " << q[3];  // [qw qx qy qz]
            fs << std::endl;
        }
    }
    return true;
}

std::optional<TumTimestampDiff> MaxMatchTimeDifference(const std::vector<TumTimestampPose>& ground)
{
    if (ground.size() < 1)
        return std::nullopt;

    // Ground truth sequence is sampled much faster 4.7ms or 210fps,
    // so one may take this sample interval of 4.7ms.
    // Alternatively, one may take twice this interval; this way is done in TUM RGB-D benchmark
    // source: file "evaluate_rpe.py" taken here https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
    std::vector<TumTimestampDiff> diffs;
    diffs.resize(ground.size() - 1);
    for (size_t i = 0; i < diffs.size(); ++i)
        diffs[i] = ground[i + 1].timestamp - ground[i].timestamp;

    auto median = LeftMedianInplace(&diffs);
    if (!median.has_value()) return std::nullopt;

    return median.value() * 2;  // x2 like in TUM RGB-D benchmark, perhaps for certainty
}

ptrdiff_t SearchClosestGroundTruthPeerByTimestamp(
    TumTimestamp ask_time,
    const std::vector<TumTimestampPose>& ground,
    ptrdiff_t start_gt_ind,
    TumTimestampDiff max_time_diff)
{
    // find ground truth candidate
    std::optional<TumTimestampDiff> prev_time_dist;
    ptrdiff_t min_time_gt_ind = -1;
    for (size_t gt_cand_ind = start_gt_ind; gt_cand_ind < ground.size(); ++gt_cand_ind)
    {
        const TumTimestampPose& g = ground[gt_cand_ind];
        if (!g.pos.has_value()) continue;  // need non-empty gt

        TumTimestampDiff cur_time_dist = std::abs(g.timestamp - ask_time);

        // assume the ground truth is ordered ascending
        // continue to search the closest ground truth sample if time difference is decreasing
        // otherwise the closest sample is found
        if (!prev_time_dist.has_value() || cur_time_dist < prev_time_dist.value())
        {
            min_time_gt_ind = gt_cand_ind;
            prev_time_dist = cur_time_dist;
        }
        else
        {
            break;
        }
    }

    if (min_time_gt_ind != -1)  // found the candidate for correspondence
    {
        TumTimestampDiff time_mismatch = std::abs(ground[min_time_gt_ind].timestamp - ask_time);
        if (time_mismatch > max_time_diff)
            min_time_gt_ind = -1;  // reject distant correspondence
    }

    return min_time_gt_ind;
}

/// For each camera position in need_match assigns the ground truth timestamp, so that
// estim[i] has ground truth ground[gt_inds[i]].
size_t AssignCloseIndsByTimestamp(
    const std::vector<TumTimestampPose>& ground,
    const std::vector<TumTimestamp>& need_match,
    std::optional<TumTimestampDiff> max_time_diff,
    std::vector<ptrdiff_t>* gt_inds)
{
    gt_inds->resize(need_match.size());

    size_t gt_ind = 0;
    size_t found_gt_count = 0;
    for (size_t ask_ind = 0; ask_ind < need_match.size(); ++ask_ind)
    {
        TumTimestamp ask_time = need_match[ask_ind];

        // find ground truth candidate
        std::optional<TumTimestampDiff> prev_time_dist;
        ptrdiff_t min_time_gt_ind = -1;
        for (size_t gt_cand_ind = gt_ind; gt_cand_ind < ground.size(); ++gt_cand_ind)
        {
            const TumTimestampPose& g = ground[gt_cand_ind];
            if (!g.pos.has_value()) continue;

            TumTimestampDiff cur_time_dist = std::abs(g.timestamp - ask_time);

            // assume the ground truth is ordered ascending
            // continue to search the closest ground truth sample if time difference is decreasing
            // otherwise the closest sample is found
            if (!prev_time_dist.has_value() || cur_time_dist < prev_time_dist.value())
            {
                min_time_gt_ind = gt_cand_ind;
                prev_time_dist = cur_time_dist;
            }
            else
            {
                break;
            }
        }

        if (min_time_gt_ind != -1 && max_time_diff.has_value())  // found the candidate for correspondence
        {
            TumTimestampDiff time_mismatch = std::abs(ground[min_time_gt_ind].timestamp - ask_time);
            if (time_mismatch > max_time_diff.value())
                min_time_gt_ind = -1;  // reject distant correspondence
        }

        (*gt_inds)[ask_ind] = min_time_gt_ind;

        // move ground truth index
        if (min_time_gt_ind != -1)
        {
            gt_ind = min_time_gt_ind;
            found_gt_count++;
        }
    }

    // postcondition
    SRK_ASSERT(gt_inds->size() == need_match.size());

    static bool compare_with_naive_impl = false;
    if (compare_with_naive_impl)
    {
        std::vector<ptrdiff_t> gt_inds_naive;
        size_t found_gt_count_naive  = AssignCloseIndsByTimestampNaive(ground, need_match, max_time_diff, &gt_inds_naive);
        SRK_ASSERT(found_gt_count_naive == found_gt_count) << "can't assign gt to trajectory, expected(naive): " << found_gt_count_naive << ", actual: " << found_gt_count;
        // assert this impl matches the naive impl
        for (size_t i = 0; i < need_match.size(); ++i)
        {
            auto expect = gt_inds_naive[i];
            auto actual = (*gt_inds)[i];
            SRK_ASSERT(expect == actual);
        }
    }
    return found_gt_count;
}

size_t AssignCloseIndsByTimestampNaive(
    const std::vector<TumTimestampPose>& ground,
    const std::vector<TumTimestamp>& need_match,
    std::optional<TumTimestampDiff> max_time_diff,
    std::vector<ptrdiff_t>* gt_inds)
{
    gt_inds->resize(need_match.size());

    constexpr ptrdiff_t kNoInd = -1;
    size_t found_gt_count = 0;
    for (size_t ask_ind = 0; ask_ind < need_match.size(); ++ask_ind)
    {
        TumTimestamp ask_stamp = need_match[ask_ind];

        auto closest_dist = std::numeric_limits<TumTimestampDiff>::max();
        auto closest_ind = kNoInd;
        for (size_t gt_ind = 0; gt_ind < ground.size(); ++gt_ind)
        {
            auto time_dist = std::abs(ground[gt_ind].timestamp - ask_stamp);
            if (time_dist < closest_dist)
            {
                closest_dist = time_dist;
                closest_ind = gt_ind;
            }
        }

        // ignore far away ground truth
        if (closest_ind != kNoInd &&
            max_time_diff.has_value() &&
            closest_dist > max_time_diff) closest_ind = kNoInd;

        (*gt_inds)[ask_ind] = closest_ind;
        found_gt_count += (closest_ind != kNoInd ? 1 : 0);
    }
    return found_gt_count;
}
}
