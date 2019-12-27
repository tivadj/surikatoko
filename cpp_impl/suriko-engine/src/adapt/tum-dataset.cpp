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
using namespace suriko::internals;
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
    const std::vector<TumTimestamp>& need_match,
    const std::vector<std::optional<TumTimestamp>>& ground,
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
            std::optional<TumTimestamp> g = ground[gt_cand_ind];
            if (!g.has_value()) continue;

            TumTimestampDiff cur_time_dist = std::abs(g.value() - ask_time);

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
            TumTimestampDiff time_mismatch = std::abs(ground[min_time_gt_ind].value() - ask_time);
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
        size_t found_gt_count_naive  = AssignCloseIndsByTimestampNaive(need_match, ground, max_time_diff, &gt_inds_naive);
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
    const std::vector<TumTimestamp>& need_match,
    const std::vector<std::optional<TumTimestamp>>& ground,
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
            std::optional<TumTimestamp> g = ground[gt_ind];
            if (!g.has_value()) continue;

            auto time_dist = std::abs(g.value() - ask_stamp);
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

std::optional<Scalar> DecodeDepthImagePixel(uint16_t encoded_depth)
{
    // https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
    constexpr Scalar kDepthFactor = 5000; // for the 16 - bit PNG files
    constexpr uint16_t kUnknownDepth = 0;

    if (encoded_depth == kUnknownDepth) return std::nullopt;

    Scalar z = encoded_depth / kDepthFactor;  // in meters
    return z;
}

void TumRgbdDataset::SetDatasetDirpath(std::filesystem::path tum_dataset_dirpath)
{
    if (tum_dataset_dirpath.empty() || !is_directory(tum_dataset_dirpath))
    {
        std::stringstream ss;
        ss << "tum_dataset_dirpath is not a directory.";
        err_msg_ = ss.str();
        return;
    }

    tum_dataset_dirpath_ = tum_dataset_dirpath;
}

void TumRgbdDataset::LoadRgbHeaders()
{
    if (HasErrors()) return;

    // load timestamps from file names of rgb images
    std::filesystem::path rgb_filepath = tum_dataset_dirpath_ / "rgb.txt"sv;
    rgb_timestamp_and_filename_.reserve(1024);
    
    bool op = ReadTumDatasetTimedRgb(rgb_filepath, &rgb_timestamp_and_filename_, &err_msg_);
    if (!op)
    {
        std::stringstream ss;
        ss << "Can't read TUM 'rgb' file: " << rgb_filepath << ", error: " << err_msg_;
        err_msg_ = ss.str();
        return;
    }

    rgb_timestamps_.resize(rgb_timestamp_and_filename_.size());
    std::transform(rgb_timestamp_and_filename_.begin(), rgb_timestamp_and_filename_.end(), rgb_timestamps_.begin(), [](const TumTimestampFilename& i) { return i.timestamp; });
}

void TumRgbdDataset::LoadAndAssignDepthImageFilepathes(std::optional<TumTimestampDiff> rgb_and_depth_max_time_diff)
{
    if (HasErrors()) return;

    // load timestamps from file names of depth images
    std::filesystem::path depth_filepath = tum_dataset_dirpath_ / "depth.txt"sv;
    depth_timestamp_and_filename_.reserve(1024);

    bool op = ReadTumDatasetTimedRgb(depth_filepath, &depth_timestamp_and_filename_, &err_msg_);
    if (!op)
    {
        std::stringstream ss;
        ss << "Can't read TUM 'depth' file: " << depth_filepath << ", error: " << err_msg_;
        err_msg_ = ss.str();
        return;
    }

    depth_timestamps_.resize(depth_timestamp_and_filename_.size());
    std::transform(depth_timestamp_and_filename_.begin(), depth_timestamp_and_filename_.end(), depth_timestamps_.begin(),
        [](const TumTimestampFilename& i) { return i.timestamp; });

    std::vector<std::optional<TumTimestamp>> depth_timestamps_opt;
    depth_timestamps_opt.resize(depth_timestamps_.size());
    std::transform(depth_timestamps_.begin(), depth_timestamps_.end(), depth_timestamps_opt.begin(),
        [](const auto& p) { return std::make_optional(p); });

    size_t found_depth_count = AssignCloseIndsByTimestamp(rgb_timestamps_, depth_timestamps_opt, rgb_and_depth_max_time_diff, &rgb_pose_to_depth_ind_);
}

/// \param premult_gt_from_cam0 transforms from frame of camera0 into the frame of ground truth: result=premult_gt_from_cam0 * cam0_from_cami
void TumRgbdDataset::LoadAndAssignGtTraj(const std::vector<TumTimestamp>& rgb_stamps,
    std::optional<TumTimestampDiff> max_time_diff,
    std::optional<Eigen::Matrix<Scalar, 3, 3>> premult_gt_from_cam0,
    std::vector<std::optional<SE3Transform>>* gt_cam_orient_cfw)
{
    if (HasErrors()) return;

    // TUM dataset has ground truth trajectory in oversampled frequency (4ms), while images are collected every 33ms.
    // ground truth
    std::filesystem::path gt_filepath = tum_dataset_dirpath_ / "groundtruth.txt"sv;
    std::vector<TumTimestampPose> oversampled_poses_wfc_gt_tum;
    std::string err_msg;
    bool op = ReadTumDatasetGroundTruth(gt_filepath, &oversampled_poses_wfc_gt_tum, &err_msg);
    if (!op)
    {
        std::stringstream ss;
        ss << "Can't read TUM ground truth file: " << gt_filepath << ", error: " << err_msg;
        err_msg_ = ss.str();
        return;
    }

    std::vector<std::optional<SE3Transform>> gt_poses_wfc_gt;
    std::transform(oversampled_poses_wfc_gt_tum.begin(), oversampled_poses_wfc_gt_tum.end(), std::back_inserter(gt_poses_wfc_gt),
        [](const auto& t) { return TimestampPoseToSE3(t); });

    double gt_total_dur = -1;
    Scalar gt_total_rot_ang = -1;
    Scalar gt_poses_traj_len = CalcTrajectoryLengthWfc(&gt_poses_wfc_gt, nullptr,
        [&](size_t i) -> std::optional<double> {
            return oversampled_poses_wfc_gt_tum[i].timestamp;
        }, &gt_total_dur, &gt_total_rot_ang);
    LOG(INFO) << "gt traj:"
        << " frames:" << oversampled_poses_wfc_gt_tum.size()
        << ", length[m]: " << gt_poses_traj_len
        << ", dur[s]: " << gt_total_dur
        << ", avg transl vel[m/s]:" << gt_poses_traj_len / gt_total_dur
        << ", avg angular vel[deg/s]:" << Rad2Deg(gt_total_rot_ang) / gt_total_dur;

    // maximal allowed difference between ground truth and rgb image timestamp
    if (!max_time_diff.has_value())
        max_time_diff = MaxMatchTimeDifference(oversampled_poses_wfc_gt_tum);
    if (!max_time_diff.has_value())
    {
        std::stringstream ss;
        ss << "Can't determine max_time_diff";
        err_msg_ = ss.str();
        return;
    }

    std::vector<std::optional<TumTimestamp>> oversampled_gt_stamps_opt{ oversampled_poses_wfc_gt_tum.size() };
    oversampled_gt_stamps_opt.resize(oversampled_poses_wfc_gt_tum.size());
    std::transform(oversampled_poses_wfc_gt_tum.begin(), oversampled_poses_wfc_gt_tum.end(), oversampled_gt_stamps_opt.begin(),
        [](const auto& p) { return std::make_optional(p.timestamp); });

    std::vector<ptrdiff_t> rgb_poses_gt_inds;
    size_t found_gt_count = AssignCloseIndsByTimestamp(rgb_stamps, oversampled_gt_stamps_opt, max_time_diff, &rgb_poses_gt_inds);

    // construct subset of ground truth trajectory, corresponding to rgb images
    std::vector<std::optional<SE3Transform>> rgb_poses_wfc_gt{ rgb_stamps.size() };
    for (size_t i = 0; i < rgb_poses_wfc_gt.size(); ++i)
    {
        ptrdiff_t gt_ind = rgb_poses_gt_inds[i];
        if (gt_ind != -1)
        {
            const TumTimestampPose& p = oversampled_poses_wfc_gt_tum[gt_ind];
            rgb_poses_wfc_gt[i] = TimestampPoseToSE3(p);
        }
    }

    double rgb_total_dur = -1;
    Scalar rgb_total_rot_ang = -1;
    Scalar rgb_poses_gt_traj_len = CalcTrajectoryLengthWfc(&rgb_poses_wfc_gt, nullptr,
        [&](size_t i) -> std::optional<double> {
            ptrdiff_t gt_ind = rgb_poses_gt_inds[i];
            if (gt_ind == -1) return std::nullopt;
            return oversampled_poses_wfc_gt_tum[gt_ind].timestamp;
        }, &rgb_total_dur, &rgb_total_rot_ang);
    LOG(INFO) << "estim traj"
        << " frames:" << rgb_poses_wfc_gt.size()
        << ", frames_with_gt:" << found_gt_count
        << ", length[m]: " << rgb_poses_gt_traj_len
        << ", dur[s]: " << rgb_total_dur
        << ", avg trans vel[m/s]:" << rgb_poses_gt_traj_len / rgb_total_dur
        << ", avg angul vel[deg/s]:" << Rad2Deg(rgb_total_rot_ang) / rgb_total_dur;

    // ground truth corresponding to the first camera frame
    ptrdiff_t first_cam_gt_ind = rgb_poses_gt_inds[0];
    const TumTimestampPose& first_cam_wfc_gt_tum = oversampled_poses_wfc_gt_tum[first_cam_gt_ind];
    std::optional<SE3Transform> first_cam_wfc0_gt = TimestampPoseToSE3(first_cam_wfc_gt_tum);
    if (!first_cam_wfc0_gt.has_value())
    {
        std::stringstream ss;
        ss << "Can't determine first_cam_wfc0_gt";
        err_msg_ = ss.str();
        return;
    }

    // transform the entire ground truth sequence, so that a ground truth frame, corresponding to the first camera's frame, coincides with the origin
    SE3Transform first_cam_c0fw_gt = SE3Inv(first_cam_wfc0_gt.value());  // rgb cam0 from world

    // this may transform from X-right-Y-bottom into X-left-Y-top (which is the default) coordinates
    SE3Transform premult_gt_from_cam0_se3 = SE3Transform::NoTransform();
    if (premult_gt_from_cam0.has_value())
        premult_gt_from_cam0_se3.R = premult_gt_from_cam0.value();

    std::array<Scalar, 4> prefix_quat;
    op = QuatFromRotationMat(premult_gt_from_cam0_se3.R, prefix_quat);
    SRK_ASSERT(op);

    for (const std::optional<SE3Transform>& wfci : rgb_poses_wfc_gt)  // world from cami
    {
        std::optional<SE3Transform> cifc0;  // cam0 from cami
        if (wfci.has_value())
        {
            auto c0fci = SE3Compose(first_cam_c0fw_gt, wfci.value());

            c0fci = SE3Compose(premult_gt_from_cam0_se3, c0fci);  // apply X-right-y_bottom to X-left-Y-top transformation

            cifc0 = SE3Inv(c0fci);
        }
        gt_cam_orient_cfw->push_back(cifc0);
    }
}

std::filesystem::path TumRgbdDataset::GetRgbDirpath() const
{
    return tum_dataset_dirpath_ / "rgb/";
}

std::optional<std::filesystem::path> TumRgbdDataset::GetDepthFilepathForRgbImage(size_t rgb_image_ind) const
{
    if (rgb_pose_to_depth_ind_.empty()) return std::nullopt;

    ptrdiff_t depth_img_ind = rgb_pose_to_depth_ind_[rgb_image_ind];
    if (depth_img_ind == -1)
        return std::nullopt;

    const TumTimestampFilename& depth_info = depth_timestamp_and_filename_[depth_img_ind];
    return tum_dataset_dirpath_ / depth_info.filename;
}
}
