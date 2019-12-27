#pragma once
#include <filesystem>
#include <vector>
#include <optional>
#include "suriko/rt-config.h"
#include "suriko/obs-geom.h"
#include "suriko/quat.h"

namespace suriko::adapt::tum
{
using TumTimestamp = double;
using TumTimestampDiff = double;

struct TumTimestampFilename
{
    TumTimestamp timestamp;  // in seconds
    std::string filename;
};

bool ReadTumDatasetTimedRgb(const std::filesystem::path& rgb_file_path,
    std::vector<TumTimestampFilename>* infos, std::string* err_msg);

struct TumTimestampPose
{
    TumTimestamp timestamp;  // in seconds
    std::optional<suriko::Point3> pos;  // [tx ty tz], null if unavailable
    std::optional < Eigen::Matrix<Scalar, 4, 1>> quat;  // [qx qy qz qw], null if unavailable
};

auto TimestampPoseToSE3(const TumTimestampPose& r)->std::optional<SE3Transform>;
auto SE3ToTimestampPose(TumTimestamp stamp, std::optional<SE3Transform> rt_opt)->TumTimestampPose;

bool ReadTumDatasetGroundTruth(const std::filesystem::path& file_path,
    std::vector<TumTimestampPose>* poses_gt, std::string* err_msg = nullptr, bool check_timestamp_order_asc = true);

bool SaveTumDatasetGroundTruth(const std::filesystem::path& file_path,
    std::vector<TumTimestampPose>* poses_gt, QuatLayout quat_layout,
    std::string_view header = std::string_view{},
    std::string* err_msg = nullptr);

std::optional<TumTimestampDiff> MaxMatchTimeDifference(const std::vector<TumTimestampPose>& ground);

ptrdiff_t SearchClosestGroundTruthPeerByTimestamp(
    TumTimestamp ask_time,
    const std::vector<TumTimestampPose>& ground,
    ptrdiff_t start_gt_ind,
    TumTimestampDiff max_time_diff);

/// For each camera position in need_match assigns the ground truth timestamp, so that
// estim[i] has ground truth ground[gt_inds[i]].
size_t AssignCloseIndsByTimestamp(
    const std::vector<TumTimestamp>& need_match,
    const std::vector<std::optional<TumTimestamp>>& ground,
    std::optional<TumTimestampDiff> max_time_diff,
    std::vector<ptrdiff_t>* gt_inds);

size_t AssignCloseIndsByTimestampNaive(
    const std::vector<TumTimestamp>& need_match,
    const std::vector<std::optional<TumTimestamp>>& ground,
    std::optional<TumTimestampDiff> max_time_diff,
    std::vector<ptrdiff_t>* gt_inds);

std::optional<Scalar> DecodeDepthImagePixel(uint16_t encoded_depth);

class TumRgbdDataset
{
    std::filesystem::path tum_dataset_dirpath_;
public:
    std::string err_msg_;
    std::vector<TumTimestampFilename> rgb_timestamp_and_filename_;
    std::vector<TumTimestamp> rgb_timestamps_;              // timestamps of input RGB images
    std::vector<ptrdiff_t> rgb_pose_to_depth_ind_;  // value[i]=index of depth image corresponding to i-th RGB image

    std::vector<TumTimestampFilename> depth_timestamp_and_filename_;
    std::vector<TumTimestamp> depth_timestamps_;            // timestamps of input depth images

public:
    void SetDatasetDirpath(std::filesystem::path tum_dataset_dirpath);
    void LoadRgbHeaders();
    bool HasErrors() const { return !err_msg_.empty(); }

    void LoadAndAssignDepthImageFilepathes(std::optional<TumTimestampDiff> rgb_and_depth_max_time_diff);

    void LoadAndAssignGtTraj(const std::vector<TumTimestamp>& rgb_stamps,
        std::optional<TumTimestampDiff> max_time_diff,
        std::optional<Eigen::Matrix<Scalar, 3, 3>> premult_gt_from_cam0,
        std::vector<std::optional<SE3Transform>>* gt_cam_orient_cfw);

    std::filesystem::path GetRgbDirpath() const;
    std::optional<std::filesystem::path> GetDepthFilepathForRgbImage(size_t rgb_image_ind) const;
};

}
