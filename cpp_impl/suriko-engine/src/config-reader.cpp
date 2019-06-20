#include <variant>
#include "suriko/config-reader.h"
#include "suriko/davison-mono-slam.h"

namespace suriko::config {

ConfigReader::ConfigReader(const std::filesystem::path& config_file_path)
{
    ReadConfig(config_file_path);
}

void ConfigReader::ReadConfig(const std::filesystem::path& config_file_path)
{
    cv::FileStorage fs;
    if (!fs.open(config_file_path.string().data(), cv::FileStorage::READ, "utf8"))
    {
        err_msg_ = "Can't read file (" + config_file_path.string() + ")";
        return;
    }

    auto read_node_fun = [&](const cv::FileNode& node) {
        std::string name = std::move(node.name());

        bool got_value_or_seq = false;
        ParameterEntry param_value{ };
        if (node.isInt())
        {
            param_value.value = node.operator int();
            got_value_or_seq = true;
        }
        else if (node.isReal())
        {
            param_value.value = node.real();
            got_value_or_seq = true;
        }
        else if (node.isString())
        {
            param_value.value = node.string();
            got_value_or_seq = true;
        }
        else if (node.isSeq())
        {
            for (const auto& seq_node : node)
            {
                if (seq_node.isInt()) param_value.seq.push_back(seq_node.operator int());
                else if (seq_node.isReal()) param_value.seq.push_back(seq_node.real());
                else if (seq_node.isString()) param_value.seq.push_back(seq_node.string());
                else
                {
                    err_msg_ = "Accept only sequence of int, double or string, sequence name=" + name + ", invalid value=" + seq_node.string();
                    return;
                }
            }
            got_value_or_seq = true;
        }
        
        if (got_value_or_seq)
            value_map_.insert_or_assign(std::move(name), param_value);
    };


    cv::FileNode r = fs.root();
    for (const auto& node : r)
    {
        if (HasErrors()) return;
        read_node_fun(node);
    }
}

std::vector<std::string_view> ConfigReader::GetUnusedParams() const
{
    std::vector<std::string_view> result;
    for (const auto& pair : value_map_)
    {
        // comments are fields starting with double slash
        // json={"// some comment": 0}
        if (pair.first.size() >= 2 && pair.first[0] == '/' && pair.first[1] == '/') continue;
        if (pair.second.read_times == 0) result.push_back(pair.first);
    }
    return result;
}

}