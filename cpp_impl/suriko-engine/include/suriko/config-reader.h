#pragma once
#include <filesystem>
#include <unordered_map>
#include <optional>
#include <variant>
#include <vector>

namespace suriko::config
{
struct ParameterEntry
{
    using MultiTypeEntry = std::variant<
        int,
        double,
        std::string
    >;

    // data is either value (optional has value) or a sequence of values
    std::optional<MultiTypeEntry> value;
    std::vector<MultiTypeEntry> seq;

    bool IsValue() const { return value.has_value(); }
    
    /// Holds the number of times the value field has been accessed. Used to warn about unused parameters.
    int read_times = 0;
};

class ConfigReader
{
    std::unordered_map<std::string, ParameterEntry> value_map_;
    std::string err_msg_;
public:
    ConfigReader(const std::filesystem::path& config_file_path);
    void ReadConfig(const std::filesystem::path& config_file_path);

    bool HasErrors() const { return !err_msg_.empty(); }
    std::string_view Error() const { return err_msg_; }

    std::vector<std::string_view> GetUnusedParams() const;

private:
    template <typename ParamType>
    auto GetOrCastValue(ParameterEntry::MultiTypeEntry multi_type_value) -> std::tuple<bool,ParamType>
    {
        static_assert(
            std::is_same_v<ParamType, double> ||
            std::is_same_v<ParamType, int> ||
            std::is_same_v<ParamType, bool> ||
            std::is_same_v<ParamType, std::string>
            );

        // is conversion possible?

        // OpenCV impl of json parser has no bool. Just redirect bool->int.
        if constexpr (std::is_same_v<ParamType, bool>)
        {
            // legitimate conversion: bool param = int{1}
            if (int* result = std::get_if<int>(&multi_type_value))
            {
                auto bool_result = static_cast<bool>(*result);
                return std::make_tuple(true, bool_result);
            }
        }
        else
        {
            // value without conversion
            if (ParamType * result = std::get_if<ParamType>(&multi_type_value))
                return std::make_tuple(true, *result);

            if constexpr (std::is_same_v<ParamType, double>)
            {
                // legitimate conversion: double param = int{1}
                if (int* int_value = std::get_if<int>(&multi_type_value))
                {
                    auto result = static_cast<double>(*int_value);
                    return std::make_tuple(true, result);
                }
            }
        }

        return std::make_tuple(false, ParamType{});
    }

public:
    template <typename ParamType>
    auto GetValue(std::string_view param_name) ->std::optional<ParamType>
    {
        // json provides {bool, int, double, string}
        // for simplicity of implementation, do not allow automatical narrow cast
        static_assert(
            std::is_same_v<ParamType, double> ||
            std::is_same_v<ParamType, int> ||
            std::is_same_v<ParamType, bool> ||
            std::is_same_v<ParamType, std::string>
            );

        auto it = value_map_.find(std::string(param_name));
        if (it == value_map_.end())
            return std::nullopt;

        ParameterEntry& param_value = it->second;
        if (!param_value.IsValue())
        {
            err_msg_ = "Invalid accesing of sequence with GetValue(), seq name=" + it->first;
            return std::nullopt;
        }

        ParamType result{};
        // ParamType p =  {(InitType)value}
        bool got_value = false;
        std::tie(got_value, result) = GetOrCastValue<ParamType>(param_value.value.value());
        if (!got_value)
            return std::nullopt;
        param_value.read_times++;  // this mutates object, making the method non-const; but it allows to track unused parameters
        return result;
    }
    
    template <typename ParamType>
    auto GetSeq(std::string_view param_name)->std::optional<std::vector<ParamType>>
    {
        static_assert(
            std::is_same_v<ParamType, double> ||
            std::is_same_v<ParamType, int>
            );

        auto it = value_map_.find(std::string(param_name));
        if (it == value_map_.end())
            return std::nullopt;

        ParameterEntry& param_value = it->second;
        if (param_value.IsValue())
        {
            err_msg_ = "Invalid accesing of value with GetSeq(), param name=" + it->first;
            return std::nullopt;
        }

        std::vector<ParamType> result;
        result.reserve(param_value.seq.size());

        for (const ParameterEntry::MultiTypeEntry& seq_item : param_value.seq)
        {
            ParamType one_value;
            bool got_value = false;
            std::tie(got_value, one_value) = GetOrCastValue<ParamType>(seq_item);
            if (!got_value)
            {
                err_msg_ = "Can't cast all elements of seq name=" + std::string(param_name) + " to common type="+std::string(typeid(ParamType).name());
                return std::nullopt;
            }
            result.push_back(one_value);
        }
        param_value.read_times++;
        return result;
    }
};

template <typename DesiredFloat>
auto FloatParam(ConfigReader* c, std::string_view param_name) -> std::optional<DesiredFloat>
{
    static_assert(std::is_same_v<DesiredFloat, double> || std::is_same_v<DesiredFloat, float>);

    if constexpr (std::is_same_v<DesiredFloat, double>)
        return c->GetValue<double>(param_name);  // config has only double
    else
    {
        std::optional<double> double_value = c->GetValue<double>(param_name);  // config has only double
        if (!double_value.has_value())
            return std::nullopt;
        return static_cast<DesiredFloat>(double_value.value());
    }
}

template <typename DesiredFloat>
auto FloatSeq(ConfigReader* c, std::string_view param_name) -> std::optional<std::vector<DesiredFloat>>
{
    static_assert(std::is_same_v<DesiredFloat, double> || std::is_same_v<DesiredFloat, float>);

    if constexpr (std::is_same_v<DesiredFloat, double>)
        return c->GetSeq<double>(param_name);
    else
    {
        std::optional<std::vector<double>> double_seq = c->GetSeq<double>(param_name);
        if (!double_seq.has_value())
            return std::nullopt;

        return std::vector<DesiredFloat> {double_seq.value().begin(), double_seq.value().end()};  // cast(float[])(double[])
    }
}
}