#ifndef FEATURE_CALCULATOR_H
#define FEATURE_CALCULATOR_H

#include <stdint.h>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <numeric>
#include <chrono>
#include <arrow/status.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/builder.h>
#include <arrow/ipc/feather.h>

class FeatherFeatureCalculator {
 public:
  virtual ~FeatherFeatureCalculator() {
    
  }

  virtual arrow::Status calculate(const std::string &train_input_path,
                                  const std::string &valid_input_path,
                                  const std::string &test_input_,
                                  const std::string &train_ouptut_path,
                                  const std::string &valid_ouptut_path,
                                  const std::string &test_ouptut_path) = 0;
  virtual std::string name() = 0;
  
 protected:
  static const uint16_t KEY_IP = 1 << 0;
  static const uint16_t KEY_APP = 1 << 1;
  static const uint16_t KEY_DEVICE = 1 << 2;
  static const uint16_t KEY_OS = 1 << 3;
  static const uint16_t KEY_CHANNEL = 1 << 4;
  static const uint16_t KEY_MASK_ALL = KEY_IP | \
    KEY_APP | \
    KEY_DEVICE | \
    KEY_OS | \
    KEY_CHANNEL;
    
  static const uint64_t MASK_IP = ((uint64_t(1) << 20) - 1) << 44;
  static const uint64_t MASK_APP = ((uint64_t(1) << 10) - 1) << 34;
  static const uint64_t MASK_DEVICE = ((uint64_t(1) << 14) - 1) << 20;
  static const uint64_t MASK_OS = ((uint64_t(1) << 10) - 1) << 10;
  static const uint64_t MASK_CHANNEL = ((uint64_t(1) << 10) - 1);
  
  inline uint64_t generate_hash_mask(uint16_t key_mask) {
    uint64_t res = 0;
    if (key_mask & KEY_IP) {
      res |= MASK_IP;
    }

    if (key_mask & KEY_APP) {
      res |= MASK_APP;
    }    

    if (key_mask & KEY_DEVICE) {
      res |= MASK_DEVICE;
    }

    if (key_mask & KEY_OS) {
      res |= MASK_OS;
    }

    if (key_mask & KEY_CHANNEL) {
      res |= MASK_CHANNEL;
    }

    return res;
  }
  
  inline uint64_t generate_hash(size_t i, uint64_t hash_mask) {
    uint64_t ip_ = ip[i];
    uint64_t app_ = app[i];
    uint64_t device_ = device[i];
    uint64_t os_ = os[i];
    uint64_t channel_ = channel[i];
    uint64_t res = (ip_ << 44) | (app_ << 34) | (device_ << 20) | (os_ << 10) | channel_;
    return res & hash_mask;
  }

  inline std::string get_feature_name_suffix(uint16_t key_mask) {
    std::string res;
    res += (key_mask & KEY_IP) ? "-ip" : "";
    res += (key_mask & KEY_APP) ? "-app" : "";
    res += (key_mask & KEY_DEVICE) ? "-dev" : "";
    res += (key_mask & KEY_OS) ? "-os" : "";
    res += (key_mask & KEY_CHANNEL) ? "-cha" : "";
    return res;
  }

  int64_t read_single_feather_file(const std::string &path);
  int64_t read_single_feather_file_with_stopwatch(const std::string &message, const std::string &path);
  /* int64_t write_single_feather_file(const std::string &path); */
  std::vector<uint32_t> ip;
  std::vector<uint16_t> app;
  std::vector<uint16_t> device;
  std::vector<uint16_t> os;
  std::vector<uint16_t> channel;
  std::vector<uint64_t> click_time;
  uint64_t train_size;
  uint64_t valid_size;
  uint64_t test_size;
  
};

template <typename TargetType, typename TargetArrowType> class GroupedFeatureCalculator :  public FeatherFeatureCalculator {
 public:
  virtual arrow::Status calculate(const std::string &train_input_path,
                                  const std::string &valid_input_path,
                                  const std::string &test_input_path,
                                  const std::string &train_output_path,
                                  const std::string &valid_output_path,
                                  const std::string &test_output_path)
  {
    assert(ip.empty());
    assert(app.empty());
    assert(device.empty());
    assert(os.empty());
    assert(channel.empty());
    assert(click_time.empty());

    const int input_size = 3;
    std::vector<std::string> input_paths = { train_input_path, valid_input_path, test_input_path };
    std::vector<std::string> output_paths = { train_output_path, valid_output_path, test_output_path };
    std::vector<std::string> read_messages = { "Read train table", "Read train table", "Read test table" };
    std::vector<int64_t> sizes;
    for (int i = 0; i < input_size; i++) {
      sizes.push_back(read_single_feather_file_with_stopwatch(read_messages[i], input_paths[i]));
    }
    assert(int64_t(ip.size()) == std::accumulate(sizes.begin(), sizes.end(), 0L));
    
    std::unordered_map<uint64_t, std::vector<size_t>> grouped_click_times;
    std::vector<std::unique_ptr<arrow::ipc::feather::TableWriter>> writers;
    std::vector<std::shared_ptr<arrow::io::FileOutputStream>> output_streams;
    for (int i = 0; i < input_size; i++) {
      auto os = std::shared_ptr<arrow::io::FileOutputStream>();
      writers.push_back(std::unique_ptr<arrow::ipc::feather::TableWriter>());
      ARROW_RETURN_NOT_OK(arrow::io::FileOutputStream::Open(output_paths[i], &os));
      ARROW_RETURN_NOT_OK(arrow::ipc::feather::TableWriter::Open(os, &writers[i]));
      output_streams.push_back(os);
      
    }

    arrow::MemoryPool* pool = arrow::default_memory_pool();
    for (auto &writer : writers) {
      writer->SetNumRows(ip.size());
    }
    
    for (uint16_t key_mask = 1; key_mask <= KEY_MASK_ALL; key_mask++) {
      if (!this->filter(key_mask)) {
        continue;
      }
      
      grouped_click_times.clear();
      
      auto hash_construction_start = std::chrono::system_clock::now();
      const uint64_t hash_mask = generate_hash_mask(key_mask);
      for (size_t i = 0; i < ip.size(); i++) {
        const uint64_t h = generate_hash(i, hash_mask);
        grouped_click_times[h].push_back(i);
      }
      auto hash_construction_duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - hash_construction_start);
      
      auto feature_caculation_start = std::chrono::system_clock::now();
      std::vector<TargetType> feature = calculate_feature(grouped_click_times);
      auto feature_caculation_duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - feature_caculation_start);

      auto output_start = std::chrono::system_clock::now();
      const std::string feature_name = this->name() + get_feature_name_suffix(key_mask);
      for (int i = 0; i < input_size; i++) {
        auto &writer = writers[i];
        auto array = std::shared_ptr<arrow::Array>();
        arrow::NumericBuilder<TargetArrowType> builder(pool);

        int64_t offset = 0;
        for (int j = 0; j < i; j++) {
          offset += sizes[j];
        }
        
        for (int64_t j = 0; j < sizes[i]; j++) {
          ARROW_RETURN_NOT_OK(builder.Append(feature[j + offset]));
        }
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        ARROW_RETURN_NOT_OK(writer->Append(feature_name, *array));
      }
      auto output_duration = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::system_clock::now() - output_start);
      std::cout << feature_name << ": hash = " << hash_construction_duration.count() << " [s], feature = "
                << feature_caculation_duration.count() << " [s], output = " << output_duration.count() << " [s]" << std::endl;
    }
    for (auto &writer: writers) {
      ARROW_RETURN_NOT_OK(writer->Finalize());
    }
    return arrow::Status::OK();
  }
  
 protected:
  virtual std::vector<TargetType> calculate_feature(const std::unordered_map<uint64_t, std::vector<size_t>> &grouped_click_times) = 0;
  virtual bool filter(uint16_t key_mask) {
    return true;
  }
};


#endif /* FEATURE_CALCULATOR_H */
