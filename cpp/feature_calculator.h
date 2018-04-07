#ifndef FEATURE_CALCULATOR_H
#define FEATURE_CALCULATOR_H

#include <stdint.h>
#include <vector>
#include <string>
#include <arrow/status.h>

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


#endif /* FEATURE_CALCULATOR_H */
