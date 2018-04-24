#include <iostream>
#include <memory>
#include <cassert>
#include <stdint.h>
#include <chrono>
#include <sstream>
#include <limits>
#include <cassert>
#include <numeric>
#include <cstdlib>

#include <arrow/ipc/feather.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/builder.h>
#include <bitset>
#include <arrow/io/interfaces.h>
#include <unordered_map>
#include "feature_calculator.h"
using namespace std;

class FeatureCalculatorImpl : public GroupedFeatureCalculator<uint16_t, arrow::UInt16Type> {
public:
  FeatureCalculatorImpl(): GroupedFeatureCalculator() {}
  
  virtual string name() {
    return "MedianAttributeTime";
  }

  virtual vector<uint16_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    const uint64_t nanoseconds_per_second = (1000 * 1000 * 1000);
    vector<uint16_t> feature(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      vector<uint64_t> time_delta;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        size_t curr_index = group_click_times[index];
        if (is_attributed[curr_index]) {
          assert(attributed_time[curr_index] >= click_time[curr_index]);
          time_delta.push_back((attributed_time[curr_index] - click_time[curr_index]) / nanoseconds_per_second);
        }
      }

      uint16_t delta = 0;
      if (time_delta.empty()) {
        delta = numeric_limits<uint16_t>::max();
      } else {
        uint64_t delta1 = time_delta[time_delta.size() / 2];
        uint64_t delta2 = time_delta[(time_delta.size() - 1) / 2];
        delta = min<uint64_t>(numeric_limits<uint16_t>::max() - 1, (delta1 + delta2) / 2);
      }

      for (size_t index = 0; index < group_click_times.size(); index++) {
        size_t curr_index = group_click_times[index];
        feature[curr_index] = delta;
      }
    }
    return feature;
  }
};

int main(int argc, char **argv)
{

  if (argc != 5) {
    fprintf(stderr, "Usage: %s (input_train_feather) (input_test_feather)"
            " (output_train_feather) (output_test_feather)\n", argv[0]);
    exit(-1);
  }

  FeatureCalculatorImpl feature_calculator;
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4]);
  assert(status.ok());
}

