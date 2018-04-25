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
    return "MedianAttributeTimePast";
  }

  virtual vector<uint16_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    const uint64_t nanoseconds_per_second = (1000 * 1000 * 1000);
    const uint64_t nanoseconds_per_day = 24 * 60 * 60 * int64_t(1000 * 1000 * 1000);
    vector<uint16_t> feature(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      size_t cursor = 0;
      vector<uint64_t> time_delta;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        size_t curr_index = group_click_times[index];
        if (time_delta.empty()) {
          feature[curr_index] = numeric_limits<uint16_t>::max();
        } else {
          uint64_t delta1 = time_delta[time_delta.size() / 2];
          uint64_t delta2 = time_delta[(time_delta.size() - 1) / 2];
          feature[curr_index] = min<uint64_t>(numeric_limits<uint16_t>::max() - 1, (delta1 + delta2) / 2);
        }
        
        while (cursor < index && (click_time[curr_index] - click_time[group_click_times[cursor]] >= nanoseconds_per_day)) {
          const size_t past_index = group_click_times[cursor];
          if (is_attributed[past_index]) {
            assert(attributed_time[past_index] >= click_time[past_index]);
            time_delta.push_back((attributed_time[past_index] - click_time[past_index]) / nanoseconds_per_second);
          }
          cursor++;
        }
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

