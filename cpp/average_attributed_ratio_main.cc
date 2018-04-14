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

class FeatureCalculatorImpl : public GroupedFeatureCalculator<float, arrow::FloatType> {
public:
  FeatureCalculatorImpl(): GroupedFeatureCalculator() {}
  
  virtual string name() {
    return "AverageAttributedRatio";
  }

  virtual vector<float> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    uint64_t nanoseconds_per_day = 24 * 60 * 60 * int64_t(1000 * 1000 * 1000);
    vector<float> feature(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      const size_t size = group_click_times.size();
      size_t cursor = 0;
      float attributed_count = 0;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        // click_count[group_click_times[index]] = min<int>(group_click_times.size(), numeric_limits<uint16_t>::max());
        while (cursor < size && (click_time[group_click_times[index]] - click_time[group_click_times[cursor]] >= nanoseconds_per_day)) {
          attributed_count += is_attributed[group_click_times[cursor]];
          cursor++;
        }

        if (cursor == 0) {
          feature[group_click_times[index]] = -1;
        } else {
          feature[group_click_times[index]] = attributed_count / cursor;
        }
      }
    }
    return feature;
  }
};

int main(int argc, char **argv)
{

  if (argc != 7) {
    fprintf(stderr, "Usage: %s (input_train_feather) (input_valid_feather) (input_test_feather)"
            " (output_train_feather) (output_valid_feather) (output_test_feather)\n", argv[0]);
    exit(-1);
  }

  FeatureCalculatorImpl feature_calculator;
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
  assert(status.ok());
}

