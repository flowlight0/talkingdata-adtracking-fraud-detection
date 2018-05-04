#include <iostream>
#include <memory>
#include <cassert>
#include <stdint.h>
#include <chrono>
#include <algorithm>
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
    return "HourlyClickCount";
  }

  virtual vector<uint16_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    vector<uint16_t> click_count(ip.size());
    const uint64_t nanoseconds_per_second = (1000 * 1000 * 1000);
    const uint64_t seconds_per_hour = 60 * 60;
    const uint64_t nanoseconds_per_hour = nanoseconds_per_second * seconds_per_hour;
    const uint64_t hours_per_day = 24;

    vector<uint64_t> hour_wise_counter(24);
    for (const auto &entry : grouped_click_times) {
      fill(hour_wise_counter.begin(), hour_wise_counter.end(), 0);
      const auto &group_click_times = entry.second;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        size_t curr_index = group_click_times[index];
        uint64_t hour = (click_time[curr_index] / nanoseconds_per_hour) % hours_per_day;
        hour_wise_counter[hour]++;
      }

      for (size_t index = 0; index < group_click_times.size(); index++) {
        size_t curr_index = group_click_times[index];
        uint64_t hour = (click_time[curr_index] / nanoseconds_per_hour) % hours_per_day;
        click_count[curr_index] = min<int>(hour_wise_counter[hour], numeric_limits<uint16_t>::max());
      }
    }
    return click_count;
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
  arrow::Status status = feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4]);
  assert(status.ok());
}
