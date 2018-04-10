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

class FutureClickCount : public GroupedFeatureCalculator<uint16_t, arrow::UInt16Type> {
public:
  FutureClickCount(uint64_t window_size_in_seconds): GroupedFeatureCalculator(), window_size_in_nanoseconds(window_size_in_seconds * 1000000000ULL) {}
  
  virtual string name() {
    // I don't care performance of this method
    std::ostringstream o;
    // I don't like this hard-coded constant
    o << window_size_in_nanoseconds / 1000000000ULL;
    return "FutureClickCount_" + o.str();
  }

  virtual vector<uint16_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    vector<uint16_t> future_click_count(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      const size_t size = group_click_times.size();
      size_t cursor = 0;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        while (cursor < size && (click_time[group_click_times[cursor]] - click_time[group_click_times[index]]) <= window_size_in_nanoseconds) {
          cursor++;
        }
        future_click_count[group_click_times[index]] = min<int>(cursor - index, numeric_limits<uint16_t>::max());
      }
    }
    return future_click_count;
  }
  
private:
  uint64_t window_size_in_nanoseconds;
};

int main(int argc, char **argv)
{

  if (argc != 8) {
    fprintf(stderr, "Usage: %s (input_train_feather) (input_valid_feather) (input_test_feather)"
            " (output_train_feather) (output_valid_feather) (output_test_feather) (window_size_in_seconds)\n", argv[0]);
    exit(-1);
  }
  
  FutureClickCount feature_calculator(atoi(argv[7]));
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
  assert(status.ok());
}
