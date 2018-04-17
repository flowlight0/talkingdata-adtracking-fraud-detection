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

class ExactSameClick : public GroupedFeatureCalculator<uint8_t, arrow::UInt16Type> {
public:
  ExactSameClick(): GroupedFeatureCalculator() {}
  
  virtual string name() {
    return "ExactSameClick";
  }

  virtual vector<uint8_t> calculate_feature(const unordered_map<uint64_t, vector<size_t>> &grouped_click_times) {
    vector<uint8_t> future_click_count(ip.size());
    for (const auto &entry : grouped_click_times) {
      const auto &group_click_times = entry.second;
      for (size_t index = 0; index < group_click_times.size(); index++) {
        future_click_count[group_click_times[index]] = group_click_times.size();
      }
    }
    return future_click_count;
  }

  virtual bool filter(uint16_t key_mask) {
    return key_mask == KEY_MASK_ALL;
  }
};

int main(int argc, char **argv)
{

  if (argc != 5) {
    fprintf(stderr, "Usage: %s (input_train_feather) (input_test_feather)"
            " (output_train_feather) (output_test_feather)\n", argv[0]);
    exit(-1);
  }
  
  ExactSameClick feature_calculator;
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4]);
  assert(status.ok());
}
