#include <iostream>
#include <memory>
#include <cassert>
#include <stdint.h>
#include <chrono>
#include <sstream>
#include <cassert>

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

class IntervalCounter : public FeatherFeatureCalculator {
public:
  IntervalCounter(uint64_t window_size_in_seconds): FeatherFeatureCalculator(), window_size_in_nanoseconds(window_size_in_seconds * 1000000000ULL) {}

  virtual string name() {
    // I don't care performance of this method
    std::ostringstream o;
    // I don't like this hard-coded constant
    o << window_size_in_nanoseconds / 1000000000ULL;
    return "PastClickCount_" + o.str();
  }
  
  
  arrow::Status calculate(const std::string &train_input_path,
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

    int64_t train_size = read_single_feather_file_with_stopwatch("Read train table", train_input_path);
    int64_t valid_size = read_single_feather_file_with_stopwatch("Read validation table", valid_input_path);
    int64_t test_size = read_single_feather_file_with_stopwatch("Read test table", test_input_path);
    assert(int64_t(ip.size()) == train_size + valid_size + test_size);

    unordered_map<uint64_t, vector<size_t>> grouped_click_times;

    auto train_writer = unique_ptr<arrow::ipc::feather::TableWriter>();
    auto valid_writer = unique_ptr<arrow::ipc::feather::TableWriter>();
    auto test_writer = unique_ptr<arrow::ipc::feather::TableWriter>();
    auto train_file = shared_ptr<arrow::io::FileOutputStream>();
    auto valid_file = shared_ptr<arrow::io::FileOutputStream>();
    auto test_file = shared_ptr<arrow::io::FileOutputStream>();
    ARROW_RETURN_NOT_OK(arrow::io::FileOutputStream::Open(string(train_output_path), &train_file));
    ARROW_RETURN_NOT_OK(arrow::io::FileOutputStream::Open(string(valid_output_path), &valid_file));
    ARROW_RETURN_NOT_OK(arrow::io::FileOutputStream::Open(string(test_output_path), &test_file));
    ARROW_RETURN_NOT_OK(arrow::ipc::feather::TableWriter::Open(train_file, &train_writer));
    ARROW_RETURN_NOT_OK(arrow::ipc::feather::TableWriter::Open(valid_file, &valid_writer));
    ARROW_RETURN_NOT_OK(arrow::ipc::feather::TableWriter::Open(test_file, &test_writer));
    arrow::MemoryPool* pool = arrow::default_memory_pool();
    train_writer->SetNumRows(ip.size());
    valid_writer->SetNumRows(ip.size());
    test_writer->SetNumRows(ip.size());
    
    for (uint16_t key_mask = 1; key_mask <= KEY_MASK_ALL; key_mask++) {
      grouped_click_times.clear();
      const uint64_t hash_mask = generate_hash_mask(key_mask);
      for (size_t i = 0; i < ip.size(); i++) {
        const uint64_t h = generate_hash(i, hash_mask);
        grouped_click_times[h].push_back(i);
      }

      vector<int> click_count(ip.size());
      for (const auto &entry : grouped_click_times) {
        const auto &group_click_times = entry.second;
        const size_t size = group_click_times.size();
        size_t cursor = 0;
        for (size_t index = 0; index < group_click_times.size(); index++) {
          while (cursor < size && (click_time[group_click_times[index]] - click_time[group_click_times[cursor]] > window_size_in_nanoseconds)) {
            cursor++;
          }
          click_count[group_click_times[index]] = index - cursor + 1;
        }
      }
      const string feature_name = this->name() + get_feature_name_suffix(key_mask);
      cout << feature_name << endl;


      // Terrible copy & paste
      {
        auto array = shared_ptr<arrow::Array>();
        arrow::NumericBuilder<arrow::UInt32Type> builder(pool);
        for (int64_t i = 0; i < train_size; i++) {
          ARROW_RETURN_NOT_OK(builder.Append(click_count[i]));
        }
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        ARROW_RETURN_NOT_OK(train_writer->Append(feature_name, *array));
      }

      {
        auto array = shared_ptr<arrow::Array>();
        arrow::NumericBuilder<arrow::UInt32Type> builder(pool);
        for (int64_t i = 0; i < valid_size; i++) {
          ARROW_RETURN_NOT_OK(builder.Append(click_count[i + train_size]));
        }
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        ARROW_RETURN_NOT_OK(valid_writer->Append(feature_name, *array));
      }

      {
        auto array = shared_ptr<arrow::Array>();
        arrow::NumericBuilder<arrow::UInt32Type> builder(pool);
        for (int64_t i = 0; i < test_size; i++) {
          ARROW_RETURN_NOT_OK(builder.Append(click_count[i + train_size + valid_size]));
        }
        ARROW_RETURN_NOT_OK(builder.Finish(&array));
        ARROW_RETURN_NOT_OK(test_writer->Append(feature_name, *array));
      }
    }
    ARROW_RETURN_NOT_OK(train_writer->Finalize());
    ARROW_RETURN_NOT_OK(valid_writer->Finalize());
    ARROW_RETURN_NOT_OK(test_writer->Finalize());
    return arrow::Status::OK();
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
  
  IntervalCounter feature_calculator(600);
  arrow::Status status =feature_calculator.calculate(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
  assert(status.ok());
}
