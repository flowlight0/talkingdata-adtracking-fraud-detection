#include "feature_calculator.h"
#include <arrow/ipc/feather.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <memory>
#include <chrono>
#include <iostream>
#include <cassert>
using std::cout;
using std::endl;

#define CHECK_RESULT_OK(s) do { \
    arrow::Status status = (s); \
    assert(status.ok()); \
  } while(0)

int find_column_number(const std::unique_ptr<arrow::ipc::feather::TableReader> &reader, const std::string &column_name) {
  for (int i = 0; i < reader->num_columns(); i++) {
    if (reader->GetColumnName(i) == column_name) {
      return i;
    }
  }
  return -1;
}

template<typename U, typename V> void read_column_array(const std::shared_ptr<arrow::Column> column, std::vector<U> &column_values) {
  column_values.reserve(column_values.size() + column->length());
  for (auto &chunk: column->data()->chunks()) {
    auto chunk_data = std::static_pointer_cast<arrow::NumericArray<V>>(chunk);
    assert(chunk_data->null_count() == 0);
    for (int i = 0; i < chunk_data->length(); i++) {
      column_values.push_back(chunk_data->Value(i));
    }
  }
}

template<typename U, typename V> void read_column_array(const std::unique_ptr<arrow::ipc::feather::TableReader> &reader, const std::string &column_name, std::vector<U> &column_values) {
  int column_number = find_column_number(reader, column_name);
  assert(column_number >= 0);
  auto column = std::shared_ptr<arrow::Column>();
  CHECK_RESULT_OK(reader->GetColumn(column_number, &column));
  read_column_array<U, V>(column, column_values);
}

int64_t FeatherFeatureCalculator::read_single_feather_file(const std::string &path) {
  auto reader = std::unique_ptr<arrow::ipc::feather::TableReader>();
  auto file = std::shared_ptr<arrow::io::MemoryMappedFile>();
  CHECK_RESULT_OK(arrow::io::MemoryMappedFile::Open(path, arrow::io::FileMode::type::READ, &file));
  CHECK_RESULT_OK(arrow::ipc::feather::TableReader::Open(file, &reader));

  int64_t initial_size = ip.size();
  read_column_array<uint32_t, arrow::UInt32Type>(reader, "ip", ip);
  read_column_array<uint16_t, arrow::UInt16Type>(reader, "app", app);
  read_column_array<uint16_t, arrow::UInt16Type>(reader, "device", device);
  read_column_array<uint16_t, arrow::UInt16Type>(reader, "os", os);
  read_column_array<uint16_t, arrow::UInt16Type>(reader, "channel", channel);
  read_column_array<uint64_t, arrow::TimestampType>(reader, "click_time", click_time);
  assert(ip.size() == app.size());
  assert(ip.size() == device.size());
  assert(ip.size() == os.size());
  assert(ip.size() == channel.size());
  assert(ip.size() == click_time.size());
  return ip.size() - initial_size;
}

int64_t FeatherFeatureCalculator::read_single_feather_file_with_stopwatch(const std::string &message, const std::string &path) {
  auto start = std::chrono::system_clock::now();
  int64_t size = read_single_feather_file(path);
  auto duration = std::chrono::duration_cast<std::chrono::minutes>(start - std::chrono::system_clock::now());
  cout << message << "(size = " << size << "): " << duration.count() << " [s]"  << endl;
  return size;
}
