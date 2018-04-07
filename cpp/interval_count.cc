#include <iostream>
#include <memory>
#include <cassert>
#include <arrow/ipc/feather.h>
#include <arrow/io/file.h>
#include <arrow/status.h>
#include <arrow/table.h>
#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/io/interfaces.h>
using namespace std;
using arrow::Status;
using arrow::Column;
using arrow::NumericArray;
using arrow::Int32Type;

#define CHECK_RESULT_OK(s) do { \
    Status status = (s); \
    assert(status.ok()); \
  } while(0)

int main(int argc, char **argv)
{

  if (argc != 2)
  {
    fprintf(stderr, "Usage: %s (train_feather) (comma separated column list)\n", argv[0]);
    exit(-1);
  }

  auto reader = unique_ptr<arrow::ipc::feather::TableReader>();
  auto file = shared_ptr<arrow::io::MemoryMappedFile>();
  CHECK_RESULT_OK(arrow::io::MemoryMappedFile::Open(string(argv[1]), arrow::io::FileMode::type::READ, &file));
  CHECK_RESULT_OK(arrow::ipc::feather::TableReader::Open(file, &reader));
  
  cout << reader->num_columns() << endl;
  cout << reader->num_rows() << endl;
  auto column = shared_ptr<Column>();
  CHECK_RESULT_OK(reader->GetColumn(0, &column));
  cout << column->name() << endl;

  for (auto &chunk: column->data()->chunks()) {
    auto ids = std::static_pointer_cast<NumericArray<Int32Type>>(chunk);
    cout << ids->length() << endl;
    cout << ids->Value(0) << endl;
  }
}
