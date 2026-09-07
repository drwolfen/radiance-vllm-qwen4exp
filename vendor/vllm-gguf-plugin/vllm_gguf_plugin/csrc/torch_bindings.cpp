// R9V modification: batched Linux mmap page advice for SSD-backed PLE.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <optional>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#include <Python.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

Tensor ggml_dequantize(Tensor W, int64_t type, int64_t m, int64_t n,
                       std::optional<ScalarType> dtype);
Tensor ggml_mul_mat_vec_a8(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_mul_mat_a8(Tensor W, Tensor X, int64_t type, int64_t row);
Tensor ggml_moe_a8(Tensor X, Tensor W, Tensor sorted_token_ids,
                   Tensor expert_ids, Tensor num_tokens_post_padded,
                   int64_t type, int64_t row, int64_t top_k, int64_t tokens);
Tensor ggml_moe_a8_vec(Tensor X, Tensor W, Tensor topk_ids, int64_t top_k,
                       int64_t type, int64_t row, int64_t tokens);
int64_t ggml_moe_get_block_size(int64_t type);

STABLE_TORCH_LIBRARY(_C_gguf, ops) {
  ops.def(
      "ggml_dequantize(Tensor W, int type, SymInt m, SymInt n, ScalarType? "
      "dtype) -> Tensor");
  ops.def(
      "ggml_mul_mat_vec_a8(Tensor W, Tensor X, int type, SymInt row) "
      "-> Tensor");
  ops.def(
      "ggml_mul_mat_a8(Tensor W, Tensor X, int type, SymInt row) -> Tensor");
  ops.def(
      "ggml_moe_a8(Tensor X, Tensor W, "
      "Tensor sorted_token_ids, Tensor expert_ids, Tensor "
      "num_tokens_post_padded, "
      "int type, SymInt row, SymInt top_k, SymInt tokens) -> Tensor");
  ops.def(
      "ggml_moe_a8_vec(Tensor X, Tensor W, "
      "Tensor topk_ids, int top_k, "
      "int type, SymInt row, SymInt tokens) -> Tensor");
  ops.def("ggml_moe_get_block_size(int type) -> int");
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CUDA, ops) {
  ops.impl("ggml_dequantize", TORCH_BOX(&ggml_dequantize));
  ops.impl("ggml_mul_mat_vec_a8", TORCH_BOX(&ggml_mul_mat_vec_a8));
  ops.impl("ggml_mul_mat_a8", TORCH_BOX(&ggml_mul_mat_a8));
  ops.impl("ggml_moe_a8", TORCH_BOX(&ggml_moe_a8));
  ops.impl("ggml_moe_a8_vec", TORCH_BOX(&ggml_moe_a8_vec));
}

STABLE_TORCH_LIBRARY_IMPL(_C_gguf, CompositeExplicitAutograd, ops) {
  ops.impl("ggml_moe_get_block_size", TORCH_BOX(&ggml_moe_get_block_size));
}

static PyObject* mmap_readahead(PyObject*, PyObject* args) {
#if defined(__linux__)
  unsigned long long address = 0;
  unsigned long long nbytes = 0;
  unsigned long long row_nbytes = 0;
  unsigned long long num_rows = 0;
  PyObject* row_ids_object = nullptr;
  if (!PyArg_ParseTuple(args, "KKKKO", &address, &nbytes, &row_nbytes,
                        &num_rows, &row_ids_object)) {
    return nullptr;
  }
  if (address == 0 || nbytes == 0 || row_nbytes == 0 || num_rows == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "mmap readahead arguments must all be positive");
    return nullptr;
  }
  const long page_size_long = sysconf(_SC_PAGESIZE);
  if (page_size_long <= 0) {
    PyErr_SetString(PyExc_RuntimeError, "sysconf(_SC_PAGESIZE) failed");
    return nullptr;
  }
  const uint64_t page_size = static_cast<uint64_t>(page_size_long);
  if (address % page_size != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "mmap readahead requires a page-aligned address");
    return nullptr;
  }

  const Py_ssize_t row_count = PySequence_Size(row_ids_object);
  if (row_count < 0) {
    PyErr_SetString(PyExc_TypeError,
                    "mmap readahead row IDs must be a sequence");
    return nullptr;
  }
  std::vector<uint64_t> pages;
  pages.reserve(static_cast<size_t>(row_count) * 2);
  for (Py_ssize_t index = 0; index < row_count; ++index) {
    PyObject* row_object = PySequence_GetItem(row_ids_object, index);
    if (row_object == nullptr) {
      return nullptr;
    }
    const long long row = PyLong_AsLongLong(row_object);
    Py_DECREF(row_object);
    if (row == -1 && PyErr_Occurred()) {
      return nullptr;
    }
    if (row < 0 || static_cast<uint64_t>(row) >= num_rows) {
      PyErr_Format(PyExc_IndexError,
                   "PLE mmap row %lld is outside [0, %llu)", row, num_rows);
      return nullptr;
    }
    const uint64_t row_start = static_cast<uint64_t>(row) * row_nbytes;
    if (row_start >= nbytes || row_nbytes - 1 > nbytes - row_start - 1) {
      PyErr_SetString(PyExc_ValueError,
                      "PLE mmap row geometry exceeds the mapped byte range");
      return nullptr;
    }
    const uint64_t first_page = row_start / page_size;
    const uint64_t final_page = (row_start + row_nbytes - 1) / page_size;
    for (uint64_t page = first_page; page <= final_page; ++page) {
      pages.push_back(page);
    }
  }
  if (pages.empty()) {
    return Py_BuildValue("(KKK)", 0ULL, 0ULL, 0ULL);
  }

  std::sort(pages.begin(), pages.end());
  pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
  uint64_t advised_bytes = 0;
  uint64_t range_start = pages.front();
  uint64_t previous = range_start;
  std::vector<struct iovec> advice_ranges;
  advice_ranges.reserve(pages.size());
  auto append_range = [&](uint64_t first_page, uint64_t end_page) {
    const uint64_t offset = first_page * page_size;
    const uint64_t end =
        std::min(end_page * page_size, static_cast<uint64_t>(nbytes));
    const uint64_t length = end - offset;
    advice_ranges.push_back(
        {reinterpret_cast<void*>(address + offset), static_cast<size_t>(length)});
    advised_bytes += length;
  };
  for (size_t index = 1; index < pages.size(); ++index) {
    const uint64_t page = pages[index];
    if (page != previous + 1) {
      append_range(range_start, previous + 1);
      range_start = page;
    }
    previous = page;
  }
  append_range(range_start, previous + 1);

  bool batch_complete = false;
#if defined(SYS_pidfd_open) && defined(SYS_process_madvise)
  const int pidfd =
      static_cast<int>(syscall(SYS_pidfd_open, static_cast<int>(getpid()), 0));
  if (pidfd >= 0) {
    const long configured_iov_max = sysconf(_SC_IOV_MAX);
    const size_t iov_max = configured_iov_max > 0
                               ? static_cast<size_t>(configured_iov_max)
                               : static_cast<size_t>(1024);
    batch_complete = true;
    for (size_t start = 0; start < advice_ranges.size(); start += iov_max) {
      const size_t count = std::min(iov_max, advice_ranges.size() - start);
      uint64_t expected_bytes = 0;
      for (size_t index = start; index < start + count; ++index) {
        expected_bytes += advice_ranges[index].iov_len;
      }
      const long advised = syscall(SYS_process_madvise, pidfd,
                                   advice_ranges.data() + start, count,
                                   MADV_WILLNEED, 0);
      if (advised < 0 || static_cast<uint64_t>(advised) != expected_bytes) {
        batch_complete = false;
        break;
      }
    }
    close(pidfd);
  }
#endif
  if (!batch_complete) {
    for (const struct iovec& range : advice_ranges) {
      if (madvise(range.iov_base, range.iov_len, MADV_WILLNEED) != 0) {
        PyErr_SetFromErrno(PyExc_OSError);
        return nullptr;
      }
    }
  }
  return Py_BuildValue("(KKK)", static_cast<unsigned long long>(pages.size()),
                       static_cast<unsigned long long>(advice_ranges.size()),
                       static_cast<unsigned long long>(advised_bytes));
#else
  PyErr_SetString(PyExc_NotImplementedError,
                  "mmap readahead is only supported on Linux");
  return nullptr;
#endif
}

static PyMethodDef _module_methods[] = {
    {"mmap_readahead", mmap_readahead, METH_VARARGS,
     "Submit merged MADV_WILLNEED ranges for packed mmap rows."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef _module_def = {
    PyModuleDef_HEAD_INIT, "_C_gguf", nullptr, -1, _module_methods,
};

extern "C" PyObject* PyInit__C_gguf(void) {
  return PyModule_Create(&_module_def);
}
