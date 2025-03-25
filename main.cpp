#include "ticktock.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <oneapi/tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/spin_mutex.h>
#include <thread>
#include <vector>

// TODO: 并行化所有这些 for 循环
// 计算密集型
template <class T, class Func>
std::vector<T> fill(std::vector<T> &arr, Func const &func) {
  TICK(fill);
  // for (size_t i = 0; i < arr.size(); i++) {
  //   arr[i] = func(i);
  // }
  tbb::parallel_for(tbb::blocked_range<size_t>(0, arr.size()),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        arr[i] = func(i);
                      }
                    });
  TOCK(fill);
  return arr;
}

template <class T> void saxpy(T a, std::vector<T> &x, std::vector<T> const &y) {
  TICK(saxpy);
  // for (size_t i = 0; i < x.size(); i++) {
  //   x[i] = a * x[i] + y[i];
  // }
  const int n = x.size();
  tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i != r.end(); ++i) {
                        x[i] = a * x[i] + y[i];
                      }
                    });

  TOCK(saxpy);
}

template <class T> T sqrtdot(std::vector<T> const &x, std::vector<T> const &y) {
  TICK(sqrtdot);
  T ret = 0;
  // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
  //   ret += x[i] * y[i];
  // }
  const int n = std::min(x.size(), y.size());
  ret = tbb::parallel_reduce(
      // define the range
      tbb::blocked_range<size_t>(0, n), (T)0,
      [&](tbb::blocked_range<size_t> r, T local_res) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          local_res = x[i] + y[i];
        }
        return local_res;
      },
      [](T x, T y) { return x + y; });
  ret = std::sqrt(ret);
  TOCK(sqrtdot);
  return ret;
}

template <class T> T minvalue(std::vector<T> const &x) {
  TICK(minvalue);
  T ret = x[0];
  // for (size_t i = 1; i < x.size(); i++) {
  //   if (x[i] < ret)
  //     ret = x[i];
  // }
  const int n = x.size();
  ret = tbb::parallel_reduce(
      // define the range
      tbb::blocked_range<size_t>(0, n), (T)0,
      [&](tbb::blocked_range<size_t> r, T local_res) {
        for (size_t i = r.begin(); i != r.end(); ++i) {
          local_res = x[i];
        }
        return local_res;
      },
      [](T x, T y) { return std::min(x, y); });
  TOCK(minvalue);
  return ret;
}

template <class T>
std::vector<T> magicfilter(std::vector<T> const &x, std::vector<T> const &y) {
  TICK(magicfilter);
  std::vector<T> res;
  // for (size_t i = 0; i < std::min(x.size(), y.size()); i++) {
  //   if (x[i] > y[i]) {
  //     res.push_back(x[i]);
  //   } else if (y[i] > x[i] && y[i] > 0.5f) {
  //     res.push_back(y[i]);
  //     res.push_back(x[i] * y[i]);
  //   }
  // }
  // --------------
  // const int n = std::min(x.size(), y.size());
  // std::mutex mtx; // 这里其实也可以换成 tbb::spin_mutex
  // res.reserve(n * 2 / 3);
  // tbb::parallel_for(
  //     tbb::blocked_range<size_t>(0, n), [&](tbb::blocked_range<size_t> r) {
  //       std::vector<float> local_a;
  //       local_a.reserve(r.size());
  //       for (size_t i = r.begin(); i < r.end(); i++) {
  //         if (x[i] > y[i]) {
  //           local_a.push_back(x[i]);
  //         } else if (y[i] > x[i] && y[i] > 0.5f) {
  //           local_a.push_back(y[i]);
  //           local_a.push_back(x[i] * y[i]);
  //         }
  //       }
  //       std::lock_guard lck(mtx);
  //       // move
  //       std::move(local_a.begin(), local_a.end(), std::back_inserter(res));
  //       // std::copy(local_a.begin(), local_a.end(),
  //       std::back_inserter(res));
  //     });
  // --------------
  std::mutex mtx;
  // create threads to do the job
  const int thread_num = 14;
  const int N = x.size();
  std::condition_variable cv;
  std::vector<std::thread> threads;

  int finished_threads = 0;

  for (int i = 0; i < thread_num; i++) {
    int idx = N / thread_num * i;
    threads.push_back(std::thread([&, idx]() {
      std::vector<T> local_a;
      local_a.reserve(N / thread_num);
      for (int j = idx; j < std::min(idx + N / thread_num, N); j++) {
        if (x[j] > y[j]) {
          local_a.push_back(x[j]);
        } else if (y[j] > x[j] && y[j] > 0.5f) {
          local_a.push_back(y[j]);
          local_a.push_back(x[j] * y[j]);
        }
      }

      {
        std::unique_lock<std::mutex> lck(mtx);
        res.insert(res.end(), std::make_move_iterator(local_a.begin()),
                   std::make_move_iterator(local_a.end()));
        if (++finished_threads == thread_num)
          cv.notify_one(); // 通知主线程
      }
    }));
  }

  // 主线程等待所有工作线程完成
  std::unique_lock<std::mutex> lck(mtx);
  cv.wait(lck, [&]() { return finished_threads == thread_num; });

  // joins
  for (auto &t : threads) {
    t.join();
  }

  TOCK(magicfilter);
  return res;
}

template <class T> T scanner(std::vector<T> &x) {
  TICK(scanner);
  T ret = 0;
  // for (size_t i = 0; i < x.size(); i++) {
  //   ret += x[i];
  //   x[i] = ret;
  // }
  const int n = x.size();
  ret = tbb::parallel_scan(
      tbb::blocked_range<size_t>(0, n), (T)(0),
      [&](const tbb::blocked_range<size_t> &r, T init, bool is_final_scan) {
        T res = init;
        for (size_t i = r.begin(); i != r.end(); ++i) {
          res += x[i];
          if (is_final_scan) {
            x[i] = res;
          }
        }
        return res;
      },
      [](T x, T y) { return x + y; });
  TOCK(scanner);
  return ret;
}

int main() {
  size_t n = 1 << 26;
  std::vector<float> x(n);
  std::vector<float> y(n);

  fill(x, [&](size_t i) { return std::sin(i); });
  fill(y, [&](size_t i) { return std::cos(i); });

  saxpy(0.5f, x, y);

  std::cout << sqrtdot(x, y) << std::endl;
  std::cout << minvalue(x) << std::endl;

  auto arr = magicfilter(x, y);
  std::cout << arr.size() << std::endl;

  scanner(x);
  std::cout << std::reduce(x.begin(), x.end()) << std::endl;

  return 0;
}
