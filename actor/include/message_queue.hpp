#include <queue>
#include <mutex>
#include <condition_variable>

template <typename Msg>
class MessageQueue {
  public:
    void push(const Msg& item) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
      }
      cv_.notify_one();
    }

    bool pop(Msg& out) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        return false;
      }
      out = queue_.front();
      queue_.pop();
      return true;
    }

    bool empty() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

  private:
    std::queue<Msg> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};
