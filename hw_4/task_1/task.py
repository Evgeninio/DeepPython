import time
import threading
import multiprocessing

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def run_sync(n, num_runs=10):
    start_time = time.time()
    for _ in range(num_runs):
        fibonacci(n)
    end_time = time.time()
    return end_time - start_time

def run_threaded(n, num_threads=10):
    threads = []
    start_time = time.time()
    for _ in range(num_threads):
        thread = threading.Thread(target=fibonacci, args=(n,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    end_time = time.time()
    return end_time - start_time

def run_multiprocess(n, num_processes=10):
    processes = []
    start_time = time.time()
    for _ in range(num_processes):
        process = multiprocessing.Process(target=fibonacci, args=(n,))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    end_time = time.time()
    return end_time - start_time


if __name__ == "__main__":
    n = 30
    num_runs = 10

    sync_time = run_sync(n, num_runs)
    threaded_time = run_threaded(n, num_runs)
    multiprocess_time = run_multiprocess(n, num_runs)



    with open("performance_results.txt", "w") as f:
        f.write(f"Synchronous time: {sync_time:.4f} seconds\n")
        f.write(f"Threaded time: {threaded_time:.4f} seconds\n")
        f.write(f"Multiprocess time: {multiprocess_time:.4f} seconds\n")

    print(f"Synchronous time: {sync_time:.4f} seconds")
    print(f"Threaded time: {threaded_time:.4f} seconds")
    print(f"Multiprocess time: {multiprocess_time:.4f} seconds")