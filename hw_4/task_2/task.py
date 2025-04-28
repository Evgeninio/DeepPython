import math
import concurrent.futures
import time
import multiprocessing

def integrate_part(f, a, b, n_iter):
    acc = 0
    step = (b - a) / n_iter
    for i in range(n_iter):
        acc += f(a + i * step) * step
    return acc

def integrate(f, a, b, *, n_jobs=1, n_iter=10000000, executor_class=concurrent.futures.ThreadPoolExecutor):
    if n_jobs == 1:
        return integrate_part(f, a, b, n_iter)

    with executor_class(max_workers=n_jobs) as executor:
        step = (b - a) / n_jobs
        futures = [executor.submit(integrate_part, f, a + i * step, a + (i + 1) * step, n_iter // n_jobs) for i in range(n_jobs)]
        return sum(future.result() for future in futures)


if __name__ == '__main__':
    cpu_num = multiprocessing.cpu_count()

    n_jobs_range = range(1, cpu_num * 2 + 1)

    thread_times = []
    process_times = []

    for n_jobs in n_jobs_range:
        start_time = time.time()
        integrate(math.cos, 0, math.pi / 2, n_jobs=n_jobs)
        thread_times.append(time.time() - start_time)

        start_time = time.time()
        integrate(math.cos, 0, math.pi / 2, n_jobs=n_jobs, executor_class=concurrent.futures.ProcessPoolExecutor)
        process_times.append(time.time() - start_time)

    with open("comparison.txt", "w") as f:
        f.write("n_jobs\tThread Time\tProcess Time\n")
        for i, n_jobs in enumerate(n_jobs_range):
            f.write(f"{n_jobs}\t{thread_times[i]}\t{process_times[i]}\n")

    print("n_jobs\tThread Time\tProcess Time")
    for i, n_jobs in enumerate(n_jobs_range):
        print(f"{n_jobs}\t{thread_times[i]}\t{process_times[i]}")