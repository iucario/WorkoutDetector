import time
import concurrent.futures
import multiprocessing


def cpu_bound(number):
    return sum(i * i for i in range(number))


def find_sums(numbers):
    for number in numbers:
        cpu_bound(number)


def find_sums_con(numbers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(cpu_bound, numbers)


def find_sums_mul(numbers):
    with multiprocessing.Pool() as pool:
        pool.map(cpu_bound, numbers)


if __name__ == "__main__":
    numbers = [10_000_000 + x for x in range(20)]

    start_time = time.time()
    find_sums_con(numbers)
    duration = time.time() - start_time
    print(f"Concurrent Duration {duration} seconds")
    start_time = time.time()
    find_sums(numbers)
    duration2 = time.time() - start_time
    print(f"Duration {duration2} seconds")
    start_time = time.time()
    find_sums_mul(numbers)
    duration3 = time.time() - start_time
    print(f"Multiprocessing Duration {duration3} seconds")

    '''Concurrent Duration 10.035139083862305 seconds
        Duration 9.954027652740479 seconds
        Multiprocessing Duration 1.4972429275512695 seconds'''
