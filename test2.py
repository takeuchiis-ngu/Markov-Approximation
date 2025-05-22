import threading
import time
import random
from decimal import Decimal, getcontext

# 十分な精度を確保（12桁以上）
getcontext().prec = 20

# タイマー情報格納用
original_values = []
remaining_times = []
timer_threads = []
lock = threading.Lock()
stop_flag = False
start_time = None

def countdown_timer(index):
    global stop_flag
    while not stop_flag:
        with lock:
            elapsed = Decimal(time.perf_counter()) - start_time
            remaining = original_values[index] - elapsed
            if remaining <= Decimal('0.000000001'):
                print(f"⏰ Timer {index} reached 0. Original value: {original_values[index]} seconds")
                stop_flag = True
                break
            remaining_times[index] = remaining
        time.sleep(0.001)  # 1ms 更新間隔

def main():
    global start_time, stop_flag
    timer_index = 0

    while not stop_flag:
        # ランダムな小数（0.5〜3秒など）で生成
        timer_value = Decimal(str(round(random.uniform(0.001, 0.04), 9)))
        original_values.append(timer_value)
        remaining_times.append(timer_value)
        print(f"⏳ Timer {timer_index} created with {timer_value} seconds")

        if timer_index == 0:
            start_time = Decimal(str(time.perf_counter()))  # 高精度の開始時刻

        t = threading.Thread(target=countdown_timer, args=(timer_index,))
        t.start()
        timer_threads.append(t)

        timer_index += 1
        # time.sleep(1)  # 1秒ごとにタイマーを追加

    # すべてのスレッドを待機
    for t in timer_threads:
        t.join()

if __name__ == "__main__":
    main()
