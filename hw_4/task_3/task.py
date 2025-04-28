import multiprocessing
import time
import codecs

def process_a(input_queue, output_queue):
    while True:
        try:
            message = input_queue.get()
            time.sleep(5)
            output_queue.put(message.lower())
        except KeyboardInterrupt:
            break

def process_b(input_queue, output_queue):
    while True:
        try:
            message = input_queue.get()
            encoded_message = codecs.encode(message, 'rot_13')
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            output_str = f"[{current_time}] Encoded message: {encoded_message}"
            print(output_str)
            output_queue.put(output_str)
        except KeyboardInterrupt:
            break

def main():
    input_queue_a = multiprocessing.Queue()
    output_queue_a = multiprocessing.Queue()
    output_queue_b = multiprocessing.Queue()
    
    process_a_instance = multiprocessing.Process(target=process_a, args=(input_queue_a, output_queue_a))
    process_b_instance = multiprocessing.Process(target=process_b, args=(output_queue_a, output_queue_b))
    
    process_a_instance.start()
    process_b_instance.start()

    with open("interaction_log.txt", "w") as log_file:
        try:
            while True:
                user_input = input("Enter message (or type 'exit' to quit): ")
                if user_input.lower() == 'exit':
                    break
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_file.write(f"[{current_time}] User input: {user_input}\n")
                input_queue_a.put(user_input)

                result_from_b = output_queue_b.get()
                log_file.write(f"{result_from_b}\n")

        except KeyboardInterrupt:
            pass
        finally:

            process_a_instance.terminate()
            process_b_instance.terminate()
            process_a_instance.join()
            process_b_instance.join()
            print("Processes terminated.")



if __name__ == '__main__':
    main()