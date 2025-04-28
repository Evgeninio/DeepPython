import sys

def wc(file):
    lines, words, bytes_count = 0, 0, 0
    for line in file:
        lines += 1
        words += len(line.split())
        bytes_count += len(line.encode('utf-8'))
    return lines, words, bytes_count

def print_stats(lines, words, bytes_count, filename=None):
    if filename:
        print(f"{lines}\t{words}\t{bytes_count}\t{filename}")
    else:
        print(f"{lines}\t{words}\t{bytes_count}")

def main():
    total_lines, total_words, total_bytes = 0, 0, 0
    if len(sys.argv) > 1:
        for filename in sys.argv[1:]:
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    lines, words, bytes_count = wc(file)
                    print_stats(lines, words, bytes_count, filename)
                    total_lines += lines
                    total_words += words
                    total_bytes += bytes_count
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.", file=sys.stderr)
                sys.exit(1)
        if len(sys.argv) > 2:
            print_stats(total_lines, total_words, total_bytes, "total")
    else:
        lines, words, bytes_count = wc(sys.stdin)
        print_stats(lines, words, bytes_count)

if __name__ == "__main__":
    main()
