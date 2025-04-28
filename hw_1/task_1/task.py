import sys

def number_lines(file):
    for i, line in enumerate(file, start=1):
        print(f"{i}\t{line.rstrip()}")

def main():
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], 'r', encoding='utf-8') as file:
                number_lines(file)
        except FileNotFoundError:
            print(f"Error: File '{sys.argv[1]}' not found.", file=sys.stderr)
            sys.exit(1)
    else:
        number_lines(sys.stdin)

if __name__ == "__main__":
    main()