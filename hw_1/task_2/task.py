import sys

def tail(file, num_lines=10):
    lines = file.readlines()
    for line in lines[-num_lines:]:
        print(line, end='')

def main():
    if len(sys.argv) > 1:
        for i, filename in enumerate(sys.argv[1:]):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    if len(sys.argv) > 2:
                        print(f"==> {filename} <==")
                    tail(file)
            except FileNotFoundError:
                print(f"Error: File '{filename}' not found.", file=sys.stderr)
                sys.exit(1)
            if i < len(sys.argv) - 2:
                print()
    else:
        tail(sys.stdin, num_lines=17)

if __name__ == "__main__":
    main()