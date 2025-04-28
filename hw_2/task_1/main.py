from task import generate_latex_table

def main():
    data = [
        ["Name", "Age", "City"],
        ["Alice", "24", "New York"],
        ["Bob", "30", "Los Angeles"],
        ["Charlie", "28", "Chicago"]
    ]
    
    table_latex = generate_latex_table(data)

    document = (
        "\\documentclass{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\begin{document}\n"
        + table_latex +
        "\n\\end{document}"
    )
    
    with open("example_table.tex", "w", encoding="utf-8") as f:
        f.write(document)

if __name__ == "__main__":
    main()
