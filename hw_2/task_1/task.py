from typing import List

def generate_latex_table(data: List[List[str]]) -> str:
    if not data or not all(isinstance(row, list) for row in data):
        raise ValueError("Input must be a non-empty list of lists.")
    
    # Определение количества колонок
    num_columns = len(data[0])
    if any(len(row) != num_columns for row in data):
        raise ValueError("All rows must have the same number of columns.")
    
    # Начало таблицы
    header = "\\begin{tabular}{" + "|".join(["c"] * num_columns) + "}\n\\hline\n"
    
    # Формируем строки таблицы
    rows = map(lambda row: " & ".join(row) + " \\\\ \\hline\n", data)
    
    # Конец таблицы
    footer = "\\end{tabular}"
    
    return header + "".join(rows) + footer
