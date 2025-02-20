#!/usr/bin/env python3
import os
import json
import src.utils as utils

def main():
    # Where the JSON file was saved
    output_path = os.path.join(
        utils.get_path("DATA_PATH"),
        "0_paper_plots",
        "intermediate_data"
    )
    json_path = os.path.join(output_path, "steering_vector_stability_results.json")
    output_tex_path = os.path.join(output_path, "steering_vector_stability_table.tex")
    
    if not os.path.isfile(json_path):
        print(f"JSON file not found: {json_path}")
        return
    
    # We define the 7 columns we want, in the exact order.
    # These keys correspond exactly to the pretty names stored in the JSON.
    desired_scenario_order = [
        "prefilled",              # prefilled
        "instruction",            # instruction
        "5-shot",                 # 5-shot
        "prefilled instruction",  # prefilled instruction
        "prefilled 5-shot",       # prefilled 5-shot
        "instruction 5-shot",     # instruction 5-shot
        "prefilled instruction 5-shot"  # prefilled instruction 5-shot
    ]
    
    # 1. Load the JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # The structure of the JSON is:
    # {
    #   "analysis": [
    #       {
    #         "dataset_name": "...",
    #         "prompt_types": {
    #            "prefilled": { "mean_cosine_similarity": ...,
    #                           "std_cosine_similarity": ... },
    #            "instruction": { ... }, etc.
    #         }
    #       },
    #       ...
    #   ]
    # }
    
    # 2. Build the top row: "Index" + the 7 columns in fixed order.
    header_cells = ["Index"] + desired_scenario_order
    
    # 3. Construct the table rows (one row per dataset)
    rows = []
    # Instead of using the dataset name, we enumerate the datasets from 1 to N.
    for idx, entry in enumerate(data["analysis"], start=1):
        # Use the dataset index as the first cell
        row_cells = [str(idx)]
        pt_dict = entry.get("prompt_types", {})
        
        # Then fill in each of the 7 scenario columns
        for key in desired_scenario_order:
            if key not in pt_dict:
                row_cells.append("--")
            else:
                stats = pt_dict[key]
                mean_cs = stats["mean_cosine_similarity"]
                std_cs = stats["std_cosine_similarity"]
                # Format as "$0.123 \pm 0.045$" to ensure math mode for \pm and numbers.
                cell_str = f"${mean_cs:.3f} \\pm {std_cs:.3f}$"
                row_cells.append(cell_str)
        
        rows.append(row_cells)
    
    # 4. Create LaTeX lines.
    # We have 1 (index) + 7 (prompt types) = 8 columns total.
    num_cols = 8
    # "l" for the first column and "c" for the next 7 columns.
    alignment_str = "l" + "c" * (num_cols - 1)
    
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{" + alignment_str + "}")
    lines.append("\\toprule")
    
    # Header row
    header_line = " & ".join(header_cells) + " \\\\"
    lines.append(header_line)
    lines.append("\\midrule")
    
    # Each data row
    for row_cells in rows:
        row_line = " & ".join(row_cells) + " \\\\"
        lines.append(row_line)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    # 5. Write to the .tex file
    with open(output_tex_path, "w") as f_out:
        for line in lines:
            f_out.write(line + "\n")
    
    print(f"LaTeX table saved to {output_tex_path}")

if __name__ == "__main__":
    main()
