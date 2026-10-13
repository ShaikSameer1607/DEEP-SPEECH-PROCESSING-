import os
import pandas as pd


os.makedirs("results", exist_ok=True)


results = [
    {
        "Task": "Spoof Detection",
        "Dataset": "ASVspoof 2019 LA Evaluation",
        "Model": "AASIST + 25-D Probabilistic Attributes + Decision Tree",
        "Accuracy": "99.07%",
        "Notes": "Bonafide vs Spoof"
    },
    {
        "Task": "Attack Attribution",
        "Dataset": "ASVspoof 2019 LA Development",
        "Model": "25-D Probabilistic Attributes + Decision Tree",
        "Accuracy": "98.57%",
        "Notes": "Known attacks A01-A06"
    },
    {
        "Task": "Attack Attribution",
        "Dataset": "Eval*",
        "Model": "25-D Probabilistic Attributes + Decision Tree",
        "Accuracy": "98.83%",
        "Notes": "A16→A04 and A19→A06 reference-attack protocol"
    },
    {
        "Task": "SHAP Analysis",
        "Dataset": "Eval* individual samples",
        "Model": "Decision Tree + SHAP TreeExplainer",
        "Accuracy": "N/A",
        "Notes": "A16→A04 and A19→A06 explanations"
    }
]


df = pd.DataFrame(results)

output_file = "results/final_results.xlsx"

df.to_excel(
    output_file,
    index=False
)

print("Final results table created:")
print(output_file)

print("\n")
print(df.to_string(index=False))