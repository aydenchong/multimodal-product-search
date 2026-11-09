import pandas as pd

df = pd.read_csv(
    "experiments/clip_matching_results.csv"
)

errors = df[df["correct"] == False].copy()

print("Total samples:", len(df))
print("Total errors:", len(errors))

print("\nIncorrect Matches:")
print(
    errors[
        [
            "product_id",
            "true_name",
            "predicted_name",
        ]
    ].to_string(index=False)
)

errors.to_csv(
    "experiments/clip_matching_errors.csv",
    index=False,
)

print("\nSaved: experiments/clip_matching_errors.csv")
