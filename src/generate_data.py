import pandas as pd
import numpy as np

def generate_student_data(num_records=200):
    np.random.seed(42)

    data = {
        "attendance": np.random.randint(40, 100, num_records),
        "internal_marks": np.random.randint(10, 50, num_records),
        "assignment_score": np.random.randint(10, 50, num_records),
    }

    df = pd.DataFrame(data)

    df["result"] = np.where(
        (df["attendance"] >= 75) &
        (df["internal_marks"] >= 25) &
        (df["assignment_score"] >= 25),
        "Pass",
        "Fail"
    )

    return df

if __name__ == "__main__":
    df = generate_student_data()
    df.to_csv("data/student_data.csv", index=False)
    print("Student data generated successfully")
