import matplotlib.pyplot as plt
import pandas as pd
# Data from CSV
data = {
    "DESCRIPTION": ["Yes", "No", "Unknown"],
    "frequency": [37671, 67892 + 17, 4508]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Create the pie chart
plt.figure(figsize=(8, 8))
plt.pie(df["frequency"], labels=df["DESCRIPTION"], autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff6666", "#99ff99", "#ffcc99"])
plt.title("Frequency of Death by Diabetes by Economic Activity (2023)")
plt.savefig('pruebas/arreglo_datos/005/economic_activity_chart.png')


# Data from CSV for education level
data_education = {
    "DESCRIPTION": ["No education", "Preschool", "Incomplete primary", "Complete primary", "Incomplete secondary", "Complete secondary", "Incomplete high school", "Complete high school", "Professional", "Postgraduate", "Not applicable to children under 3 years", "Not specified"],
    "frequency": [18920, 99, 27415, 28421, 2232, 13763, 1492, 6605, 7665, 410, 13, 3053]
}

# Create a DataFrame for education level
df_education = pd.DataFrame(data_education)

# Create the stacked bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_education["DESCRIPTION"], df_education["frequency"], color=["#4c72b0", "#55a868", "#c44e52", "#8172b3", "#ccb974", "#64b5cd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#ff7f0e"])
plt.xticks(rotation=90)
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.title("Distribution of Education Level in 2023")
plt.savefig('pruebas/arreglo_datos/005/education_level_chart.png')


# Data from CSV for performance metrics
data_metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Value": [0.9998, 0.92, 1.00, 0.96]
}

# Create a DataFrame for performance metrics
df_metrics = pd.DataFrame(data_metrics)

# Create scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(df_metrics["Metric"], df_metrics["Value"], color='red', s=100, label="Actual Values")
plt.ylim(0.65, 1.05)
plt.xlabel("Metric")
plt.ylabel("Value")
plt.title("Model Performance Metrics")
plt.axhline(y=0.7, color='green', linestyle='--', label="Lower Acceptable Limit")
plt.axhline(y=1.0, color='blue', linestyle='--', label="Ideal Value")
plt.grid(True)
plt.legend()
plt.savefig('pruebas/arreglo_datos/005/scatter_plot.png')
