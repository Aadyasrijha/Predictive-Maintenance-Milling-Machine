import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('ai4i2020_engineered.csv')

broken = df[df['Machine failure'] == 1]
normal = df[df['Machine failure'] == 0].sample(n=len(broken), random_state=42)
balanced_df = pd.concat([normal, broken]).copy()

balanced_df['Status'] = balanced_df['Machine failure'].map({0: 'Normal', 1: 'Failure'})

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

sns.scatterplot(
    data=balanced_df,
    x='Rotational speed [rpm]',
    y='Torque [Nm]',
    hue='Status',
    hue_order=['Normal', 'Failure'],
    palette={'Normal': '#2563EB', 'Failure': '#DC2626'},
    style='Status',
    style_order=['Normal', 'Failure'],
    markers={'Normal': 'o', 'Failure': 'X'},
    s=90,
    alpha=0.75
)

plt.title("Mechanical Stress Distribution: Torque vs. Rotational Speed", fontsize=14, fontweight='bold')
plt.xlabel("Rotational Speed [rpm]", fontsize=12)
plt.ylabel("Torque [Nm]", fontsize=12)
plt.legend(title="Machine Status")
plt.tight_layout()

plt.savefig('fig3_2_stress_clusters.png', dpi=300)
plt.show()