import pandas as pd
import matplotlib.pyplot as plt

file_name = '../sets/dataset.csv'
df = pd.read_csv(file_name)

disease_counts = df['diseases'].value_counts()

mean = disease_counts.mean()
std_dev = disease_counts.std()

print(f"Ortalama: {mean:.2f}")
print(f"Standart Sapma: {std_dev:.2f}")

if std_dev / mean < 0.5:
    print("Dataset dengeli görünüyor.")
else:
    print("Dataset dengesiz olabilir.")

plt.figure(figsize=(10, 6))
plt.hist(disease_counts, bins=30, color='skyblue', edgecolor='black')
plt.title('Disease Counts Distribution', fontsize=16)
plt.xlabel('Count of Diseases', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.show()

disease_counts.to_csv('../sets/disease_counts.csv', header=['Count'])