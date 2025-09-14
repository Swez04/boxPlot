import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_box(data, column):
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data[column], color='lightblue', flierprops=dict(marker='o', color='red', markersize=6))
    plt.title(f'Box Plot of {column}')
    plt.xlabel(column)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_interactive(data, column):
    fig = px.box(data, y=column, title=f'Interactive Box Plot of {column}')
    fig.show()