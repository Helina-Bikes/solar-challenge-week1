import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Simulate data loading or processing here
    data = pd.DataFrame({
        'Category': ['A', 'B', 'C'],
        'Value': [10, 20, 15]
    })
    return data

def plot_bar_chart(data):
    fig, ax = plt.subplots()
    ax.bar(data['Category'], data['Value'])
    ax.set_title('Sample Bar Chart')
    return fig
