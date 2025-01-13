import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, HTML
import scipy.stats
import statistics as stats
import seaborn as sns
from matplotlib.patches import Patch
import holoviews as hv
import hvplot.pandas

import panel as pn

# Initialize Panel
# Load CSS
pn.extension('tabulator',
    raw_css=[
        """
        .custom-dashboard {
            background-color: #f8f9fa;
            border: 2px solid #ccc;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }
        """
    ]
)

# Load the dataset
file_path = "Data_Wrangling.csv"
viet_housing = pd.read_csv(file_path)

# defining the price by property type

price_of_all= viet_housing['price']
price_of_house = viet_housing['price'][viet_housing['Type'] == 'House']
price_of_apartment = viet_housing['price'][viet_housing['Type'] == 'Apartment']
price_of_land = viet_housing['price'][viet_housing['Type'] == 'Land']
price_of_villa = viet_housing['price'][viet_housing['Type'] == 'Villa']


# defining the price in number of bedroom for house
price_of_house_1 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 1)]
price_of_house_2 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 2)]
price_of_house_3 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 3)]
price_of_house_4 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 4)]
price_of_house_5 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 5)]
price_of_house_6 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 6)]
price_of_house_7 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 7)]
price_of_house_8 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 8)]
price_of_house_9 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 9)]


# defining the price in number of bedroom for apartment
price_of_apartment_1 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 1)]
price_of_apartment_2 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 2)]
price_of_apartment_3 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 3)]
price_of_apartment_4 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 4)]
price_of_apartment_5 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 5)]
price_of_apartment_6 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 6)]
price_of_apartment_7 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 7)]


# defining the price in number of bedroom for land
price_of_land_1 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 1)]
price_of_land_2 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 2)]
price_of_land_3 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 3)]
price_of_land_4 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 4)]
price_of_land_5 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 5)]
price_of_land_6 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 6)]
price_of_land_7 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 7)]
price_of_land_8 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 8)]
price_of_land_9 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 9)]


# defining the price number in of bedroom for villa
price_of_villa_1 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 1)]
price_of_villa_2 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 2)]
price_of_villa_3 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 3)]
price_of_villa_4 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 4)]
price_of_villa_5 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 5)]
price_of_villa_6 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 6)]
price_of_villa_7 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 7)]
price_of_villa_8 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 8)]


# Plot 1
# visualisation for price vs house type
# Overall price distribution
mean_overall = np.mean(price_of_all)
median_overall = np.median(price_of_all)
mode_overall = stats.mode(price_of_all)
plt.figure(figsize = (9, 9))
sns.histplot(price_of_all, kde=False, color='red')
sns.histplot(price_of_all, color='red')
plt.axvline(mean_overall, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_overall:.2f}')
plt.axvline(median_overall, color='red', linestyle='--', linewidth=2, label=f'Median: {median_overall:.2f}')
plt.axvline(mode_overall, color='blue', linestyle='--', linewidth=2, label=f'Mode: {mode_overall:.2f}')
plt.legend(loc = 'upper right')
plt.xlim(0,None)
plt.title('Overall Price distribution/density')

overall_price_distribution = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory


# Plot 2
# Number of properties by type

type_counts = viet_housing['Type'].value_counts()

plt.figure(figsize=(9, 9))
sns.barplot(x=type_counts.index, y=type_counts.values, palette='Reds')

for i, count in enumerate(type_counts.values):
    plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)

plt.title('Overall Distribution of Property Types')
plt.xlabel('Property Type')
plt.ylabel('Count')
plt.xticks()

number_of_properties_by_type = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory


# Plot 3
# House price distribution
mean_house = np.mean(price_of_house)
median_house = np.median(price_of_house)
mode_house = stats.mode(price_of_house)
plt.figure(figsize = (9, 9))
sns.histplot(price_of_house, kde=False, color='green')
sns.histplot(price_of_house, color='green')
plt.axvline(mean_house, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_house:.2f}')
plt.axvline(median_house, color='red', linestyle='--', linewidth=2, label=f'Median: {median_house:.2f}')
plt.axvline(mode_house, color='blue', linestyle='--', linewidth=2, label=f'Mode: {mode_house:.2f}')
plt.legend(loc = 'upper right')
plt.xlim(0,None)
plt.title('House Price distribution/density')

house_price_distribution = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory


# Plot 4
# Apartment price distribution
mean_apartment = np.mean(price_of_apartment)
median_apartment = np.median(price_of_apartment)
mode_apartment = stats.mode(price_of_apartment)
plt.figure(figsize = (9, 9))
sns.histplot(price_of_apartment, kde=False, color='blue')
sns.histplot(price_of_apartment, color='blue')
plt.axvline(mean_apartment, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_apartment:.2f}')
plt.axvline(median_apartment, color='red', linestyle='--', linewidth=2, label=f'Median: {median_apartment:.2f}')
plt.axvline(mode_apartment, color='blue', linestyle='--', linewidth=2, label=f'Mode: {mode_apartment:.2f}')
plt.legend(loc = 'upper right')
plt.xlim(0,None)
plt.title('Apartment Price distribution/density')

apartment_price_distribution = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory


# Plot 5
# Land price distribution
mean_land = np.mean(price_of_land)
median_land = np.median(price_of_land)
mode_land = stats.mode(price_of_land)
plt.figure(figsize = (9, 9))
sns.histplot(price_of_land, kde=False, color='orange')
sns.histplot(price_of_land, color='orange')
plt.axvline(mean_land, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_land:.2f}')
plt.axvline(median_land, color='red', linestyle='--', linewidth=2, label=f'Median: {median_land:.2f}')
plt.axvline(mode_land, color='blue', linestyle='--', linewidth=2, label=f'Mode: {mode_land:.2f}')
plt.legend(loc = 'upper right')
plt.xlim(0,None)
plt.title('Land Price distribution/density')

land_price_distribution = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory


# Plot 6
# Villa price distribution
mean_villa = np.mean(price_of_villa)
median_villa = np.median(price_of_villa)
mode_villa = stats.mode(price_of_villa)
plt.figure(figsize = (9, 9))
sns.histplot(price_of_villa, kde=False, color='purple')
sns.histplot(price_of_villa, color='purple')
plt.axvline(mean_villa, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_villa:.2f}')
plt.axvline(median_villa, color='red', linestyle='--', linewidth=2, label=f'Median: {median_villa:.2f}')
plt.axvline(mode_villa, color='blue', linestyle='--', linewidth=2, label=f'Mode: {mode_villa:.2f}')
plt.legend(loc = 'upper right')
plt.xlim(0,None)
plt.title('Villa Price distribution/density')

villa_price_distribution = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close()  # Close the figure to free memory



# violin plot, which could display all types of properties in one graph with the same y axis
# reference from: https://matplotlib.org/stable/gallery/statistics/customized_violin.html#sphx-glr-gallery-statistics-customized-violin-py

import matplotlib.pyplot as plt
import numpy as np


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


price_of_house = viet_housing['price'][viet_housing['Type'] == 'House']
price_of_apartment = viet_housing['price'][viet_housing['Type'] == 'Apartment']
price_of_land = viet_housing['price'][viet_housing['Type'] == 'Land']
price_of_villa = viet_housing['price'][viet_housing['Type'] == 'Villa']


data1 = sorted(price_of_house)
data2 = sorted(price_of_apartment)
data3 = sorted(price_of_land)
data4 = sorted(price_of_villa)

data = [data1, data2, data3, data4]

fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title('Price distribution by property type')
ax.set_xlabel('Property type')
ax.set_ylabel('Price')


colors = ['#D43F3A', '#FF8C00', '#1E90FF', '#32CD32']

parts = ax.violinplot(
        data, showmeans=False, showmedians=False,
        showextrema=False)

for idx, pc in enumerate(parts['bodies']): # this line of code was helped by chatgpt with the colour index
    pc.set_facecolor(colors[idx]) # this line of code was helped by chatgpt with colour index
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
whiskers_min = []
whiskers_max = []

for d in data:
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

    lower, upper = adjacent_values(d, q1, q3)
    whiskers_min.append(lower)
    whiskers_max.append(upper)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)
whiskers_min = np.array(whiskers_min)
whiskers_max = np.array(whiskers_max)
# line 210 to line 224 were helped  by chatgpt, original code was only using one quartile on all data together, not separately

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['House', 'Apartment', 'Land', 'Villa']
set_axis_style(ax, labels)

# Add a legend
legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_patches, loc='upper right', title='Property Type')


# Plot 7
# Dynamically adjust y-axis limits based on data
all_data = np.concatenate(data)
ax.set_ylim(0, all_data.max() * 1.1)

plt.subplots_adjust(bottom=0.15, wspace=0.05)

price_distribution_by_property_type = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close(fig)  # Close the figure to free memory



# visualisation for bedroom number vs price
price_of_house_1 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 1)]
price_of_house_2 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 2)]
price_of_house_3 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 3)]
price_of_house_4 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 4)]
price_of_house_5 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 5)]
price_of_house_6 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 6)]
price_of_house_7 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 7)]
price_of_house_8 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 8)]
price_of_house_9 = viet_housing['price'][(viet_housing['Type'] == 'House') & (viet_housing['bedrooms'] == 9)]



data_house_1 = sorted(price_of_house_1)
data_house_2 = sorted(price_of_house_2)
data_house_3 = sorted(price_of_house_3)
data_house_4 = sorted(price_of_house_4)
data_house_5 = sorted(price_of_house_5)
data_house_6 = sorted(price_of_house_6)
data_house_7 = sorted(price_of_house_7)
data_house_8 = sorted(price_of_house_8)
data_house_9 = sorted(price_of_house_9)



data_house = [data_house_1, data_house_2, data_house_3, data_house_4, data_house_5, data_house_6, data_house_7, data_house_8, data_house_9]


fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title('Price distribution by number of bedroom for house')
ax.set_ylabel('Price')
ax.set_xlabel('Number of bedroom')


colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22']

parts = ax.violinplot(
        data_house, showmeans=False, showmedians=False,
        showextrema=False)

for idx, pc in enumerate(parts['bodies']): 
    pc.set_facecolor(colors[idx])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
whiskers_min = []
whiskers_max = []

for d in data_house:
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

    lower, upper = adjacent_values(d, q1, q3)
    whiskers_min.append(lower)
    whiskers_max.append(upper)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)
whiskers_min = np.array(whiskers_min)
whiskers_max = np.array(whiskers_max)

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
set_axis_style(ax, labels)

# Add a legend
legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_patches, loc='upper right', title='Number of bedroom')


# Plot 8
# Dynamically adjust y-axis limits based on data
all_data_house = np.concatenate(data_house)
ax.set_ylim(0, all_data_house.max() * 1.1)

plt.subplots_adjust(bottom=0.15, wspace=0.05)

price_distribution_by_number_of_bedroom_for_house = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close(fig)  # Close the figure to free memory


# visualisation of bedroom number for apartment
price_of_apartment_1 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 1)]
price_of_apartment_2 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 2)]
price_of_apartment_3 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 3)]
price_of_apartment_4 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 4)]
price_of_apartment_5 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 5)]
price_of_apartment_6 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 6)]
price_of_apartment_7 = viet_housing['price'][(viet_housing['Type'] == 'Apartment') & (viet_housing['bedrooms'] == 7)]



data_apartment_1 = sorted(price_of_apartment_1)
data_apartment_2 = sorted(price_of_apartment_2)
data_apartment_3 = sorted(price_of_apartment_3)
data_apartment_4 = sorted(price_of_apartment_4)
data_apartment_5 = sorted(price_of_apartment_5)
data_apartment_6 = sorted(price_of_apartment_6)
data_apartment_7 = sorted(price_of_apartment_7)



data_apartment = [data_apartment_1, data_apartment_2, data_apartment_3, data_apartment_4, data_apartment_5, data_apartment_6, data_apartment_7]


fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title('Price distribution by number of bedroom for apartment')
ax.set_ylabel('Price')
ax.set_xlabel('Number of bedroom')


colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2']

parts = ax.violinplot(
        data_apartment, showmeans=False, showmedians=False,
        showextrema=False)

for idx, pc in enumerate(parts['bodies']): 
    pc.set_facecolor(colors[idx])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
whiskers_min = []
whiskers_max = []

for d in data_apartment:
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

    lower, upper = adjacent_values(d, q1, q3)
    whiskers_min.append(lower)
    whiskers_max.append(upper)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)
whiskers_min = np.array(whiskers_min)
whiskers_max = np.array(whiskers_max)

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['1', '2', '3', '4', '5', '6', '7']
set_axis_style(ax, labels)

# Add a legend
legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_patches, loc='upper right', title='Number of bedroom')


# Plot 9
# Dynamically adjust y-axis limits based on data
all_data_apartment = np.concatenate(data_apartment)
ax.set_ylim(0, all_data_apartment.max() * 1.1)

plt.subplots_adjust(bottom=0.15, wspace=0.05)

price_distribution_by_number_of_bedroom_for_apartment = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close(fig)  # Close the figure to free memory


# visualisation of bedroom number for land
price_of_land_1 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 1)]
price_of_land_2 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 2)]
price_of_land_3 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 3)]
price_of_land_4 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 4)]
price_of_land_5 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 5)]
price_of_land_6 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 6)]
price_of_land_7 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 7)]
price_of_land_8 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 8)]
price_of_land_9 = viet_housing['price'][(viet_housing['Type'] == 'Land')  & (viet_housing['bedrooms'] == 9)]



data_land_1 = sorted(price_of_land_1)
data_land_2 = sorted(price_of_land_2)
data_land_3 = sorted(price_of_land_3)
data_land_4 = sorted(price_of_land_4)
data_land_5 = sorted(price_of_land_5)
data_land_6 = sorted(price_of_land_6)
data_land_7 = sorted(price_of_land_7)
data_land_8 = sorted(price_of_land_8)
data_land_9 = sorted(price_of_land_9)



data_land = [data_land_1, data_land_2, data_land_3, data_land_4, data_land_5, data_land_6, data_land_7, data_land_8, data_land_9]



fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title('Price distribution by number of bedroom for land')
ax.set_ylabel('Price')
ax.set_xlabel('Number of bedroom')


colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22']

parts = ax.violinplot(
        data_land, showmeans=False, showmedians=False,
        showextrema=False)

for idx, pc in enumerate(parts['bodies']): 
    pc.set_facecolor(colors[idx])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
whiskers_min = []
whiskers_max = []

for d in data_land:
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

    lower, upper = adjacent_values(d, q1, q3)
    whiskers_min.append(lower)
    whiskers_max.append(upper)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)
whiskers_min = np.array(whiskers_min)
whiskers_max = np.array(whiskers_max)

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
set_axis_style(ax, labels)

# Add a legend
legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_patches, loc='upper right', title='Number of bedroom')


# Plot 10
# Dynamically adjust y-axis limits based on data
all_data_land = np.concatenate(data_land)
ax.set_ylim(0, all_data_land.max() * 1.1)

plt.subplots_adjust(bottom=0.15, wspace=0.05)

price_distribution_by_number_of_bedroom_for_land = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close(fig)  # Close the figure to free memory


# visualisation of bedroom number for villa
price_of_villa_1 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 1)]
price_of_villa_2 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 2)]
price_of_villa_3 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 3)]
price_of_villa_4 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 4)]
price_of_villa_5 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 5)]
price_of_villa_6 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 6)]
price_of_villa_7 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 7)]
price_of_villa_8 = viet_housing['price'][(viet_housing['Type'] == 'Villa')  & (viet_housing['bedrooms'] == 8)]



data_villa_1 = sorted(price_of_villa_1)
data_villa_2 = sorted(price_of_villa_2)
data_villa_3 = sorted(price_of_villa_3)
data_villa_4 = sorted(price_of_villa_4)
data_villa_5 = sorted(price_of_villa_5)
data_villa_6 = sorted(price_of_villa_6)
data_villa_7 = sorted(price_of_villa_7)
data_villa_8 = sorted(price_of_villa_8)



data_villa = [data_villa_1, data_villa_2, data_villa_3, data_villa_4, data_villa_5, data_villa_6, data_villa_7, data_villa_8]

data_villa = [d for d in data_villa if len(d) > 0] # this line was helped by chatgpt for troubleshooting


fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title('Price distribution by number of bedroom for villa')
ax.set_ylabel('Price')
ax.set_xlabel('Number of bedroom')


colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F']

parts = ax.violinplot(
        data_villa, showmeans=False, showmedians=False,
        showextrema=False)

for idx, pc in enumerate(parts['bodies']): 
    pc.set_facecolor(colors[idx])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

quartile1 = []
medians = []
quartile3 = []
whiskers_min = []
whiskers_max = []

for d in data_villa:
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    quartile1.append(q1)
    medians.append(med)
    quartile3.append(q3)

    lower, upper = adjacent_values(d, q1, q3)
    whiskers_min.append(lower)
    whiskers_max.append(upper)

quartile1 = np.array(quartile1)
medians = np.array(medians)
quartile3 = np.array(quartile3)
whiskers_min = np.array(whiskers_min)
whiskers_max = np.array(whiskers_max)

inds = np.arange(1, len(medians) + 1)
ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

# Set style for the axes
labels = ['1', '2', '3', '4', '5', '6', '7', '8']
set_axis_style(ax, labels)

# Add a legend
legend_patches = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i]) for i in range(len(labels))]
ax.legend(handles=legend_patches, loc='upper right', title='Number of bedroom')


# Plot 11
# Dynamically adjust y-axis limits based on data
all_data_villa = np.concatenate(data_villa)
ax.set_ylim(0, all_data_villa.max() * 1.1)

plt.subplots_adjust(bottom=0.15, wspace=0.05)

price_distribution_by_number_of_bedroom_for_villa = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")
# plt.show()
plt.close(fig)  # Close the figure to free memory


# Plot 12
# Display the price of all the district
sns.set(style="ticks", palette="muted", color_codes=True)


plt.figure(figsize=(10, 10))
# Plot the orbital period with horizontal boxes
ax = sns.boxplot(x="price", y="district", data=viet_housing,
                 whis=np.inf, color="c")

# Add in points to show each observation
sns.stripplot(x="price", y="district", data=viet_housing,
              jitter=True, size=3, color=".3", linewidth=0)
plt.yticks(rotation=45, fontsize = 8)


# Make the quantitative axis logarithmic
sns.despine(trim=True)

price_by_location = pn.pane.Matplotlib(plt.gcf(), sizing_mode="stretch_width")

# plt.show()
plt.close()  # Close the figure to free memory


# Linking the plots

# Title with inline HTML for styling
dashboard_title = pn.pane.Markdown(
    "<h2 style='text-align: center; color: #333;'>Price Analysis Dashboard</h2>"
)

dashboard = pn.Column(
    dashboard_title, 
    pn.Row(overall_price_distribution, number_of_properties_by_type, sizing_mode="stretch_width", height=1000),  # Row 1
    pn.Row(house_price_distribution, apartment_price_distribution, sizing_mode="stretch_width", height=1000),  # Row 2
    pn.Row(land_price_distribution, villa_price_distribution, sizing_mode="stretch_width", height=1000),  # Row 3
    pn.Row(price_distribution_by_property_type, price_distribution_by_number_of_bedroom_for_house, sizing_mode="stretch_width", height=1000),  # Row 4
    pn.Row(price_distribution_by_number_of_bedroom_for_apartment, price_distribution_by_number_of_bedroom_for_land, sizing_mode="stretch_width", height=1000),  # Row 5
    pn.Row(price_distribution_by_number_of_bedroom_for_villa, price_by_location, sizing_mode="stretch_width"),  # Row 6
    css_classes=["custom-dashboard"],
)

# Arrange the plots in rows
app = pn.template.FastListTemplate(
    sidebar=[
            pn.pane.Markdown('<div style="text-align: center; font-size: 24px; font-weight: bold;">Housing Prices</div>',
    sizing_mode='stretch_width'),
            pn.pane.PNG('House.png', sizing_mode='scale_both', margin=(20, 0, 20, 0)), 
            pn.pane.Markdown("""
            <div style="text-align: justify; font-size: 16px; font-weight: bold;">
            Selling a real estate property, be it a home, apartment, or office building, isn’t as simple as having a garage sale or selling an item online. 
            Before you can even dub a real estate property as “For Sale,” it has to undergo certain processes, which is quite a lot of paperwork. 
            One of those processes is a real estate market analysis. This dashboard analyzes the trends in property prices, locations, and features, 
            which helps provide valuable insights into market dynamics.
            </div>
            """,
            sizing_mode='stretch_width'),
        ],
    main=[dashboard],
)

# Show Dashboard
app.servable()