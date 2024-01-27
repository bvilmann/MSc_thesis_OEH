import pandas as pd
import os

# Define the folder path
folder_path = r"C:\Users\bvilm\Dropbox\DTU\11_master thesis\data\oeh_clusters"

# Initialize an empty DataFrame to store the data
all_data = pd.DataFrame()

# Define the scaling factor
scaling_factor = 150 / 124.86

# Loop through the files in the folder
for i in range(6):
    # Construct the file path
    file_path = os.path.join(folder_path, f"cluster{i}.csv")
    
    # Read the CSV file into a DataFrame
    cluster_data = pd.read_csv(file_path, header=None, names=['Dist', 'Value'])
    
    # Convert the distance to kilometers
    cluster_data['Value'] = cluster_data['Value'] * scaling_factor
    
    # Add a new column to identify the cluster
    cluster_data['Cluster'] = f"{i+1}"
    # cluster_data['Cluster'] = '\\textbf{'+f"{i+1}" + '}'
    
    # Concatenate the data into the main DataFrame
    all_data = pd.concat([all_data, cluster_data], ignore_index=True)

# Now, all_data contains all the data from the files, with an additional column indicating the cluster


# Group by the 'Cluster' column and calculate the desired statistics
stats = all_data.groupby('Cluster')['Value'].agg(['count', 'min', 'mean', 'median', 'max'])

# Calculate overall statistics
overall_stats = pd.DataFrame({
    'count': [all_data['Value'].count()],
    'min': [all_data['Value'].min()],
    'mean': [all_data['Value'].mean()],
    'median': [all_data['Value'].median()],
    'max': [all_data['Value'].max()]
}, index=['Overall'])

# Concatenate the overall statistics to the stats DataFrame
stats = pd.concat([stats, overall_stats])

# Convert the 'count' column to integers
stats['count'] = stats['count'].astype(int)

# Round all float columns to 2 decimal places
stats.loc[:, stats.dtypes == float] = stats.loc[:, stats.dtypes == float].round(2)

# Convert the 'count' column to integers
stats['count'] = stats['count'].astype(int)
stats.loc[:, stats.dtypes == int] = stats.loc[:, stats.dtypes == int].round(0)

# Make the index bold and start with an uppercase letter
stats.index = stats.index.str.capitalize()
stats.index = '\\textbf{' + stats.index + '}'
stats.index.name = '\\textbf{Cluster}'

stats = stats.T
stats.index = stats.index.str.capitalize()
stats.index = '\\textbf{' + stats.index + '}'

# Transpose the DataFrame and convert it to a LaTeX table
latex_table = stats.to_latex(column_format='lrrrrrrr', position='H', escape=False)

# Add a caption and label to the LaTeX table
caption = "\\caption[Mapped OWF distances from different OEH clusters]{Mapped OWF distances from different OEH clusters based on \\cite{mart2022a}.}"
label = "\\label{tab:oeh_owf_distances}"
latex_table = latex_table.replace("\\begin{table}[H]", "\\begin{table}[H]"+f"\n  {caption}\n  {label}")

print(latex_table)

# Write the LaTeX table to a .tex file
# with open(r"C:\Users\bvilm\Dropbox\Apps\Overleaf\Thesis - Stability Analysis of MMC-HVDC Connections with Parallel Grid-Forming Mode in Offshore Energy Hubs\sources\00_appendices\tab_00_C_distances.tex", "w") as file:
#     file.write(latex_table)
#%% EXTRAPOLATION
def extrapolate(x,x1,x2,y1,y2):
    y = y1 + (x-x1)/(x2-x1)*(y2-y1)
    print(y)
    return 
    

extrapolate(525,200,400,3000,3500)



