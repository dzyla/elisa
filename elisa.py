import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.express as px
from scipy.stats import sem
import hashlib

# pip install streamlit pandas numpy matplotlib plotly scipy hashlib

# Define wells and rows, columns
ROWS = list("ABCDEFGH")
COLUMNS = list(range(1, 13))
WELL_IDS = [f"{row}{col}" for row in ROWS for col in COLUMNS]
WELL_PLATE_96 = np.reshape(WELL_IDS, (8, 12))

def initialize_session_state():
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = []
    if 'condition_grid' not in st.session_state:
        st.session_state.condition_grid = pd.DataFrame(
            WELL_PLATE_96, index=ROWS, columns=COLUMNS
        )
    if 'row_labels' not in st.session_state:
        st.session_state.row_labels = ROWS.copy()
    if 'column_labels' not in st.session_state:
        st.session_state.column_labels = COLUMNS.copy()
    if 'conditions' not in st.session_state:
        st.session_state.conditions = []
    if 'final_results' not in st.session_state:
        st.session_state.final_results = pd.DataFrame()
    if 'current_step' not in st.session_state:
        st.session_state.current_step = "Upload Data"
    if 'delimiter' not in st.session_state:
        st.session_state.delimiter = ''
    if 'plate_names' not in st.session_state:
        st.session_state.plate_names = []
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = pd.DataFrame()
    # normalized_data will hold transformed plates after normalization
    if 'normalized_data' not in st.session_state:
        st.session_state.normalized_data = []

def validate_well_identifier(well):
    if len(well) < 2 or len(well) > 3:
        return False
    row, col = well[0].upper(), well[1:]
    if row not in "ABCDEFGH":
        return False
    if not col.isdigit():
        return False
    col_num = int(col)
    if not 1 <= col_num <= 12:
        return False
    return True

def validate_plate_dataframe(df, source):
    expected_shape = (8, 12)
    if df.shape != expected_shape:
        st.warning(f"{source}: Expected {expected_shape[0]} rows and {expected_shape[1]} columns, got {df.shape}. Proceeding.")
    df.index = ROWS[:df.shape[0]]
    df.columns = COLUMNS[:df.shape[1]]
    return df

def color_viridis(val, vmin, vmax):
    if pd.isna(val):
        return ''
    norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    cmap = plt.get_cmap('coolwarm')
    rgba = cmap(norm)
    return f'background-color: {mpl.colors.rgb2hex(rgba)}'

def apply_viridis(df):
    vmin = df.min().min()
    vmax = df.max().max()
    return df.style.map(lambda x: color_viridis(x, vmin, vmax))

# Existing steps here...
def upload_data():
    st.header("1. Upload Raw ELISA Plate Data")
    upload_method = st.radio(
        "Choose data upload method:",
        ("Upload Files", "Paste Tab-Separated Values"),
        horizontal=True, index=1
    )
    
    if upload_method == "Upload Files":
        uploaded_files = st.file_uploader("Choose CSV or Excel files", accept_multiple_files=True, type=["csv", "xlsx"])
        if uploaded_files:
            for file in uploaded_files:
                try:
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file, delimiter='\t', header=None)
                    elif file.name.endswith('.xlsx'):
                        df = pd.read_excel(file, header=None)
                    df = validate_plate_dataframe(df, file.name)
                    st.session_state.raw_data.append(df)
                except Exception as e:
                    st.error(f"Error reading {file.name}: {e}")
            if st.session_state.raw_data:
                st.success(f"Uploaded {len(uploaded_files)} file(s) successfully.")
    
    elif upload_method == "Paste Tab-Separated Values":
        st.subheader("Paste Your Tab-Separated Plate Data Below")
        plate_string = st.text_area(
            "Enter tab-separated values (8 rows x 12 columns, no headers):",
            height=200,
            placeholder="Paste your data here..."
        )
        
        if st.button("Parse Pasted Data"):
            if plate_string.strip() == "":
                st.error("Please paste your tab-separated data.")
            else:
                try:
                    df = pd.read_csv(StringIO(plate_string), delimiter='\t', header=None)
                    df = validate_plate_dataframe(df, "Pasted String")
                    st.session_state.raw_data.append(df)
                    st.success("Pasted string data parsed successfully.")
                except Exception as e:
                    st.error(f"Error parsing the pasted string: {e}")
    
    # Synchronize plate_names with raw_data
    while len(st.session_state.plate_names) < len(st.session_state.raw_data):
        idx = len(st.session_state.plate_names)
        st.session_state.plate_names.append(f"Plate {idx+1}")
    while len(st.session_state.plate_names) > len(st.session_state.raw_data):
        st.session_state.plate_names.pop()

    if st.session_state.raw_data:
        st.subheader("Uploaded Plates")
        for idx, df in enumerate(st.session_state.raw_data, start=1):
            current_name = st.session_state.plate_names[idx-1]
            st.markdown(f"**{current_name}**")
            
            new_name = st.text_input(
                f"Rename {current_name}",
                value=current_name,
                key=f"plate_name_{idx}"
            )
            st.session_state.plate_names[idx-1] = new_name
            
            styled_df = apply_viridis(df).format("{:.2f}")
            st.dataframe(styled_df)
            
            if st.button(f"Remove Plate {idx}", key=f"remove_plate_{idx}"):
                st.session_state.raw_data.pop(idx-1)
                st.session_state.plate_names.pop(idx-1)
                st.success(f"Plate {idx} removed successfully.")
                st.rerun()


def color_string(val):
    """
    Assigns a unique color to each unique string by hashing it.
    If val is not a string or is NaN, returns no background.
    """
    if pd.isna(val) or not isinstance(val, str):
        return ''
    
    # Hash the string to get a reproducible numeric value
    hash_object = hashlib.sha256(val.encode('utf-8'))
    hash_int = int(hash_object.hexdigest()[:8], 16)
    # Normalize hash to [0,1]
    norm = hash_int / 0xFFFFFFFF
    
    # Use a colormap (e.g., Viridis) to map to a color
    cmap = plt.get_cmap('RdGy')
    rgba = cmap(norm)
    return f'background-color: {mpl.colors.rgb2hex(rgba)}'

def annotate_conditions():
    st.header("2. Annotate Plate Conditions")
    if not st.session_state.raw_data:
        st.warning("Please upload raw data before annotating conditions.")
        return False
    annotation_style = st.radio(
        "Choose annotation method:",
        ("Manual Entry", "Automatic Annotation"),
        index=0
    )
    if annotation_style == "Manual Entry":
        st.subheader("Define Conditions on the 96-Well Plate")
        st.markdown("Enter condition names in the wells. Leave a cell empty to treat it as a unique condition.")
        edited_grid = st.data_editor(
            st.session_state.condition_grid,
            column_config={col: st.column_config.TextColumn(
                label=f"{col}",
                help=f"Enter condition name for well {col}",
                default="",
                max_chars=50,
            ) for col in COLUMNS},
            hide_index=False,
            use_container_width=True,
            key='condition_grid_editor'
        )
        edited_grid = edited_grid.astype(str)
        edited_grid = edited_grid.replace("nan", "")
        st.session_state.condition_grid = edited_grid
        st.markdown("---")
        
    elif annotation_style == "Automatic Annotation":
        st.markdown("### Automatic Annotation Setup")

        st.write("Provide custom row and column labels. The tool will combine them to form conditions for each well.")
        st.write("For example, if Row A is 'Sample1' and Column 1 is 'Time0', and underscore is chosen as a delimiter, A1 becomes 'Sample1_Time0'.")

        # Choose a delimiter for combining row and column labels
        delimiter_options = {
            "None": '',
            "Underscore (_)": '_',
            "Dash (-)": '-',
            "Space ( )": ' '
        }
        selected_delimiter = st.selectbox(
            "Delimiter between row and column labels:",
            list(delimiter_options.keys()),
            index=0
        )
        delimiter = delimiter_options[selected_delimiter]

        st.subheader("Row Labels")
        st.write("Enter labels for each row (A-H). These replace the default row identifiers when forming conditions.")
        row_labels = []
        for i, row_id in enumerate(ROWS):
            row_label = st.text_input(f"Label for Row {row_id}:", value=f"Row{i+1}", key=f"auto_row_{i}")
            row_labels.append(row_label.strip())

        st.subheader("Column Labels")
        st.write("Enter labels for each column (1-12). These replace the default column identifiers when forming conditions.")
        col_labels = []
        for j, col_id in enumerate(COLUMNS):
            col_label = st.text_input(f"Label for Column {col_id}:", value=f"Col{j+1}", key=f"auto_col_{j}")
            col_labels.append(col_label.strip())

        # Automatically populate the condition grid
        auto_grid = pd.DataFrame(index=ROWS, columns=COLUMNS)
        for i, row_id in enumerate(ROWS):
            for j, col_id in enumerate(COLUMNS):
                r_label = row_labels[i]
                c_label = col_labels[j]
                # Form the condition name
                condition_name = f"{r_label}{delimiter}{c_label}" if delimiter else f"{r_label}{c_label}"
                auto_grid.at[row_id, col_id] = condition_name

        # Convert column names to strings to match manual annotation behavior
        auto_grid.columns = [str(c) for c in auto_grid.columns]

        st.markdown("### Preview and Edit Conditions")
        st.write("Below is the automatically generated condition grid. You can edit any cell before submitting, similar to manual annotation.")
        edited_grid = st.data_editor(
            auto_grid,
            column_config={str(col): st.column_config.TextColumn(
                label=str(col),
                help=f"Enter condition name for well {col}",
                default="",
                max_chars=50,
            ) for col in auto_grid.columns},
            hide_index=False,
            use_container_width=True,
            key='auto_condition_grid_editor'
        )

        # Ensure condition grid columns are strings
        edited_grid.columns = [str(c) for c in edited_grid.columns]

        st.session_state.condition_grid = edited_grid

    styled_preview = edited_grid.style.map(color_string)
    st.write("Color-coded view of conditions (non-editable preview):")
    st.dataframe(styled_preview)
    
    if st.button("Submit Condition Assignments"):
        condition_mapping = {}
        for row in ROWS:
            for col in [str(c) for c in COLUMNS]:
                condition = st.session_state.condition_grid.at[row, col]
                if pd.notna(condition) and condition.strip() != "":
                    condition = condition.strip().lower()
                    if condition in condition_mapping:
                        condition_mapping[condition].append(f"{row}{col}")
                    else:
                        condition_mapping[condition] = [f"{row}{col}"]
                else:
                    unique_condition = f"unique_{row}{col}"
                    condition_mapping[unique_condition] = [f"{row}{col}"]

        st.session_state.conditions = []
        for condition_name, wells in condition_mapping.items():
            st.session_state.conditions.append({
                "name": condition_name.title(),
                "wells": wells,
                "aggregation": "Average"
            })

        # Check for duplicate wells
        all_selected_wells = sum([c['wells'] for c in st.session_state.conditions], [])
        duplicates = set([w for w in all_selected_wells if all_selected_wells.count(w) > 1])
        if duplicates:
            st.error(f"Duplicate wells detected: {', '.join(duplicates)}. Assign unique wells.")
        else:
            st.success("Condition assignments submitted successfully.")
            st.session_state.current_step = "Plate Modification/Normalization"

##########################################################
# NEW SECTION: Plate Modification/Normalization
##########################################################
def plate_modification():
    st.header("3. Plate Modification/Normalization")
    if not st.session_state.raw_data:
        st.warning("Please upload raw data first.")
        return False
    if not st.session_state.conditions:
        st.warning("Please annotate conditions before normalization.")
        return False
    
    # Initialize normalized_data if empty
    if not st.session_state.normalized_data:
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]
    
    st.markdown("You can normalize each plate based on min and/or max values.")
    st.markdown("If both min and max are given, perform min-max scaling. If only min is given, subtract min. If only max is given, divide by max. If neither, no normalization.")
    
    scale_choice = st.radio("Select Scale for Min-Max:", ("0-1", "0-100%"), index=0)
    scale_factor = 1.0 if scale_choice == "0-1" else 100.0
    
    # Inputs for min value
    min_input_type = st.selectbox("Min Value Input Type:", ["None", "Numeric", "Well"])
    if min_input_type == "Numeric":
        min_val = st.number_input("Enter numeric min value:", value=0.0)
        min_well_str = None
    elif min_input_type == "Well":
        min_well_str = st.text_input("Enter min well(s), separated by commas (e.g. A1 or A1,A2):", value="A1")
        min_val = None
    else:
        min_val = None
        min_well_str = None
    
    # Inputs for max value
    max_input_type = st.selectbox("Max Value Input Type:", ["None", "Numeric", "Well"])
    if max_input_type == "Numeric":
        max_val = st.number_input("Enter numeric max value:", value=1.0)
        max_well_str = None
    elif max_input_type == "Well":
        max_well_str = st.text_input("Enter max well(s), separated by commas (e.g. A2 or G11,H11):", value="A2")
        max_val = None
    else:
        max_val = None
        max_well_str = None
    
    def get_plate_reference_value(plate, wells_str):
        """
        Given a plate (DataFrame) and a string of wells separated by commas,
        validate each well, extract values, and return the average.
        If no valid wells are found, return None.
        """
        if not wells_str:
            return None
        wells = [w.strip() for w in wells_str.split(',') if w.strip()]
        
        values = []
        for well in wells:
            if validate_well_identifier(well):
                row = well[0].upper()
                col = int(well[1:])
                try:
                    value = plate.at[row, col]
                    # Ensure it's numeric
                    value = float(value)
                    values.append(value)
                except Exception:
                    st.warning(f"Could not retrieve numeric value for well {well} on this plate.")
            else:
                st.warning(f"Invalid well identifier {well}. Skipping this well.")
        
        if values:
            return np.mean(values)
        else:
            return None

    # Button to apply normalization
    if st.button("Apply Normalization"):
        normalized_data = []
        for plate_idx, plate in enumerate(st.session_state.raw_data):
            plate_modified = plate.copy()
            
            # Determine min value for this plate
            if min_input_type == "Well":
                plate_min = get_plate_reference_value(plate_modified, min_well_str)
            else:
                plate_min = min_val
            
            # Determine max value for this plate
            if max_input_type == "Well":
                plate_max = get_plate_reference_value(plate_modified, max_well_str)
            else:
                plate_max = max_val
            
            # Apply normalization logic
            # 1) Both min and max: (plate - min) / (max - min) * scale_factor
            # 2) Only min: plate - min
            # 3) Only max: plate / max
            # 4) None: no change
            if plate_min is not None and plate_max is not None:
                denom = plate_max - plate_min
                if denom != 0:
                    plate_modified = (plate_modified - plate_min) / denom * scale_factor
                else:
                    st.warning("Max and min are the same. Cannot min-max normalize this plate.")
            elif plate_min is not None and plate_max is None:
                plate_modified = plate_modified - plate_min
            elif plate_min is None and plate_max is not None:
                if plate_max != 0:
                    plate_modified = plate_modified / plate_max
                else:
                    st.warning("Max value is zero. Cannot divide by zero for this plate.")
            
            normalized_data.append(plate_modified)
        
        st.session_state.normalized_data = normalized_data
        st.success("Normalization applied.")
    
    if st.button("Restore Raw Data"):
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]
        st.success("Raw data restored.")
    
    # Show normalized plates
    st.subheader("Current Plate Data (After Normalization)")
    for idx, df in enumerate(st.session_state.normalized_data, start=1):
        st.markdown(f"**{st.session_state.plate_names[idx-1]}**")
        styled_df = apply_viridis(df).format("{:.2f}")
        st.dataframe(styled_df)
    
    # Advance to next step if desired
    if st.button("Proceed to Process Data"):
        st.session_state.current_step = "Process Data"
        return True
    return False

def process_data():
    st.header("4. Process Data")
    
    if not st.session_state.raw_data:
        st.warning("Please upload raw data before processing.")
        return False
    if not st.session_state.conditions:
        st.warning("Please annotate conditions before processing.")
        return False
    if not st.session_state.normalized_data:
        # If no normalization done, just copy raw_data
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]
    
    if st.button("Submit and Process Data"):
        all_condition_names = [c['name'] for c in st.session_state.conditions]
        plate_means_list = []
        plate_sems_list = []
        
        for plate_idx, plate in enumerate(st.session_state.normalized_data, start=1):
            plate_means = {}
            plate_sems = {}
            for condition in st.session_state.conditions:
                wells = condition['wells']
                condition_name = condition['name']
                
                values = []
                for well in wells:
                    if not validate_well_identifier(well):
                        continue
                    row = well[0].upper()
                    col = int(well[1:])
                    try:
                        value = float(plate.at[row, col])
                        values.append(value)
                    except KeyError:
                        continue
                    except ValueError:
                        continue
                if len(values) > 1:
                    cond_mean = np.mean(values)
                    cond_sem = sem(values, nan_policy='omit')
                elif len(values) == 1:
                    cond_mean = values[0]
                    cond_sem = np.nan
                else:
                    cond_mean = np.nan
                    cond_sem = np.nan
                
                plate_means[condition_name] = cond_mean
                plate_sems[condition_name] = cond_sem
            
            plate_means_list.append(plate_means)
            plate_sems_list.append(plate_sems)
        
        per_plate_means = pd.DataFrame(plate_means_list, index=st.session_state.plate_names, columns=all_condition_names)
        per_plate_sems = pd.DataFrame(plate_sems_list, index=st.session_state.plate_names, columns=all_condition_names)
        
        st.session_state.per_plate_means = per_plate_means
        st.session_state.per_plate_sems = per_plate_sems
        
        st.subheader("Per-Plate Mean Values")
        st.dataframe(per_plate_means.style.format("{:.2f}"))
        
        st.subheader("Per-Plate SEM Values")
        st.dataframe(per_plate_sems.style.format("{:.2f}"))
        
        valid_conditions = per_plate_means.dropna(axis=1, how='all').columns
        final_results_means = per_plate_means[valid_conditions].mean().to_frame(name='Averaged/Summed Value')
        final_results_sems = per_plate_means[valid_conditions].apply(lambda x: sem(x, nan_policy='omit'))
        
        final_results = pd.concat([final_results_means, final_results_sems], axis=1)
        final_results.reset_index(inplace=True)
        final_results.columns = ['Condition', 'Averaged/Summed Value', 'Standard Error']
        
        st.session_state.final_results = final_results
        st.success("Data processing complete.")
        st.session_state.current_step = "Display Results"
        return True
    return False

def display_results():
    st.header("5. Final Results")
    
    if st.session_state.final_results.empty:
        st.warning("No results to display. Please process the data first.")
        return
    
    st.subheader("Combined Results Across Plates")
    c1, c2, c3 = st.columns(3)
    sort_data = c1.checkbox("Sort Combined Data", value=False)
    if sort_data:
        sort_column = c2.selectbox("Select column to sort combined data by:", ['Condition', 'Averaged/Summed Value', 'Standard Error'])
        st.session_state.final_results = st.session_state.final_results.sort_values(sort_column, ascending=False)
    
    styled_results = st.session_state.final_results.style.format({
        'Averaged/Summed Value': "{:.2f}",
        'Standard Error': "{:.2f}"
    })
    st.dataframe(styled_results)
    
    csv = st.session_state.final_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='elisa_results.csv',
        mime='text/csv',
    )
    
    fig_bar = px.bar(
        st.session_state.final_results,
        x='Condition',
        y='Averaged/Summed Value',
        error_y='Standard Error',
        color='Averaged/Summed Value',
        color_continuous_scale='Viridis',
        title='Combined ELISA Results (Across All Plates)',
        labels={'Averaged/Summed Value': 'Value'},
    )
    fig_bar.update_layout(xaxis_title='Condition', yaxis_title='Averaged/Summed Value', title_x=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    show_individual = st.checkbox("Show Individual Plates with Error Bars", value=False)
    if show_individual and 'per_plate_means' in st.session_state and 'per_plate_sems' in st.session_state:
        per_plate_means = st.session_state.per_plate_means
        per_plate_sems = st.session_state.per_plate_sems
        
        st.subheader("Individual Plate Results")
        
        means_long = per_plate_means.reset_index().melt(id_vars='index', var_name='Condition', value_name='Value')
        means_long = means_long.rename(columns={'index': 'Plate'})
        
        sems_long = per_plate_sems.reset_index().melt(id_vars='index', var_name='Condition', value_name='SEM')
        sems_long = sems_long.rename(columns={'index': 'Plate'})
        
        plate_data_long = pd.merge(means_long, sems_long, on=['Plate', 'Condition'], how='left')
        
        c_plt1, c_plt2, c_plt3 = st.columns(3)
        sort_per_plate = c_plt1.checkbox("Sort Individual Plate Data", value=False)
        if sort_per_plate:
            sort_column_plate = c_plt2.selectbox(
                "Select column to sort by (Per Plate):",
                ['Condition', 'Value', 'SEM']
            )
        
        for plate_name in plate_data_long['Plate'].unique():
            plate_subset = plate_data_long[plate_data_long['Plate'] == plate_name].dropna(subset=['Value'])
            if sort_per_plate:
                if sort_column_plate in ['Value', 'SEM']:
                    plate_subset = plate_subset.sort_values(sort_column_plate, ascending=False)
                else:
                    plate_subset = plate_subset.sort_values(sort_column_plate, ascending=False)
            
            fig_plate = px.bar(
                plate_subset,
                x='Condition',
                y='Value',
                error_y='SEM',
                color='Value',
                color_continuous_scale='Viridis',
                title=f'ELISA Results for {plate_name}',
                labels={'Value': 'Value'},
            )
            fig_plate.update_layout(xaxis_title='Condition', yaxis_title='Value', title_x=0.5)
            st.plotly_chart(fig_plate, use_container_width=True)

def main():
    st.set_page_config(page_title="ELISA Plate Data Annotator and Analyzer", layout="wide")
    st.title("ELISA Plate Data Annotator and Analyzer")
    
    initialize_session_state()
    
    steps = ["Upload Data", "Annotate Conditions", "Plate Modification/Normalization", "Process Data", "Display Results"]
    
    st.sidebar.title("Navigation")
    selected_step = st.sidebar.radio("Go to", options=steps, index=steps.index(st.session_state.current_step))
    
    # Enforce prerequisites for each step
    step_index = steps.index(selected_step)
    prerequisites = {
        "Annotate Conditions": len(st.session_state.raw_data) > 0,
        "Plate Modification/Normalization": len(st.session_state.conditions) > 0,
        "Process Data": len(st.session_state.conditions) > 0,  # Conditions required
        "Display Results": not st.session_state.final_results.empty
    }
    
    if selected_step == "Annotate Conditions" and not prerequisites["Annotate Conditions"]:
        st.sidebar.warning("Please complete the Upload Data step first.")
        st.stop()
    elif selected_step == "Plate Modification/Normalization" and not prerequisites["Plate Modification/Normalization"]:
        st.sidebar.warning("Please complete the Annotate Conditions step first.")
        st.stop()
    elif selected_step == "Process Data" and not prerequisites["Process Data"]:
        st.sidebar.warning("Please complete the Plate Modification/Normalization step first.")
        st.stop()
    elif selected_step == "Display Results" and not prerequisites["Display Results"]:
        st.sidebar.warning("Please complete the Process Data step first.")
        st.stop()
    
    if selected_step == "Upload Data":
        upload_data()
    elif selected_step == "Annotate Conditions":
        annotate_conditions()
    elif selected_step == "Plate Modification/Normalization":
        plate_modification()
    elif selected_step == "Process Data":
        process_data()
    elif selected_step == "Display Results":
        display_results()
    
    # Progress indicator
    progress = 0
    if len(st.session_state.raw_data) > 0:
        progress += 20
    if len(st.session_state.conditions) > 0:
        progress += 20
    if 'normalized_data' in st.session_state and st.session_state.normalized_data:
        progress += 20
    if not st.session_state.final_results.empty:
        progress += 40
    st.sidebar.progress(progress)

if __name__ == "__main__":
    main()
