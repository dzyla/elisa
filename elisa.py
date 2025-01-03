import streamlit as st  
import pandas as pd  
import numpy as np  
from io import StringIO  
import matplotlib.pyplot as plt  
import matplotlib as mpl  
import plotly.express as px  
from scipy.stats import sem  
import hashlib  
import plotly.graph_objects as go
from datetime import datetime

# Define wells and rows, columns  
ROWS = list("ABCDEFGH")  
COLUMNS = list(range(1, 13))  
WELL_IDS = [f"{row}{col}" for row in ROWS for col in COLUMNS]  
WELL_PLATE_96 = np.reshape(WELL_IDS, (8, 12))  

def initialize_session_state():  
    if 'raw_data' not in st.session_state:  
        st.session_state.raw_data = []  
    if 'condition_grids' not in st.session_state:  
        st.session_state.condition_grids = []  
    if 'row_labels' not in st.session_state:  
        st.session_state.row_labels = ROWS.copy()  
    if 'column_labels' not in st.session_state:  
        st.session_state.column_labels = COLUMNS.copy()  
    if 'conditions_list' not in st.session_state:  
        st.session_state.conditions_list = []  
    if 'final_results' not in st.session_state:  
        st.session_state.final_results = pd.DataFrame()  
    if 'current_step' not in st.session_state:  
        st.session_state.current_step = "Upload Data"  
    if 'delimiter' not in st.session_state:  
        st.session_state.delimiter = ''  
    if 'plate_names' not in st.session_state:  
        st.session_state.plate_names = []  
    if 'normalized_data' not in st.session_state:  
        st.session_state.normalized_data = []  
    if 'per_plate_means' not in st.session_state:  
        st.session_state.per_plate_means = []  
    if 'per_plate_sems' not in st.session_state:  
        st.session_state.per_plate_sems = []  
    if 'plate_annotations' not in st.session_state:  
        st.session_state.plate_annotations = {}  
    if 'annotation_methods' not in st.session_state:  
        st.session_state.annotation_methods = {}  
    if 'temp_condition_grids' not in st.session_state:  
        st.session_state.temp_condition_grids = {}  
    if 'experiment_name' not in st.session_state:
        st.session_state.experiment_name = ""

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

    try:  
        df = df.astype(float)  
    except Exception as e:  
        st.warning(f"Could not convert data to numeric: {e}. Make sure all values are numbers.")  
        st.stop()  
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
    return df.style.map(lambda val: color_viridis(val, vmin, vmax))  

def color_string(val):  
    if pd.isna(val) or not isinstance(val, str):  
        return ''  
    hash_object = hashlib.sha256(val.encode('utf-8'))  
    hash_int = int(hash_object.hexdigest()[:8], 16)  
    norm = hash_int / 0xFFFFFFFF  
    cmap = plt.get_cmap('RdGy')  
    rgba = cmap(norm)  
    return f'background-color: {mpl.colors.rgb2hex(rgba)}'  

def upload_data():  
    st.header("1. Upload Raw ELISA Plate Data")  

    # Add experiment name input at the top
    st.session_state.experiment_name = st.text_input(
        "Experiment Name:",
        value=st.session_state.experiment_name,
        help="This name will be used in downloaded files"
    )

    upload_method = "Paste Tab-Separated Values"  

    if upload_method == "Paste Tab-Separated Values":  
        num_plates = st.number_input("Number of Plates to Upload:", min_value=1, value=1, step=1)  
        plate_strings = []  

        # Initialize or resize condition_grids if needed  
        while len(st.session_state.condition_grids) < num_plates:  
            st.session_state.condition_grids.append(  
                pd.DataFrame(WELL_PLATE_96, index=ROWS, columns=COLUMNS)  
            )  

        for plate_idx in range(num_plates):  
            st.subheader(f"Plate {plate_idx + 1}")  
            plate_string = st.text_area(  
                f"Enter tab-separated values for Plate {plate_idx + 1} (8 rows x 12 columns, no headers):",  
                height=100,  
                placeholder="Paste your data here...",  
                key=f"plate_string_{plate_idx}"  
            )  
            plate_strings.append(plate_string)  

        if st.button("Parse Pasted Data"):  
            st.session_state.raw_data = []  # Clear existing data  
            for idx, plate_string in enumerate(plate_strings):  
                if plate_string.strip() == "":  
                    st.error(f"Please paste your tab-separated data for Plate {idx + 1}.")  
                else:  
                    try:  
                        df = pd.read_csv(StringIO(plate_string), delimiter='\t', header=None)  
                        df = validate_plate_dataframe(df, f"Pasted String for Plate {idx + 1}")  
                        st.session_state.raw_data.append(df)  
                    except Exception as e:  
                        st.error(f"Error parsing the pasted string for Plate {idx + 1}: {e}")  

            if st.session_state.raw_data:  
                st.success("Pasted string data parsed successfully.")  

        # Synchronize plate_names with raw_data  
        while len(st.session_state.plate_names) < len(st.session_state.raw_data):  
            idx = len(st.session_state.plate_names)  
            st.session_state.plate_names.append(f"Plate {idx + 1}")  
        while len(st.session_state.plate_names) > len(st.session_state.raw_data):  
            st.session_state.plate_names.pop()  

        if st.session_state.raw_data:  
            st.subheader("Uploaded Plates")  
            for idx, df in enumerate(st.session_state.raw_data):  
                current_name = st.session_state.plate_names[idx]  
                st.markdown(f"**{current_name}**")  

                new_name = st.text_input(  
                    "Rename Plate",  
                    value=current_name,  
                    key=f"plate_name_{idx}"  
                )  
                st.session_state.plate_names[idx] = new_name  

                styled_df = apply_viridis(df).format("{:.2f}")  
                st.dataframe(styled_df)  

                if st.button(f"Remove Plate {idx + 1}", key=f"remove_plate_{idx}"):  
                    st.session_state.raw_data.pop(idx)  
                    st.session_state.plate_names.pop(idx)  
                    if idx < len(st.session_state.condition_grids):  
                        st.session_state.condition_grids.pop(idx)  
                    # Clear related annotations  
                    plate_key = f"plate_{idx}"  
                    if plate_key in st.session_state.plate_annotations:  
                        del st.session_state.plate_annotations[plate_key]  
                    if plate_key in st.session_state.annotation_methods:  
                        del st.session_state.annotation_methods[plate_key]  
                    st.success(f"Plate {idx + 1} removed successfully.")  
                    st.rerun()  

def get_stored_annotations(plate_idx):  
    """Helper function to retrieve stored annotations for a plate"""  
    plate_key = f"plate_{plate_idx}"  
    return st.session_state.plate_annotations.get(plate_key, {})  

def get_condition_grid(plate_idx):  
    """Helper function to get or create a condition grid for a plate"""  
    if plate_idx < len(st.session_state.condition_grids):  
        return st.session_state.condition_grids[plate_idx]  
    else:  
        new_grid = pd.DataFrame(WELL_PLATE_96, index=ROWS, columns=COLUMNS)  
        st.session_state.condition_grids.append(new_grid)  
        return new_grid  

def annotate_conditions():  
    st.header("2. Annotate Plate Conditions")  
    if not st.session_state.raw_data:  
        st.warning("Please upload raw data before annotating conditions.")  
        return False  

    st.write("**Please annotate each plate individually.**")  

    # Initialize condition grids if needed  
    while len(st.session_state.condition_grids) < len(st.session_state.raw_data):  
        st.session_state.condition_grids.append(  
            pd.DataFrame(WELL_PLATE_96, index=ROWS, columns=COLUMNS)  
        )  

    plates_with_errors = []
    all_plates_annotated = True

    # Create tabs for each plate
    tabs = st.tabs([f"Plate {i+1}: {name}" for i, name in enumerate(st.session_state.plate_names)])

    for idx, (tab, plate_name) in enumerate(zip(tabs, st.session_state.plate_names)):
        with tab:
            plate_key = f"plate_{idx}"

            # Get or set annotation method from session state  
            if plate_key not in st.session_state.annotation_methods:  
                st.session_state.annotation_methods[plate_key] = "Manual Entry"  

            annotation_style = st.radio(
                f"Choose annotation method for {plate_name}:",
                ("Manual Entry", "Automatic Annotation"),
                index=0 if st.session_state.annotation_methods[plate_key] == "Manual Entry" else 1,
                key=f"annotation_style_{idx}"
            )
            st.session_state.annotation_methods[plate_key] = annotation_style

            # Get existing condition grid or create new one
            condition_grid = get_condition_grid(idx)

            if annotation_style == "Manual Entry":
                st.markdown("Enter condition names in the wells. Leave a cell empty to treat it as a unique condition.")  

                # Get existing grid from temp storage or use current grid  
                temp_key = f"temp_grid_{idx}"  
                if temp_key in st.session_state.temp_condition_grids:  
                    condition_grid = st.session_state.temp_condition_grids[temp_key]  

                edited_grid = st.data_editor(
                    condition_grid,
                    column_config={str(col): st.column_config.TextColumn(
                        label=str(col),
                        help=f"Enter condition name for well {col}",
                        default="",
                        max_chars=50,
                    ) for col in COLUMNS},
                    hide_index=False,
                    use_container_width=True,
                    key=f'condition_grid_editor_{idx}'
                )

                # Store in temporary storage  
                st.session_state.temp_condition_grids[temp_key] = edited_grid  
                condition_grid = edited_grid  

            elif annotation_style == "Automatic Annotation":  
                st.markdown("### Automatic Annotation Setup")  

                delimiter_options = {  
                    "None": '',  
                    "Underscore (_)": '_',  
                    "Dash (-)": '-',  
                    "Space ( )": ' '  
                }  
                selected_delimiter = st.selectbox(  
                    "Delimiter between row and column labels:",  
                    list(delimiter_options.keys()),  
                    index=0,  
                    key=f"delimiter_{idx}"  
                )  
                delimiter = delimiter_options[selected_delimiter]  

                st.subheader("Row Labels")  
                row_labels = []  
                for i, row_id in enumerate(ROWS):  
                    saved_label = st.session_state.get(f"row_label_{idx}_{i}", row_id)  
                    row_label = st.text_input(  
                        f"Label for Row {row_id}:",
                        value=saved_label,  
                        key=f"auto_row_{idx}_{i}"  
                    )  
                    st.session_state[f"row_label_{idx}_{i}"] = row_label  
                    row_labels.append(row_label.strip())  

                st.subheader("Column Labels")  
                col_labels = []  
                for j, col_id in enumerate(COLUMNS):  
                    saved_label = st.session_state.get(f"col_label_{idx}_{j}", str(col_id))  
                    col_label = st.text_input(  
                        f"Label for Column {col_id}:",
                        value=saved_label,  
                        key=f"auto_col_{idx}_{j}"  
                    )  
                    st.session_state[f"col_label_{idx}_{j}"] = col_label  
                    col_labels.append(col_label.strip())  

                # Automatically populate the condition grid  
                auto_grid = pd.DataFrame(index=ROWS, columns=COLUMNS)  
                for i, row_id in enumerate(ROWS):  
                    for j, col_id in enumerate(COLUMNS):  
                        r_label = row_labels[i]  
                        c_label = col_labels[j]  
                        condition_name = f"{r_label}{delimiter}{c_label}" if delimiter else f"{r_label}{c_label}"  
                        auto_grid.at[row_id, col_id] = condition_name  

                # Convert column names to strings  
                auto_grid.columns = [str(c) for c in auto_grid.columns]  

                st.markdown("### Preview and Edit Conditions")  
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
                    key=f'auto_condition_grid_editor_{idx}'
                )
                condition_grid = edited_grid  

            # Show preview of conditions  
            styled_preview = condition_grid.style.map(color_string)     
            st.write("Color-coded view of conditions (non-editable preview):")  
            st.dataframe(styled_preview)  

            # Validate plate annotations  
            empty_wells = condition_grid.isna().sum().sum()
            if empty_wells > 0:
                st.warning(f"This plate has {empty_wells} unannotated wells.")
                all_plates_annotated = False
                plates_with_errors.append(plate_name)

            # Store the current grid  
            st.session_state.condition_grids[idx] = condition_grid  

    # Single confirmation button outside the tabs
    col1, col2 = st.columns([2, 1])
    with col1:
        if not all_plates_annotated:
            st.warning(f"Please complete annotations for: {', '.join(plates_with_errors)}")

    with col2:
        if st.button("Confirm All Plate Annotations and Continue", 
                    disabled=not all_plates_annotated,
                    type="primary"):
            # Process all plates
            for idx, plate_name in enumerate(st.session_state.plate_names):
                condition_grid = st.session_state.condition_grids[idx]
                plate_key = f"plate_{idx}"

                # Create condition mapping
                condition_mapping = {}
                for row in ROWS:
                    for col in [str(c) for c in COLUMNS]:
                        condition = condition_grid.at[row, col]
                        if pd.notna(condition) and condition.strip() != "":
                            condition = condition.strip()
                            if condition in condition_mapping:
                                condition_mapping[condition].append(f"{row}{col}")
                            else:
                                condition_mapping[condition] = [f"{row}{col}"]
                        else:
                            unique_condition = f"unique_{row}{col}"
                            condition_mapping[unique_condition] = [f"{row}{col}"]

                # Store in session state
                st.session_state.plate_annotations[plate_key] = condition_mapping
                st.session_state.condition_grids[idx] = condition_grid

                conditions = []
                for condition_name, wells in condition_mapping.items():
                    conditions.append({
                        "name": condition_name,
                        "wells": wells,
                        "aggregation": "Average"
                    })

                if idx < len(st.session_state.conditions_list):
                    st.session_state.conditions_list[idx] = conditions
                else:
                    st.session_state.conditions_list.append(conditions)

            st.success("All plate annotations confirmed! Proceeding to next step...")
            st.session_state.current_step = "Plate Modification/Normalization"
            st.rerun()

def get_plate_reference_value(plate, well_str):
    """
    Calculate reference value from specified wells in a plate.

    Args:
        plate (pd.DataFrame): Plate data
        well_str (str): Comma-separated list of wells (e.g., "A1,A2,A3")

    Returns:
        float: Mean value of specified wells
    """
    if not well_str:
        return None

    wells = [well.strip().upper() for well in well_str.split(',')]
    values = []

    for well in wells:
        try:
            # Extract row and column from well
            row = well[0]
            col = int(well[1:])

            # Get value from plate
            value = plate.loc[row, col]
            values.append(value)
        except (KeyError, ValueError, IndexError):
            st.warning(f"Well {well} not found in plate. Skipping.")
            continue

    if not values:
        st.error("No valid wells found.")
        return None

    return np.mean(values)

def plate_modification():  
    st.header("3. Plate Modification/Normalization")  
    if not st.session_state.raw_data:  
        st.warning("Please upload raw data first.")  
        return False  
    if not st.session_state.conditions_list:  
        st.warning("Please annotate conditions before normalization.")  
        return False  

    # Initialize normalized_data if empty  
    if not st.session_state.normalized_data:  
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]  

    st.markdown("You can normalize each plate based on min and/or max values.")  

    scale_choice = st.radio("Select Scale for Min-Max:", ("0-1", "0-100%"), index=0)  
    scale_factor = 1.0 if scale_choice == "0-1" else 100.0  

    # Add automatic min-max normalization option  
    auto_normalize = st.checkbox("Use Automatic Min-Max Normalization", value=False)  

    if auto_normalize:  
        st.info("Automatic normalization will use the minimum and maximum values from each plate.")

    for idx, plate_name in enumerate(st.session_state.plate_names):  
        st.subheader(f"Normalization for {plate_name}")  

        # Store normalization settings in session state  
        norm_key = f"norm_settings_{idx}"  
        if norm_key not in st.session_state:  
            st.session_state[norm_key] = {  
                'min_type': 'None',  
                'min_val': 0.0,  
                'min_well': 'A1',  
                'max_type': 'None',  
                'max_val': 1.0,  
                'max_well': 'A2'  
            }  

        # Inputs for min value  
        min_input_type = st.selectbox(  
            f"Min Value Input Type for {plate_name}:",  
            ["None", "Numeric", "Well"],  
            key=f"min_input_type_{idx}",  
            index=["None", "Numeric", "Well"].index(st.session_state[norm_key]['min_type'])  
        )  
        st.session_state[norm_key]['min_type'] = min_input_type  

        if min_input_type == "Numeric":  
            min_val = st.number_input(  
                f"Enter numeric min value for {plate_name}:",  
                value=st.session_state[norm_key]['min_val'],  
                key=f"min_val_{idx}"  
            )  
            st.session_state[norm_key]['min_val'] = min_val  
            min_well_str = None  
        elif min_input_type == "Well":  
            min_well_str = st.text_input(  
                f"Enter min well(s), separated by commas for {plate_name}:",  
                value=st.session_state[norm_key]['min_well'],  
                key=f"min_well_str_{idx}"  
            )  
            st.session_state[norm_key]['min_well'] = min_well_str  
            min_val = None  
        else:  
            min_val = None  
            min_well_str = None  

        # Inputs for max value  
        max_input_type = st.selectbox(  
            f"Max Value Input Type for {plate_name}:",  
            ["None", "Numeric", "Well"],  
            key=f"max_input_type_{idx}",  
            index=["None", "Numeric", "Well"].index(st.session_state[norm_key]['max_type'])  
        )  
        st.session_state[norm_key]['max_type'] = max_input_type  

        if max_input_type == "Numeric":  
            max_val = st.number_input(  
                f"Enter numeric max value for {plate_name}:",  
                value=st.session_state[norm_key]['max_val'],  
                key=f"max_val_{idx}"  
            )  
            st.session_state[norm_key]['max_val'] = max_val  
            max_well_str = None  
        elif max_input_type == "Well":  
            max_well_str = st.text_input(  
                f"Enter max well(s), separated by commas for {plate_name}:",  
                value=st.session_state[norm_key]['max_well'],  
                key=f"max_well_str_{idx}"  
            )  
            st.session_state[norm_key]['max_well'] = max_well_str  
            max_val = None  
        else:  
            max_val = None  
            max_well_str = None  

        if st.button(f"Apply Normalization for {plate_name}", key=f"apply_norm_{idx}"):  
            plate = st.session_state.normalized_data[idx]  
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
            if auto_normalize:  
                # Calculate plate-wide min and max  
                plate_min_auto = plate_modified.values.min()  
                plate_max_auto = plate_modified.values.max()  

                # Apply min-max normalization  
                if plate_max_auto != plate_min_auto:  
                    plate_modified = (plate_modified - plate_min_auto) / (plate_max_auto - plate_min_auto) * scale_factor  
                    st.success(f"Automatic min-max normalization applied to {plate_name}.")  
                    st.info(f"Plate min: {plate_min_auto:.2f}, Plate max: {plate_max_auto:.2f}")  
                else:  
                    st.warning(f"All values in {plate_name} are identical ({plate_min_auto:.2f}). Cannot normalize.")  
            else:  
                # Manual normalization logic  
                if plate_min is not None and plate_max is not None:  
                    denom = plate_max - plate_min  
                    if denom != 0:  
                        plate_modified = (plate_modified - plate_min) / denom * scale_factor  
                    else:  
                        st.warning(f"Max and min are the same for {plate_name}. Cannot min-max normalize this plate.")  
                elif plate_min is not None and plate_max is None:  
                    plate_modified = plate_modified - plate_min  
                elif plate_min is None and plate_max is not None:  
                    if plate_max != 0:  
                        plate_modified = plate_modified / plate_max  
                    else:  
                        st.warning(f"Max value is zero for {plate_name}. Cannot divide by zero for this plate.")  

            st.session_state.normalized_data[idx] = plate_modified  
            st.success(f"Normalization applied to {plate_name}.")  

        if st.button(f"Restore Raw Data for {plate_name}", key=f"restore_raw_{idx}"):  
            st.session_state.normalized_data[idx] = st.session_state.raw_data[idx].copy()  
            st.success(f"Raw data restored for {plate_name}.")  

        # Show normalized plate  
        st.markdown(f"**Current Data for {plate_name} (After Normalization)**")  
        styled_df = apply_viridis(st.session_state.normalized_data[idx]).format("{:.2f}")  
        st.dataframe(styled_df)  

    if st.button("Proceed to Process Data"):  
        st.session_state.current_step = "Process Data"  
        return True  
    return False  

def process_data():  
    st.header("4. Process Data")  

    if not st.session_state.raw_data:  
        st.warning("Please upload raw data before processing.")  
        return False  
    if not st.session_state.conditions_list:  
        st.warning("Please annotate conditions before processing.")  
        return False  
    if not st.session_state.normalized_data:  
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]  

    if st.button("Submit and Process Data"):  
        all_plate_means = []  
        all_plate_sems = []  
        all_individual_values = []  # New list to store individual values
        all_conditions = set()  

        for plate_idx, plate in enumerate(st.session_state.normalized_data):  
            plate_name = st.session_state.plate_names[plate_idx]  
            conditions = st.session_state.conditions_list[plate_idx]  

            plate_means = {}  
            plate_sems = {}  

            for condition in conditions:  
                wells = condition['wells']  
                condition_name = condition['name']  
                all_conditions.add(condition_name)  

                values = []  
                for well in wells:  
                    if not validate_well_identifier(well):  
                        continue  
                    row = well[0].upper()  
                    col = int(well[1:])  
                    try:  
                        value = float(plate.at[row, col])  
                        values.append(value)  
                        # Store individual values
                        all_individual_values.append({
                            'Condition': condition_name,
                            'Plate': plate_name,
                            'Value': value,
                            'Well': well
                        })
                    except (KeyError, ValueError):  
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

            all_plate_means.append(pd.Series(plate_means, name=plate_name))  
            all_plate_sems.append(pd.Series(plate_sems, name=plate_name))  

        # Create DataFrames from all plates  
        print(all_plate_means)
        per_plate_means = pd.DataFrame(all_plate_means)  
        per_plate_sems = pd.DataFrame(all_plate_sems)  

        # Create DataFrame with individual values  
        individual_values_df = pd.DataFrame(all_individual_values)  

        st.session_state.per_plate_means = per_plate_means  
        st.session_state.per_plate_sems = per_plate_sems  
        st.session_state.individual_values = individual_values_df  # Store individual values  

        # Create final results
        final_results_means = per_plate_means.mean().to_frame(name='Mean')  
        final_results_sems = per_plate_means.apply(lambda x: sem(x, nan_policy='omit')).to_frame(name='SEM')  
        print(final_results_sems)
        
        final_results = pd.concat([final_results_means, final_results_sems], axis=1)  
        final_results.reset_index(inplace=True)  
        final_results.columns = ['Condition', 'Mean', 'SEM']  

        st.session_state.final_results = final_results  

        # Display mean and SEM tables  
        st.subheader("Per-Plate Mean Values")  
        st.dataframe(per_plate_means.style.format("{:.2f}"))  

        st.subheader("Per-Plate SEM Values")  
        st.dataframe(per_plate_sems.style.format("{:.2f}"))  

        # Display individual values table  
        st.subheader("Individual Values")  
        st.dataframe(individual_values_df.style.format({'Value': '{:.2f}'}))  

        # Generate filename with experiment name, date, and plate names  
        experiment_name = st.session_state.experiment_name.strip().replace(' ', '_') if st.session_state.experiment_name.strip() else "ELISA"
        date_str = datetime.now().strftime('%Y%m%d')  
        plate_names_str = '_'.join([name.replace(' ', '_') for name in st.session_state.plate_names])  
        filename = f'{experiment_name}_results_{date_str}_{plate_names_str}.csv'  

        # Create combined downloadable format  
        download_df = pd.DataFrame({  
            'Condition': individual_values_df['Condition'],
            'Plate': individual_values_df['Plate'], 
            'Value': individual_values_df['Value'],  
            'Error': individual_values_df['Value'].groupby(individual_values_df['Condition']).transform(lambda x: sem(x, nan_policy='omit'))
        })  

        # Add download button  
        csv = download_df.to_csv(index=False).encode('utf-8')  
        st.download_button(  
            label="Download Combined Results (Condition, Plate, Value, Error)",  
            data=csv,  
            file_name=filename,  
            mime='text/csv',  
        )  

        st.success("Data processing complete.")  
        st.session_state.current_step = "Display Results"  
        return True  
    return False  

def process_data():
    st.header("4. Process Data")

    if not st.session_state.raw_data:
        st.warning("Please upload raw data before processing.")
        return False
    if not st.session_state.conditions_list:
        st.warning("Please annotate conditions before processing.")
        return False
    if not st.session_state.normalized_data:
        st.session_state.normalized_data = [df.copy() for df in st.session_state.raw_data]

    if st.button("Submit and Process Data"):
        all_plate_means = []
        all_plate_sems = []
        all_individual_values = []
        all_conditions = set()

        # First pass: collect all individual values
        for plate_idx, plate in enumerate(st.session_state.normalized_data):
            plate_name = st.session_state.plate_names[plate_idx]
            conditions = st.session_state.conditions_list[plate_idx]

            plate_means = {}
            plate_sems = {}

            for condition in conditions:
                wells = condition['wells']
                condition_name = condition['name']
                all_conditions.add(condition_name)

                values = []
                for well in wells:
                    if not validate_well_identifier(well):
                        continue
                    row = well[0].upper()
                    col = int(well[1:])
                    try:
                        value = float(plate.at[row, col])
                        values.append(value)
                        all_individual_values.append({
                            'Condition': condition_name,
                            'Plate': plate_name,
                            'Value': value,
                            'Well': well
                        })
                    except (KeyError, ValueError):
                        continue

                # Calculate per-plate statistics
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

            all_plate_means.append(pd.Series(plate_means, name=plate_name))
            all_plate_sems.append(pd.Series(plate_sems, name=plate_name))

        # Create DataFrames
        per_plate_means = pd.DataFrame(all_plate_means)
        per_plate_sems = pd.DataFrame(all_plate_sems)
        individual_values_df = pd.DataFrame(all_individual_values)

        # Calculate overall statistics from all individual values
        overall_stats = (individual_values_df
                        .groupby('Condition')
                        .agg({
                            'Value': ['mean', lambda x: sem(x, nan_policy='omit')]
                        })
                        .reset_index())
        overall_stats.columns = ['Condition', 'Mean', 'SEM']

        st.session_state.per_plate_means = per_plate_means
        st.session_state.per_plate_sems = per_plate_sems
        st.session_state.individual_values = individual_values_df
        st.session_state.final_results = overall_stats

        # Display results
        st.subheader("Overall Results (across all values)")
        st.dataframe(overall_stats.style.format({'Mean': '{:.2f}', 'SEM': '{:.2f}'}))

        st.subheader("Per-Plate Mean Values")
        st.dataframe(per_plate_means.style.format("{:.2f}"))

        st.subheader("Per-Plate SEM Values")
        st.dataframe(per_plate_sems.style.format("{:.2f}"))

        st.subheader("Individual Values")
        st.dataframe(individual_values_df.style.format({'Value': '{:.2f}'}))

        # Generate filename
        experiment_name = st.session_state.experiment_name.strip().replace(' ', '_') if st.session_state.experiment_name.strip() else "ELISA"
        date_str = datetime.now().strftime('%Y%m%d')
        plate_names_str = '_'.join([name.replace(' ', '_') for name in st.session_state.plate_names])
        filename = f'{experiment_name}_results_{date_str}_{plate_names_str}.csv'

        # Create combined downloadable format
        download_df = pd.DataFrame({
            'Condition': individual_values_df['Condition'],
            'Plate': individual_values_df['Plate'],
            'Value': individual_values_df['Value'],
            'Well': individual_values_df['Well'],
            'Overall_Mean': individual_values_df['Condition'].map(overall_stats.set_index('Condition')['Mean']),
            'Overall_SEM': individual_values_df['Condition'].map(overall_stats.set_index('Condition')['SEM'])
        })

        # Add download button
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Combined Results",
            data=csv,
            file_name=filename,
            mime='text/csv',
        )

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
        sort_column = c2.selectbox(  
            "Select column to sort combined data by:",  
            ['Condition', 'Mean', 'SEM']  
        )  
        st.session_state.final_results = st.session_state.final_results.sort_values(  
            sort_column,  
            ascending=False  
        )  
        
    # Display final results  
    styled_results = st.session_state.final_results.style.format({  
        'Mean': "{:.2f}",  
        'SEM': "{:.2f}"  
    })  
    st.dataframe(styled_results)  

    # Display individual values if available  
    if 'individual_values' in st.session_state and not st.session_state.individual_values.empty:
        st.subheader("Individual Values")  
        individual_df = st.session_state.individual_values  
        st.dataframe(individual_df.style.format({  
            'Value': "{:.2f}"  
        }))  

        # Generate filename with experiment name, date, and plate names  
        experiment_name = st.session_state.experiment_name.strip().replace(' ', '_') if st.session_state.experiment_name.strip() else "ELISA"
        date_str = datetime.now().strftime('%Y%m%d')  
        plate_names_str = '_'.join([name.replace(' ', '_') for name in st.session_state.plate_names])  
        filename = f'{experiment_name}_individual_values_{date_str}_{plate_names_str}.csv'  

        # Download buttons for both formats  
        # 1. Mean and SEM format  
        means_sems_df = st.session_state.final_results  

        csv_means_sems = means_sems_df.to_csv(index=False).encode('utf-8')  
        st.download_button(
            label="Download Mean and SEM Results",
            data=csv_means_sems,
            file_name=f'{experiment_name}_means_sems_{date_str}_{plate_names_str}.csv',
            mime='text/csv',
        )

        # 2. Individual values format  
        csv_individual = individual_df.to_csv(index=False).encode('utf-8')  
        st.download_button(  
            label="Download Individual Values",  
            data=csv_individual,  
            file_name=filename,  
            mime='text/csv',  
        )  

    # Plot combined results  
    fig_bar = px.bar(  
        st.session_state.final_results,  
        x='Condition',  
        y='Mean',  
        error_y='SEM',  
        color='Mean',  
        color_continuous_scale='Viridis',  
        title='Combined ELISA Results (Across All Plates)',  
        labels={'Mean': 'Value'},  
    )  
    fig_bar.update_layout(xaxis_title='Condition', yaxis_title='Value', title_x=0.5)  
    st.plotly_chart(fig_bar, use_container_width=True)  

    # Individual plate results  
    show_individual = st.checkbox("Show Individual Plates with Error Bars", value=False)  
    if show_individual and 'individual_values' in st.session_state and not st.session_state.individual_values.empty:  
        st.subheader("Individual Plate Results")  

        for plate_name in st.session_state.individual_values['Plate'].unique():  
            plate_subset = st.session_state.individual_values[  
                st.session_state.individual_values['Plate'] == plate_name  
            ].groupby('Condition').agg({
                'Value': ['mean', 'sem']
            }).reset_index()  

            plate_subset.columns = ['Condition', 'Mean', 'SEM']  

            fig_plate = px.bar(  
                plate_subset,  
                x='Condition',  
                y='Mean',  
                error_y='SEM',  
                color='Mean',  
                color_continuous_scale='Viridis',  
                title=f'ELISA Results for {plate_name}',  
                labels={'Mean': 'Value'},  
            )  
            fig_plate.update_layout(  
                xaxis_title='Condition',  
                yaxis_title='Value',  
                title_x=0.5  
            )  
            st.plotly_chart(fig_plate, use_container_width=True)  

    # Add checkbox for combined bar and scatter plot  
    show_combined = st.checkbox("Show Combined Bar and Individual Points Plot", value=False)  
    if show_combined and 'individual_values' in st.session_state and not st.session_state.individual_values.empty:  
        st.subheader("Combined Bar and Individual Points Plot")  

        # Get individual points  
        individual_df = st.session_state.individual_values  
        # Calculate summary statistics  
        summary_stats = individual_df.groupby('Condition').agg({
            'Value': ['mean', 'sem']
        }).reset_index()  
        
        summary_stats.columns = ['Condition', 'Mean', 'SEM']  

        # Create single figure  
        fig = go.Figure()  

        # Add bar plot first (it will be in the background)  
        fig.add_trace(go.Bar(  
            x=summary_stats['Condition'],  
            y=summary_stats['Mean'],  
            error_y=dict(  
                type='data',  
                array=summary_stats['SEM'],  
                visible=True,  
                thickness=1.5,  
                width=4  
            ),  
            name='Mean ± SEM',  
            marker_color='rgba(158,202,225,0.6)',  
            marker_line_color='rgb(8,48,107)',  
            marker_line_width=1.5,  
            width=0.6  # Make bars narrower  
        ))  

        # Add scatter points for each condition  
        for condition in individual_df['Condition'].unique():
            condition_data = individual_df[individual_df['Condition'] == condition]  

            # Calculate positions for swarm-like distribution  
            values = condition_data['Value'].values  
            n_points = len(values)  

            # Calculate jitter based on number of points
            jitter = np.zeros(shape=n_points) #np.random.uniform(-0.01, 0.1, size=n_points)

            fig.add_trace(go.Scatter(  
                x=[condition] * n_points + jitter.tolist(),  
                y=values.tolist() + values.tolist(),  
                mode='markers',  
                name=condition,  
                marker=dict(  
                    size=8,  
                    opacity=0.7,  
                    color='grey'  
                ),  
                hovertemplate=(
                    "Condition: %{text}<br>" +
                    "Value: %{y:.2f}<br>" +
                    "Plate: %{customdata}<extra></extra>"
                ),  
                text=[condition] * n_points,  
                customdata=condition_data['Plate'],  
                showlegend=False  
            ))  

        # Update layout  
        fig.update_layout(  
            title='Combined Bar Plot with Individual Points',  
            xaxis_title='Condition',  
            yaxis_title='Value',  
            title_x=0.5,  
            showlegend=True,  
            plot_bgcolor='white',  
            xaxis=dict(  
                showgrid=False,  
                showline=True,  
                linecolor='black',  
                tickfont=dict(size=12),  
                ticks='outside',  
                type='category'  # This ensures proper categorical axis  
            ),  
            yaxis=dict(  
                showgrid=True,  
                showline=True,  
                linecolor='black',  
                gridcolor='lightgrey',  
                tickfont=dict(size=12),  
                ticks='outside'  
            )  
        )  

        # Customize colors and sizes if needed  
        # Currently using predefined colors and sizes

        # Show updated plot  
        st.plotly_chart(fig, use_container_width=True)  


def main():  
    st.set_page_config(  
        page_title="ELISA Plate Data Annotator and Analyzer",  
        layout="wide"  
    )  
    st.title("ELISA Plate Data Annotator and Analyzer")  

    initialize_session_state()  

    steps = [  
        "Upload Data",  
        "Annotate Conditions",  
        "Plate Modification/Normalization",  
        "Process Data",  
        "Display Results"  
    ]  

    st.sidebar.title("Navigation")  
    selected_step = st.sidebar.radio(  
        "Go to",  
        options=steps,  
        index=steps.index(st.session_state.current_step)  
    )  

    # Enforce prerequisites for each step  
    prerequisites = {  
        "Annotate Conditions": len(st.session_state.raw_data) > 0,  
        "Plate Modification/Normalization": len(st.session_state.conditions_list) == len(st.session_state.raw_data),  
        "Process Data": len(st.session_state.conditions_list) == len(st.session_state.raw_data),  
        "Display Results": not st.session_state.final_results.empty  
    }  

    if selected_step in prerequisites and not prerequisites[selected_step]:  
        st.sidebar.warning(f"Please complete the previous steps before accessing {selected_step}.")  
        selected_step = st.session_state.current_step  

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
    if len(st.session_state.conditions_list) == len(st.session_state.raw_data):  
        progress += 20  
    if 'normalized_data' in st.session_state and st.session_state.normalized_data:  
        progress += 20  
    if not st.session_state.final_results.empty:  
        progress += 40  
    st.sidebar.progress(progress)  

if __name__ == "__main__":  
    main()