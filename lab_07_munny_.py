import pandas as pd
import numpy as np
import os

class DataCleaner:

    def CheckIfFileExists(self, file_path):
        if not os.path.exists(file_path):
            print("The file does not exist. Exiting program.")
            exit()

    def CheckIfFileHasData(self, df):
        if df.empty:
            print("The file does not have any data. Exiting program.")
            exit()

    def GetMatrix(self, df):
        # Remove the first row (assuming it's the label)
        matrix = df.values[1:]
        print("The data has been successfully placed into a matrix.")
        return matrix

    def RemoveColsWithJunk(self, matrix):
        print(f"Before cleaning the junk columns, the total number of columns is: {matrix.shape[1]}")
        
        # Threshold: 70% of the rows must not be junk (NaN values)
        threshold = 0.7 * matrix.shape[0]
        
        # Function to determine if a column is 'junk'
        is_junk = lambda col: np.sum(np.isnan(pd.to_numeric(col, errors='coerce'))) > threshold
        
        # Filter out junk columns
        matrix_cleaned = matrix[:, [not is_junk(matrix[:, i]) for i in range(matrix.shape[1])]]
        
        print(f"After cleaning the junk columns, the total number of columns is: {matrix_cleaned.shape[1]}")
        return matrix_cleaned

    def RemoveColsWithZero(self, matrix):
        print(f"Before cleaning the zero columns, the total number of columns is: {matrix.shape[1]}")
        
        threshold = 0.7 * matrix.shape[0]
        is_zero = lambda col: np.sum(col == 0) > threshold
        
        matrix_cleaned = matrix[:, [not is_zero(matrix[:, i]) for i in range(matrix.shape[1])]]
        
        print(f"After cleaning the zero columns, the total number of columns is: {matrix_cleaned.shape[1]}")
        return matrix_cleaned

    def RemoveRowsWithInvalidIC50(self, matrix):
        print(f"Before cleaning rows with invalid IC50, the total number of rows is: {matrix.shape[0]}")
        
        # Convert the IC50 column (first column) to numeric, setting errors to NaN for non-numeric values
        ic50_column = pd.to_numeric(matrix[:, 0], errors='coerce')
        
        # Remove rows where IC50 is NaN or greater than 20000
        matrix_cleaned = matrix[(~np.isnan(ic50_column)) & (ic50_column < 20000)]
        
        print(f"After cleaning rows with invalid IC50, the total number of rows is: {matrix_cleaned.shape[0]}")
        return matrix_cleaned


    def SetOtherJunktoAverageOfTheColumn(self, matrix):
        print("Before handling junk values, calculating junk count.")
    
        nan_count_total = 0  # To keep track of total junk (NaN) values
    
        # Iterate over each column and handle junk values
        for i in range(matrix.shape[1]):
            column = pd.to_numeric(matrix[:, i], errors='coerce')  # Convert column to numeric, setting junk to NaN
            nan_count = np.isnan(column).sum()  # Count the NaN values in the current column
            nan_count_total += nan_count  # Add to total count of NaNs
    
            avg_value = np.nanmean(column)  # Calculate the column average, ignoring NaNs
            column[np.isnan(column)] = avg_value  # Replace NaNs with the average value
    
            matrix[:, i] = column  # Update the matrix with the cleaned column

        print(f"Total junk values before replacement: {nan_count_total}")
        print(f"After handling junk values, all remaining junk replaced with column averages.")
        
        return matrix

    def RescaleData(self, matrix):
        print("Starting rescaling of the data.")
    
        # Initialize the rescaled matrix with the same shape as the input matrix
        matrix_rescaled = np.zeros(matrix.shape)
    
        # Loop through each column to perform rescaling
        for i in range(matrix.shape[1]):
            col_min = matrix[:, i].min()
            col_max = matrix[:, i].max()
    
            if col_max - col_min == 0:
                # If the column has no range (max == min), set the column to a constant (e.g., 0)
                matrix_rescaled[:, i] = 0
                print(f"Column {i} has no range. Setting rescaled values to 0.")
            else:
                # Rescale the column
                matrix_rescaled[:, i] = (matrix[:, i] - col_min) / (col_max - col_min)
    
        print("The Data have been rescaled.")
        return matrix_rescaled


    def NormalizeData(self, matrix):
        print("Starting normalization of the data.")
        
        # Calculate mean and standard deviation
        mean = np.mean(matrix, axis=0)
        std_dev = np.std(matrix, axis=0)
        
        # Avoid division by zero by checking if std_dev is zero
        std_dev[std_dev == 0] = 1  # Set zero std_dev values to 1 to avoid division by zero
    
        # Normalize the matrix
        matrix_normalized = (matrix - mean) / std_dev
    
        print("The Data have been normalized.")
        return matrix_normalized

    def SortTheNormalizedData(self, matrix):
        # Sort by IC50 (assuming IC50 is the first column)
        matrix_sorted = matrix[matrix[:, 0].argsort()]
        print("The data is now sorted based on IC50.")
        return matrix_sorted

    def GetTrainingData(self, matrix):
        # Get training data (rows 1, 2, 5, 6, 9, 10, etc.)
        training_data = matrix[::2]  # Select every 2nd row
        print("Training data created.")
        return training_data

    def GetValidationData(self, matrix):
        # Get validation data (rows 3, 7, 11, etc.)
        validation_data = matrix[1::4]  # Select every 4th row starting from row 3
        print("Validation data created.")
        return validation_data

    def GetTestingData(self, matrix):
        # Get testing data (rows 4, 8, 12, etc.)
        testing_data = matrix[3::4]  # Select every 4th row starting from row 4
        print("Testing data created.")
        return testing_data

def main():
    file_path = './XandY.xlsx'  # Replace this with your actual file path
    data_cleaner = DataCleaner()

    # Step 1: Check if the file exists
    data_cleaner.CheckIfFileExists(file_path)

    # Step 2: Load the data into a pandas DataFrame
    df = pd.read_excel(file_path)

    # Step 3: Check if the file has data
    data_cleaner.CheckIfFileHasData(df)

    # Step 4: Process the matrix
    matrix = data_cleaner.GetMatrix(df)
    matrix = data_cleaner.RemoveColsWithJunk(matrix)
    matrix = data_cleaner.RemoveColsWithZero(matrix)
    matrix = data_cleaner.RemoveRowsWithInvalidIC50(matrix)
    matrix = data_cleaner.SetOtherJunktoAverageOfTheColumn(matrix)

    # Step 5: Rescale, normalize, and sort the data
    matrix_rescaled = data_cleaner.RescaleData(matrix)
    matrix_normalized = data_cleaner.NormalizeData(matrix_rescaled)
    matrix_sorted = data_cleaner.SortTheNormalizedData(matrix_normalized)

    # Step 6: Get training, validation, and testing data
    training_data = data_cleaner.GetTrainingData(matrix_sorted)
    validation_data = data_cleaner.GetValidationData(matrix_sorted)
    testing_data = data_cleaner.GetTestingData(matrix_sorted)

    # Save results to an Excel file with multiple sheets
    with pd.ExcelWriter("CleanedData.xlsx") as writer:
        pd.DataFrame(matrix).to_excel(writer, sheet_name='Cleaned')
        pd.DataFrame(matrix_rescaled).to_excel(writer, sheet_name='Rescaled')
        pd.DataFrame(matrix_normalized).to_excel(writer, sheet_name='Normalized')
        pd.DataFrame(matrix_sorted).to_excel(writer, sheet_name='Sorted')
        pd.DataFrame(training_data).to_excel(writer, sheet_name='TrainedData')
        pd.DataFrame(validation_data).to_excel(writer, sheet_name='ValidatedData')
        pd.DataFrame(testing_data).to_excel(writer, sheet_name='TestingData')

if __name__ == "__main__":
    main()

