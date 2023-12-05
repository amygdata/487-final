import pandas as pd

def process_file(file_path1, file_path2):
    # Read the TSV file into a DataFrame
    df1 = pd.read_csv(file_path1, sep='\t')
    print(df1.columns)
    df2 = pd.read_csv(file_path2, sep='\t')

    for index, row in df1.iterrows():
        if row['Stance'] == 4:
            print(f'Text: {row["Text"]}')
            response = input('Enter response (0, 1, 2, d, f): ')

            if response in ['1', '2', '0']:
                # Update Stance for both files
                df1.at[index, 'Stance'] = int(response)
                df1.to_csv(file_path1, sep='\t', index=False)  # Save changes immediately
                df2.at[index, 'Stance'] = int(response)
                df2.to_csv(file_path2, sep='\t', index=False)  # Save changes immediately

            elif response == 'd':
                # Delete the row from both files
                df1 = df1.drop(index)
                df1.to_csv(file_path1, sep='\t', index=False)  # Save changes immediately
                df2 = df2.drop(index)
                df2.to_csv(file_path2, sep='\t', index=False)  # Save changes immediately

            elif response == 'f':
                # Print Text from the second file
                print(f'Text from second file: {df2.at[index, "Text"]}')
                second_response = input('Enter response (d, 0, 1, 2): ')

                if second_response in ['1', '2', '0']:
                    # Update Stance for both files
                    df1.at[index, 'Stance'] = int(second_response)
                    df1.to_csv(file_path1, sep='\t', index=False)  # Save changes immediately
                    df2.at[index, 'Stance'] = int(second_response)
                    df2.to_csv(file_path2, sep='\t', index=False)  # Save changes immediately

                elif second_response == 'd':
                    # Delete the row from both files
                    df1 = df1.drop(index)
                    df1.to_csv(file_path1, sep='\t', index=False)  # Save changes immediately
                    df2 = df2.drop(index)
                    df2.to_csv(file_path2, sep='\t', index=False)  # Save changes immediately

# Replace 'first_file.tsv' with the actual path to your first file
process_file('reddit_title_data.txt', 'reddit_body_data.txt')
