import pandas as pd

def calculate_aac(sequence):
    # Define the standard amino acids
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    total_count = len(sequence)
    
    # Calculate the AAC for each amino acid
    aac = {aa: sequence.count(aa) / total_count for aa in amino_acids}
    
    # Return as a vector in alphabetical order of amino acids
    return [aac[aa] for aa in amino_acids]

def process_csv(input_file, output_file):
    # Read input CSV
    df = pd.read_csv(input_file)
    
    # Ensure there is a column named 'sequence'
    if 'Sequence' not in df.columns:
        raise ValueError("Input CSV must contain a column named 'sequence'")
    
    # Compute AAC for each sequence
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    aac_vectors = df['Sequence'].apply(calculate_aac)
    
    # Expand AAC vectors into separate columns
    aac_df = pd.DataFrame(aac_vectors.tolist(), columns=list(amino_acids))
    
    # Concatenate the original dataframe with AAC columns
    result_df = pd.concat([df, aac_df], axis=1)
    
    # Save the result to output CSV
    result_df.to_csv(output_file, index=False)

# Example usage for multiple files
file_pairs = [
    ('ACP_full.csv', 'ACP_full_aac.csv'),
]

for input_file, output_file in file_pairs:
    process_csv(input_file, output_file)
    print(f"Finished processing {input_file} -> {output_file}")
