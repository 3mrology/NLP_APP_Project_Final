import csv
import time
import os
from tqdm import tqdm
from neo4j import GraphDatabase

# You'll need to define these variables before running the script
NEO4J_URI="neo4j+s://f305e72f.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="2PT1aXnSBZcXHMUL9dd-nIzLUtCmKwD0jQ_8YA5fbjg"

CSV_FILE_PATH = "depression_anxiety_data.csv"
# Configuration
REQUEST_INTERVAL = 2  # Seconds between API calls to avoid rate limiting

# Initialize Neo4j connection
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define the only valid disorders
VALID_DISORDERS = [
    "Generalized Anxiety Disorder",
    "Major Depressive Disorder"
]

# Define weight problem categories
WEIGHT_CATEGORIES = {
    "Class I Obesity": "Obesity",
    "Class II Obesity": "Severe Obesity",
    "Class III Obesity": "Morbid Obesity",
    "Overweight": "Overweight",
    "Underweight": "Underweight"
}

# Define fixed relationship types to ensure consistency
RELATIONSHIP_TYPES = {
    "patient_to_symptom": "REPORTS",
    "patient_to_disorder": "DIAGNOSED_WITH",
    "patient_to_weight_problem": "HAS_CONDITION",
    "disorder_to_symptom": "EXHIBITS",
    "patient_to_treatment": "RESPONDS_TO"
}

# Create Neo4j constraints
def create_constraints(tx):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disorder) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TherapyType) REQUIRE t.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (w:WeightCondition) REQUIRE w.name IS UNIQUE"
    ]

    for constraint in constraints:
        try:
            tx.run(constraint)
        except Exception as e:
            print(f"Warning: Constraint creation issue: {e}")

# Process a row from the CSV and classify data
def process_csv_row(row, row_number):
    """Process a row from the CSV and classify columns"""
    original_id = row.get("id", "0")

    # Generate a new unique ID with offset to avoid conflicts
    # Use 10000000 as offset to ensure we're far from existing IDs
    try:
        patient_id = str(10000000 + int(original_id))
    except ValueError:
        # If original_id is not a number, use row_number with offset
        patient_id = str(10000000 + row_number)

    # Only process rows where depressiveness or anxiousness is TRUE
    if row.get("depressiveness", "FALSE").upper() != "TRUE" and row.get("anxiousness", "FALSE").upper() != "TRUE":
        return None

    # Initialize data structures
    symptoms = []
    disorders = []
    treatments = []
    weight_condition = None

    # Extract patient attributes
    patient_attrs = {
        "id": patient_id,
        "original_id": original_id,  # Keep original ID as reference
        "age": row.get("age", ""),
        "gender": row.get("gender", ""),
        "bmi": row.get("bmi", ""),
        "school_year": row.get("school_year", "")
    }

    # Process depression data
    if row.get("depressiveness", "FALSE").upper() == "TRUE":
        disorders.append("Major Depressive Disorder")
        symptoms.append(f"Depression (PHQ Score: {row.get('phq_score', 'N/A')})")
        symptoms.append(f"Depression Severity: {row.get('depression_severity', 'N/A')}")

        # Add depression treatment if available
        if row.get("depression_treatment", "FALSE").upper() == "TRUE":
            treatments.append("Depression Treatment")

    # Process anxiety data
    if row.get("anxiousness", "FALSE").upper() == "TRUE":
        disorders.append("Generalized Anxiety Disorder")
        symptoms.append(f"Anxiety (GAD Score: {row.get('gad_score', 'N/A')})")
        symptoms.append(f"Anxiety Severity: {row.get('anxiety_severity', 'N/A')}")

        # Add anxiety treatment if available
        if row.get("anxiety_treatment", "FALSE").upper() == "TRUE":
            treatments.append("Anxiety Treatment")

    # Process BMI data for weight condition
    who_bmi = row.get("who_bmi", "")

    if who_bmi and who_bmi != "Normal":
        # Map WHO BMI category to weight condition
        if who_bmi in WEIGHT_CATEGORIES:
            weight_condition = WEIGHT_CATEGORIES[who_bmi]
        else:
            weight_condition = who_bmi  # Use as is if not in mapping

    # Process suicidal thoughts as symptom
    if row.get("suicidal", "FALSE").upper() == "TRUE":
        symptoms.append("Suicidal Thoughts")

    # Process sleepiness as symptom
    if row.get("sleepiness", "FALSE").upper() == "TRUE":
        symptoms.append(f"Excessive Sleepiness (Epworth Score: {row.get('epworth_score', 'N/A')})")

    return {
        "patient_attrs": patient_attrs,
        "symptoms": symptoms,
        "disorders": disorders,
        "treatments": treatments,
        "weight_condition": weight_condition
    }

def create_direct_cypher_query(patient_data):
    """Create a Cypher query directly without using external API"""
    patient_attrs = patient_data['patient_attrs']
    symptoms = patient_data['symptoms']
    disorders = patient_data['disorders']
    treatments = patient_data['treatments']
    weight_condition = patient_data['weight_condition']

    # Create patient attributes string for Neo4j
    attrs_string = ", ".join([f"{k}: $patient_{k}" for k in patient_attrs.keys()])

    # Start building the query
    query = f"""
    // Create patient node with attributes
    MERGE (p:Patient {{{attrs_string}}})
    """

    # Add disorders
    for i, disorder in enumerate(disorders):
        query += f"""
    MERGE (d{i}:Disorder {{name: $disorder_{i}}})
    MERGE (p)-[:DIAGNOSED_WITH]->(d{i})
        """

    # Add symptoms
    for i, symptom in enumerate(symptoms):
        query += f"""
    MERGE (s{i}:Symptom {{name: $symptom_{i}}})
    MERGE (p)-[:REPORTS]->(s{i})
        """

        # Connect relevant disorders to symptoms
        for j, disorder in enumerate(disorders):
            query += f"""
    MERGE (d{j})-[:EXHIBITS]->(s{i})
            """

    # Add treatments
    for i, treatment in enumerate(treatments):
        query += f"""
    MERGE (t{i}:TherapyType {{name: $treatment_{i}}})
    MERGE (p)-[:RESPONDS_TO]->(t{i})
        """

    # Add weight condition if present
    if weight_condition:
        query += f"""
    MERGE (w:WeightCondition {{name: $weight_condition}})
    MERGE (p)-[:HAS_CONDITION]->(w)
        """

    # Add comorbidity relationships between disorders if more than one
    if len(disorders) > 1:
        for i in range(len(disorders)):
            for j in range(i+1, len(disorders)):
                query += f"""
    MERGE (d{i})-[:COMORBID_WITH]->(d{j})
    MERGE (d{j})-[:COMORBID_WITH]->(d{i})
                """

    # Create parameter dict
    params = {}

    # Add patient attribute parameters
    for k, v in patient_attrs.items():
        params[f'patient_{k}'] = v

    # Add disorder parameters
    for i, disorder in enumerate(disorders):
        params[f'disorder_{i}'] = disorder

    # Add symptom parameters
    for i, symptom in enumerate(symptoms):
        params[f'symptom_{i}'] = symptom

    # Add treatment parameters
    for i, treatment in enumerate(treatments):
        params[f'treatment_{i}'] = treatment

    # Add weight condition parameter
    if weight_condition:
        params['weight_condition'] = weight_condition

    return query, params

def process_csv_file():
    """Process the entire CSV file row by row"""
    print("Setting up Neo4j constraints...")
    with neo4j_driver.session() as session:
        session.execute_write(create_constraints)

    print(f"Processing CSV file: {CSV_FILE_PATH}")
    rows_processed = 0
    success_count = 0
    filtered_count = 0

    try:
        with open(CSV_FILE_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for line in open(CSV_FILE_PATH)) - 1  # Subtract header

            # Reset file pointer
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)

            # Process each row with progress bar
            for row_num, row in enumerate(tqdm(reader, total=total_rows, desc="Processing patients")):
                rows_processed += 1

                # Process the row data with row number for unique ID generation
                patient_data = process_csv_row(row, row_num + 1)  # +1 to avoid zero-indexing

                # Skip if filtered out (not depressive or anxious)
                if not patient_data:
                    filtered_count += 1
                    continue

                # Skip if no meaningful data
                if not patient_data['symptoms'] and not patient_data['disorders']:
                    print(f"Skipping row {rows_processed} - No meaningful data")
                    continue

                # Generate Cypher query directly
                cypher_query, params = create_direct_cypher_query(patient_data)

                # Execute the Cypher query
                original_id = patient_data['patient_attrs']['original_id']
                new_id = patient_data['patient_attrs']['id']
                print(f"Executing Cypher query for patient {original_id} (new ID: {new_id})...")

                try:
                    with neo4j_driver.session() as session:
                        result = session.run(cypher_query, params)
                        success_count += 1
                        print(f"Successfully processed patient {original_id} (new ID: {new_id})")
                except Exception as e:
                    print(f"Neo4j error for patient {original_id}: {e}")

                # Additional delay between processing rows
                time.sleep(0.5)

    except Exception as e:
        print(f"Error processing CSV: {e}")

    print(f"\nProcessing complete! Successfully processed {success_count} out of {rows_processed} rows.")
    print(f"Filtered out {filtered_count} rows (not depressive or anxious).")

def main():
    start_time = time.time()
    print("=" * 80)
    print("STUDENT MENTAL HEALTH DATA PROCESSING SCRIPT")
    print("=" * 80)

    process_csv_file()

    # Close Neo4j connection
    neo4j_driver.close()

    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print("=" * 80)
    print(f"Processing complete! Time elapsed: {int(minutes)}m {int(seconds)}s")
    print("=" * 80)

if __name__ == "__main__":
    main()