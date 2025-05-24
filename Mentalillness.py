import csv
import time
import json
import os
from tqdm import tqdm
import google.generativeai as genai
from neo4j import GraphDatabase

# You'll need to define these variables before running the script
NEO4J_URI="neo4j+s://f305e72f.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="2PT1aXnSBZcXHMUL9dd-nIzLUtCmKwD0jQ_8YA5fbjg"

CSV_FILE_PATH = "Mentalillness.csv"
# Configuration
REQUEST_INTERVAL = 2  # Seconds between API calls to avoid rate limiting

# Initialize Neo4j connection
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Define the only valid disorders - everything else is a symptom
VALID_DISORDERS = [
    "Generalized Anxiety Disorder",
    "Bipolar Disorder",
    "Major Depressive Disorder",
    "Panic Disorder"
]

# Define fixed relationship types to ensure consistency
RELATIONSHIP_TYPES = {
    "patient_to_symptom": "REPORTS",
    "patient_to_behavior": "DISPLAYS",
    "patient_to_disorder": "DIAGNOSED_WITH",
    "disorder_to_symptom": "EXHIBITS",
    "symptom_to_behavior": "MANIFESTS_AS",
    "disorder_to_disorder": "COMORBID_WITH"
}

# Map column names to proper disorder names
DISORDER_MAPPING = {
    "Depression": "Major Depressive Disorder",
    "Bipolar disorder": "Bipolar Disorder",
    "Anxiety disorder": "Generalized Anxiety Disorder",
    "Schizophrenia": "Schizophrenia",  # Not in valid disorders, treated as symptom
    "PTSD": "PTSD"  # Not in valid disorders, treated as symptom
}

# Rate limiter function
def rate_limited_request(prompt):
    """Make API request with rate limiting"""
    try:
        response = invoke(prompt)
        time.sleep(REQUEST_INTERVAL)  # Wait to avoid hitting rate limits
        return response
    except Exception as e:
        print(f"Error with Gemma API: {e}")
        time.sleep(REQUEST_INTERVAL * 2)  # Wait longer on errors
        return None

# Create Neo4j constraints
def create_constraints(tx):
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Disorder) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Symptom) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Behavior) REQUIRE b.name IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TherapyType) REQUIRE t.name IS UNIQUE"
    ]

    for constraint in constraints:
        try:
            tx.run(constraint)
        except Exception as e:
            print(f"Warning: Constraint creation issue: {e}")

# Execute Cypher query
def execute_query(query):
    """Execute Cypher query in Neo4j"""
    try:
        with neo4j_driver.session() as session:
            result = session.run(query)
            return result
    except Exception as e:
        print(f"Neo4j error: {e}")
        print(f"Failed query: {query}")
        return None

def process_csv_row(row):
    """Process a row from the CSV and classify columns"""
    patient_id = row.get("ID", "unknown")

    symptoms = []
    behaviors = []
    disorders = []

    # Check if the patient has any of the four valid disorders directly
    has_disorder = False

    # Process each column in the row
    for column, value in row.items():
        if column == "ID":
            continue

        # Skip empty values
        if not value or value == "0":
            continue

        # Map column name to proper disorder name if applicable
        if column in DISORDER_MAPPING:
            mapped_name = DISORDER_MAPPING[column]

            # Check if this is one of the valid disorders
            if mapped_name in VALID_DISORDERS:
                disorders.append(mapped_name)
                has_disorder = True
            else:
                # If mapped but not a valid disorder, it's a symptom
                symptoms.append(mapped_name)
        else:
            # Classify remaining columns as symptoms or behaviors
            if any(keyword in column.lower() for keyword in ["mood", "feeling", "distress", "disturbance", "memories", "flashback", "traumatic", "emotional"]):
                symptoms.append(column)
            else:
                behaviors.append(column)

    # If no disorder was found, assign one based on symptom patterns
    if not has_disorder:
        # This is a simplified heuristic - you might want to implement more sophisticated logic
        depression_symptoms = ["Depressed mood", "Persistent sadness or low mood", "Loss of interest or pleasure in activities",
                              "Fatigue or loss of energy", "Difficulty concentrating or making decisions"]
        anxiety_symptoms = ["Excessive worry or fear", "Restlessness", "Irritability"]
        bipolar_symptoms = ["Inflated self-esteem", "Racing thoughts", "Decreased need for sleep",
                           "More talkative than usual", "Increase in goal-directed activity"]

        # Count matching symptoms for each disorder
        depression_count = sum(1 for s in depression_symptoms if s in symptoms)
        anxiety_count = sum(1 for s in anxiety_symptoms if s in symptoms)
        bipolar_count = sum(1 for s in bipolar_symptoms if s in symptoms)

        # Assign disorder with most matching symptoms
        if depression_count >= anxiety_count and depression_count >= bipolar_count:
            disorders.append("Major Depressive Disorder")
        elif anxiety_count >= depression_count and anxiety_count >= bipolar_count:
            disorders.append("Generalized Anxiety Disorder")
        else:
            disorders.append("Bipolar Disorder")

    return {
        "patient_id": patient_id,
        "symptoms": symptoms,
        "behaviors": behaviors,
        "disorders": disorders
    }

def generate_gemma_prompt(patient_data):
    """Generate a prompt for Gemma to create a Cypher query"""
    prompt = f"""
    Create a Neo4j Cypher query to insert the following mental health data:

    Patient ID: {patient_data['patient_id']}

    Symptoms: {', '.join(patient_data['symptoms'])}

    Behaviors: {', '.join(patient_data['behaviors'])}

    Disorders: {', '.join(patient_data['disorders'])}

    Use the following graph schema:
    1. (Patient) node with properties: id
    2. (Symptom) nodes with properties: name
    3. (Behavior) nodes with properties: name
    4. (Disorder) nodes with properties: name

    And use ONLY these exact relationship types:
    1. (Patient)-[REPORTS]->(Symptom)
    2. (Patient)-[DISPLAYS]->(Behavior)
    3. (Patient)-[DIAGNOSED_WITH]->(Disorder)
    4. (Disorder)-[EXHIBITS]->(Symptom)
    5. (Symptom)-[MANIFESTS_AS]->(Behavior)
    6. (Disorder)-[COMORBID_WITH]->(Disorder)

    IMPORTANT: Only these are valid disorders: Generalized Anxiety Disorder, Bipolar Disorder, Major Depressive Disorder, and Panic Disorder. Everything else must be treated as a symptom.

    The query should:
    1. Create the Patient node if it doesn't exist
    2. Create Symptom, Behavior, and Disorder nodes if they don't exist
    3. Create appropriate relationships between entities using ONLY the relationship types listed above
    4. Use MERGE for all node creation to avoid duplicates
    5. Ensure that the Patient has a DIAGNOSED_WITH relationship to each Disorder in the list

    Return ONLY the Cypher query with no explanations.
    """
    return prompt

def create_direct_cypher_query(patient_data):
    """Create a Cypher query directly without using Gemini API"""
    patient_id = patient_data['patient_id']
    symptoms = patient_data['symptoms']
    behaviors = patient_data['behaviors']
    disorders = patient_data['disorders']

    # Start building the query
    query = """
    // Create patient node
    MERGE (p:Patient {id: $patient_id})

    // Create disorder nodes and relationships
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

    # Add behaviors
    for i, behavior in enumerate(behaviors):
        query += f"""
    MERGE (b{i}:Behavior {{name: $behavior_{i}}})
    MERGE (p)-[:DISPLAYS]->(b{i})
        """

        # Connect some symptoms to behaviors
        if i < len(symptoms):
            query += f"""
    MERGE (s{i})-[:MANIFESTS_AS]->(b{i})
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
    params = {'patient_id': patient_id}

    # Add disorder parameters
    for i, disorder in enumerate(disorders):
        params[f'disorder_{i}'] = disorder

    # Add symptom parameters
    for i, symptom in enumerate(symptoms):
        params[f'symptom_{i}'] = symptom

    # Add behavior parameters
    for i, behavior in enumerate(behaviors):
        params[f'behavior_{i}'] = behavior

    return query, params

def process_csv_file():
    """Process the entire CSV file row by row"""
    print("Setting up Neo4j constraints...")
    with neo4j_driver.session() as session:
        session.execute_write(create_constraints)

    print(f"Processing CSV file: {CSV_FILE_PATH}")
    rows_processed = 0
    success_count = 0

    try:
        with open(CSV_FILE_PATH, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            total_rows = sum(1 for line in open(CSV_FILE_PATH)) - 1  # Subtract header

            # Reset file pointer
            csvfile.seek(0)
            reader = csv.DictReader(csvfile)

            # Process each row with progress bar
            for row in tqdm(reader, total=total_rows, desc="Processing patients"):
                rows_processed += 1

                # Process the row data
                patient_data = process_csv_row(row)

                # Skip if no meaningful data
                if not patient_data['symptoms'] and not patient_data['behaviors']:
                    print(f"Skipping row {rows_processed} - No meaningful data")
                    continue

                # Ensure patient has at least one disorder
                if not patient_data['disorders']:
                    # Assign a default disorder if none detected
                    patient_data['disorders'] = ["Major Depressive Disorder"]
                    print(f"Patient {patient_data['patient_id']} had no disorders, assigning default.")

                # Choose method: Direct Cypher generation or Gemma API
                use_direct_cypher = True  # Set to False to use Gemma/Gemini API

                if use_direct_cypher:
                    # Generate Cypher query directly
                    cypher_query, params = create_direct_cypher_query(patient_data)

                    # Execute the Cypher query
                    print(f"Executing Cypher query for patient {patient_data['patient_id']}...")

                    try:
                        with neo4j_driver.session() as session:
                            result = session.run(cypher_query, params)
                            success_count += 1
                            print(f"Successfully processed patient {patient_data['patient_id']}")
                    except Exception as e:
                        print(f"Neo4j error for patient {patient_data['patient_id']}: {e}")

                else:
                    # Generate prompt for Gemma
                    prompt = generate_gemma_prompt(patient_data)

                    # Get Cypher query from Gemma
                    print(f"\nRequesting Cypher query for patient {patient_data['patient_id']}...")
                    cypher_query = rate_limited_request(prompt)

                    if not cypher_query:
                        print(f"Failed to get response for patient {patient_data['patient_id']}")
                        continue

                    # Clean up the query (remove markdown code blocks if present)
                    cypher_query = cypher_query.strip()
                    if cypher_query.startswith("```cypher"):
                        cypher_query = cypher_query.split("```")[1]
                    elif cypher_query.startswith("```"):
                        cypher_query = cypher_query[3:]
                    if cypher_query.endswith("```"):
                        cypher_query = cypher_query[:-3]

                    cypher_query = cypher_query.strip()

                    # Execute the Cypher query
                    print(f"Executing Cypher query for patient {patient_data['patient_id']}...")
                    result = execute_query(cypher_query)

                    if result:
                        success_count += 1
                        print(f"Successfully processed patient {patient_data['patient_id']}")

                # Additional delay between processing rows
                time.sleep(0.5)

    except Exception as e:
        print(f"Error processing CSV: {e}")

    print(f"\nProcessing complete! Successfully processed {success_count} out of {rows_processed} rows.")

def main():
    start_time = time.time()
    print("=" * 80)
    print("MENTAL HEALTH DATA PROCESSING SCRIPT")
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