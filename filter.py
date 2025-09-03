import os
import time
import yaml
import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Load Config ---------------- #
with open("config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

RAW_PATH = CONFIG["data"]["raw_path"]
CURATED_PATH = CONFIG["data"]["curated_path"]
SCHEMA_PATH = CONFIG["schema"]["save_path"]
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(CURATED_PATH, exist_ok=True)
os.makedirs(SCHEMA_PATH, exist_ok=True)


# ---------------- Helpers ---------------- #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Generic cleaning: drop duplicates, fill NA."""
    df = df.drop_duplicates()
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df

def save_as_parquet(df: pd.DataFrame, path: str):
    """Save dataframe as parquet with compression."""
    df.to_parquet(path, engine="pyarrow", compression=CONFIG["storage"]["compression"])

def load_previous_schema():
    """Load last saved schema for drift detection."""
    files = sorted([f for f in os.listdir(SCHEMA_PATH) if f.endswith(".json")])
    if not files:
        return None
    with open(os.path.join(SCHEMA_PATH, files[-1]), "r") as f:
        return json.load(f)

def detect_drift(old_schema, new_schema):
    """Compare schemas and detect anomalies/drift."""
    if not old_schema:
        return []
    drift = []
    if old_schema["fact_table"] != new_schema["fact_table"]:
        drift.append("Fact table structure changed.")
    if set(old_schema["dimensions"].keys()) != set(new_schema["dimensions"].keys()):
        drift.append("Dimension tables changed.")
    return drift

def versioned_schema_save(schema):
    """Save schema with versioning and history limit."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    schema_file = os.path.join(SCHEMA_PATH, f"schema_{timestamp}.json")
    with open(schema_file, "w") as f:
        json.dump(schema, f, indent=4)

    # enforce history limit
    if CONFIG["schema"]["versioning"]:
        files = sorted([f for f in os.listdir(SCHEMA_PATH) if f.endswith(".json")])
        if len(files) > CONFIG["drift"]["history_limit"]:
            os.remove(os.path.join(SCHEMA_PATH, files[0]))
    return schema_file

def generate_schema_diagram(schema, save_path):
    """Visualize schema as PNG using NetworkX."""
    G = nx.DiGraph()

    # Fact table node
    G.add_node("Fact", shape="square")

    # Dimensions
    for dim in schema["dimensions"].keys():
        G.add_node(dim)
        G.add_edge("Fact", dim)

    # Hierarchies
    for h in schema.get("hierarchies", []):
        parent, child = f"dim_{h['parent']}", f"dim_{h['child']}"
        if parent in G.nodes and child in G.nodes:
            G.add_edge(parent, child)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_weight="bold")
    plt.savefig(save_path, format="png")
    plt.close()
    logging.info(f"Schema diagram saved → {save_path}")


def auto_schema_design(df: pd.DataFrame, curated_path: str, schema_path: str):
    """
    Analyze dataset and generate star/snowflake schema automatically.
    Generic → detects hierarchies in categorical columns (not hardcoded).
    """
    schema = {"fact_table": None, "dimensions": {}, "hierarchies": [], "dimensionality": None}

    # Identify column types
    id_like = [col for col in df.columns if "id" in col.lower() or "key" in col.lower()]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    if not id_like:
        id_like = [df.columns[0]]  # assume first col as identifier

    # Fact table = IDs + numbers
    fact_cols = list(set(id_like + num_cols))
    fact_table = df[fact_cols].dropna(axis=1, how="all")
    schema["fact_table"] = list(fact_table.columns)
    save_as_parquet(fact_table, os.path.join(curated_path, "fact_table.parquet"))

    # Dimension tables
    for col in cat_cols:
        if col not in fact_cols:
            dim_name = f"dim_{col}"
            dim_table = df[[col]].drop_duplicates().reset_index(drop=True)
            save_as_parquet(dim_table, os.path.join(curated_path, f"{dim_name}.parquet"))
            schema["dimensions"][dim_name] = [col]

    # Snowflake detection (hierarchies between categorical columns)
    for higher in cat_cols:
        for lower in cat_cols:
            if higher != lower:
                combos = df[[higher, lower]].drop_duplicates()
                if combos[higher].nunique() == combos.shape[0]:  # unique mapping higher → lower
                    schema["hierarchies"].append({"parent": higher, "child": lower})

    # Dimensionality
    schema["dimensionality"] = f"{len(schema['dimensions']) + 1}D"

    # Drift detection
    if CONFIG["drift"]["enable"]:
        old_schema = load_previous_schema()
        drift = detect_drift(old_schema, schema)
        if drift:
            schema["drift_detected"] = drift
            logging.warning(f"Data drift detected: {drift}")

    # Save versioned schema
    schema_file = versioned_schema_save(schema)

    # Generate schema diagram
    diagram_file = os.path.join(schema_path, f"diagram_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    generate_schema_diagram(schema, diagram_file)


def process_file(file_path: str):
    """Ingest, clean, save, and schema-design a file (generic)."""
    try:
        logging.info(f"Processing {file_path} ...")
        df = pd.read_csv(file_path)
        df = clean_data(df)

        curated_file = os.path.join(CURATED_PATH, os.path.basename(file_path).replace(".csv", ".parquet"))
        save_as_parquet(df, curated_file)
        logging.info(f"Saved curated parquet → {curated_file}")

        if CONFIG["schema"]["enable_auto"]:
            auto_schema_design(df, CURATED_PATH, SCHEMA_PATH)

    except Exception as e:
        logging.error(f"Failed processing {file_path}: {e}")


# ---------------- Multi-Source Ingestion ---------------- #
def ingest_sources():
    for source in CONFIG["ingestion"]["sources"]:
        if source == "csv":
            for fname in os.listdir(RAW_PATH):
                if fname.endswith(".csv"):
                    process_file(os.path.join(RAW_PATH, fname))
        elif source == "api":
            logging.info("API ingestion placeholder (implement later).")
        elif source == "huggingface":
            logging.info("Hugging Face ingestion placeholder (implement later).")
        elif source == "kaggle":
            logging.info("Kaggle ingestion placeholder (implement later).")


# ---------------- Continuous Ingestion ---------------- #
def watch_folder():
    logging.info("Starting continuous ingestion...")
    seen_files = set()

    while CONFIG["ingestion"]["continuous"]:
        ingest_sources()

        for fname in os.listdir(RAW_PATH):
            if fname.endswith(".csv"):
                fpath = os.path.join(RAW_PATH, fname)
                if fpath not in seen_files:
                    process_file(fpath)
                    seen_files.add(fpath)

        time.sleep(CONFIG["ingestion"]["poll_interval"])


if __name__ == "__main__":
    watch_folder()

