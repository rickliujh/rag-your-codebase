import os
import sys
import uuid

import git
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, Retrieval, Tool, VertexRagStore
import vertexai
from vertexai import rag
from concurrent.futures import ThreadPoolExecutor, as_completed


# @param {"type":"string", "placeholder": "https://github.com/google/adk-python"}
# GITHUB_URL = "https://github.com/google/adk-python"
GITHUB_URL = "https://github.com/rickliujh/Flowise.git"
# @param {type:"string", placeholder: "[your-project-id]", isTemplate: true}
PROJECT_ID = "ai-expr-alpha"
LOCATION = "us-central1"              # @param {type:"string"}
# @param {type:"string", placeholder: "[your-bucket-name]", isTemplate: true}
BUCKET_NAME = "code_for_index"
# @param {type:"string", placeholder: "rag-code-data", isTemplate: true}
GCS_FOLDER_PATH = "repo1"
MAX_FILE_SIZE_MB = 10  # @param {type:"number"}
# @param {type:"string", isTemplate: true}
EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"
# @param {type:"string", "placeholder": "gemini-2.5-flash", isTemplate: true}
MODEL_ID = "gemini-2.5-flash"

GCS_FOLDER_PATH = GCS_FOLDER_PATH.strip('/')
BUCKET_URI = f"gs://{BUCKET_NAME}"
GCS_UPLOAD_URI = f"{
    BUCKET_URI}/{GCS_FOLDER_PATH}" if GCS_FOLDER_PATH else BUCKET_URI
GCS_IMPORT_URI = f"{GCS_UPLOAD_URI}/"
LOCAL_REPO_PATH = "./cloned_repo"

# RAG Engine Configuration
_UUID = uuid.uuid4()
RAG_CORPUS_DISPLAY_NAME = f"rag-corpus-code-{_UUID}"
RAG_ENGINE_DISPLAY_NAME = f"rag-engine-code-{_UUID}"
MAX_WORKERS = os.cpu_count()

# Supported file extensions for ingestion (adjust as needed)
# Common code, config, and documentation file types
SUPPORTED_EXTENSIONS = [
    ".py", ".java", ".js", ".ts", ".go", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".md", ".txt", ".rst", ".html", ".css", ".scss",
    ".yaml", ".yml", ".json", ".xml", ".proto", "Dockerfile", ".sh",
    ".tf", ".tfvars", ".bicep", ".gradle", "pom.xml", "requirements.txt",
    "package.json", "go.mod", "go.sum", "Cargo.toml"
]


def main():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    if len(sys.argv) > 1 and sys.argv[1] == "q":
        rag_corpus = "shit"
        tool = create_retrieval_tool(rag_corpus)
        question(client, tool, sys.argv[2])

    storage_client = storage.Client(project=PROJECT_ID)
    clone_repo()
    bucket = verify_GCS_access(storage_client)
    rag_corpus = create_rag_corpus()
    multithreads_ingest_corpus(rag_corpus, LOCAL_REPO_PATH, bucket)

    # upload_file_GCS(bucket)
    # rag_corpus = create_rag_corpus()
    # ingest_files_into_corpus(rag_corpus)
    # create_retrieval_tool(rag_corpus)


def clone_repo():
    try:
        git.Repo.clone_from(GITHUB_URL, LOCAL_REPO_PATH)
    except git.GitCommandError as e:
        print(f"Error cloning repository: {e}")


def verify_GCS_access(storage_client):
    try:
        bucket = storage_client.get_bucket(BUCKET_NAME)
        print(bucket.name)
        return bucket
    except Exception as e:
        print(f"Error accessing GCS bucket '{BUCKET_NAME}': {e}")


def upload_file_GCS(bucket):
    uploaded_file_count = 0
    skipped_file_count = 0

    # Calculate max size in bytes once, if applicable
    max_bytes = 0
    if MAX_FILE_SIZE_MB > 0:
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        print(f"Applying max file size limit: {
              MAX_FILE_SIZE_MB} MB ({max_bytes} bytes)")

    for root, dirs, files in os.walk(LOCAL_REPO_PATH):
        # Skip '.git' directory explicitly to avoid uploading git metadata
        if '.git' in dirs:
            dirs.remove('.git')

        for file in files:
            file_lower = file.lower()  # Use lowercase for case-insensitive checks
            local_file_path = os.path.join(root, file)

            # Check if the file has a supported extension (case-insensitive)
            # or if the exact filename is in the list (e.g., "Dockerfile")
            is_supported = any(file_lower.endswith(ext.lower()) for ext in SUPPORTED_EXTENSIONS) or \
                file in SUPPORTED_EXTENSIONS  # Check exact match for non-extension files

            if is_supported:
                # Change extension to .txt so parser will accept
                old_file_path = local_file_path
                local_file_path += "txt"
                os.rename(old_file_path, local_file_path)

                try:
                    if max_bytes > 0:  # Only check if a limit is set
                        file_size_bytes = os.path.getsize(local_file_path)
                        if file_size_bytes > max_bytes:
                            print(f"  Skipping large file ({
                                  (file_size_bytes / (1024*1024)):.2f} MB > {MAX_FILE_SIZE_MB} MB): {local_file_path}")
                            skipped_file_count += 1
                            continue  # Skip to the next file

                    # Create a relative path to maintain structure in GCS
                    relative_path = os.path.relpath(
                        local_file_path, LOCAL_REPO_PATH)

                    # Construct the destination blob name within the specified GCS folder path
                    if GCS_FOLDER_PATH:
                        gcs_blob_name = os.path.join(
                            GCS_FOLDER_PATH, relative_path)
                    else:
                        gcs_blob_name = relative_path
                    gcs_blob_name = gcs_blob_name.replace("\\", "/")

                    # Get the blob object and upload the file
                    blob = bucket.blob(gcs_blob_name)
                    blob.upload_from_filename(local_file_path)
                    uploaded_file_count += 1

                    # Print progress periodically
                    if uploaded_file_count % 100 == 0:
                        print(f"  Uploaded {uploaded_file_count} files (skipped: {
                              skipped_file_count})...")

                except Exception as e:
                    print(f"  Error uploading {
                          local_file_path} to gs://{BUCKET_NAME}/{gcs_blob_name}: {e}")

    # --- Final Report ---
    print(f"\nFinished uploading.")
    print(f"Total supported files uploaded: {uploaded_file_count}")
    if skipped_file_count > 0:
        print(f"Total files skipped due to size limit: {skipped_file_count}")

    if uploaded_file_count == 0:
        print(f"\nWarning: No supported files were found (within size limits) in '{
              LOCAL_REPO_PATH}' or uploaded to '{GCS_UPLOAD_URI}'.")
        print("The RAG Engine will have no data from this source.")


def create_rag_corpus():
    rag_corpus = rag.create_corpus(
        display_name=RAG_CORPUS_DISPLAY_NAME,
        description=f"Codebase files from {GITHUB_URL}",
        backend_config=rag.RagVectorDbConfig(
            rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model=EMBEDDING_MODEL
                )
            )
        )
    )
    print(rag_corpus)
    return rag_corpus


def ingest_files_into_corpus(rag_corpus):
    print(GCS_IMPORT_URI)
    import_response = rag.import_files(
        corpus_name=rag_corpus.name,
        paths=[GCS_IMPORT_URI],
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=1024,
                chunk_overlap=256
            )
        ),
        timeout=None,
    )
    print(import_response.failed_rag_files_count)
    print(import_response.imported_rag_files_count)
    print(import_response.skipped_rag_files_count)
    print(import_response.partial_failures_bigquery_table)
    print(import_response.partial_failures_gcs_path)


def create_retrieval_tool(rag_corpus):
    rag_retrieval_tool = Tool(
        retrieval=Retrieval(
            vertex_rag_store=VertexRagStore(
                rag_corpora=[rag_corpus.name],
                similarity_top_k=10,
                vector_distance_threshold=0.5,
            )
        )
    )
    return rag_retrieval_tool


def _upload_single_file(
    local_file_abs_path: str,
    gcs_file_path: str,
    gcs_bucket: storage.Bucket,
) -> str:
    gcs_file_path += ".txt"
    print(f"  Renaming '{local_file_abs_path}' to '{
          gcs_file_path}' (in GCS) for upload.")

    blob = gcs_bucket.blob(gcs_file_path)
    blob.upload_from_filename(local_file_abs_path)

    print(f"Uploaded {local_file_abs_path} to {
          gcs_file_path}")

    full_gcs_uri = f"gs://{gcs_bucket.name}/{gcs_file_path}"
    return full_gcs_uri


def ingest_files_info_corpus_v2(rag_corpus, files_gcs_uri):
    print(GCS_IMPORT_URI)
    import_response = rag.import_files(
        corpus_name=rag_corpus.name,
        paths=files_gcs_uri,
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=1024,
                chunk_overlap=256
            )
        ),
        timeout=None,
    )
    print(import_response.failed_rag_files_count)
    print(import_response.imported_rag_files_count)
    print(import_response.skipped_rag_files_count)
    print(import_response.partial_failures_bigquery_table)
    print(import_response.partial_failures_gcs_path)
    return import_response


def multithreads_ingest_corpus(
    rag_corpus,
    folder,
    gcs_bucket,
):
    upload_futures = []
    ingest_futures = []
    gcs_files = []
    rag_corpus_limit = 25
    successful_uploads = 0
    failed_uploads = 0
    skipped_uploads = 0

    # Calculate max size in bytes once, if applicable
    max_bytes = 0
    if MAX_FILE_SIZE_MB > 0:
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        print(f"Applying max file size limit: {
              MAX_FILE_SIZE_MB} MB ({max_bytes} bytes)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for root, dirs, files in os.walk(folder):
            # Skip '.git' directory explicitly to avoid uploading git metadata
            if '.git' in dirs:
                dirs.remove('.git')

            for file in files:
                file_lower = file.lower()  # Use lowercase for case-insensitive checks
                local_file_path = os.path.join(root, file)

                # Check if the file has a supported extension (case-insensitive)
                # or if the exact filename is in the list (e.g., "Dockerfile")
                is_supported = any(file_lower.endswith(ext.lower()) for ext in SUPPORTED_EXTENSIONS) or \
                    file in SUPPORTED_EXTENSIONS  # Check exact match for non-extension files

                if is_supported:

                    if max_bytes > 0:  # Only check if a limit is set
                        file_size_bytes = os.path.getsize(local_file_path)
                        if file_size_bytes > max_bytes:
                            print(f"  Skipping large file ({
                                  (file_size_bytes / (1024*1024)):.2f} MB > {MAX_FILE_SIZE_MB} MB): {local_file_path}")
                            skipped_uploads += 1
                            continue  # Skip to the next file

                    # Create a relative path to maintain structure in GCS
                    relative_path = os.path.relpath(
                        local_file_path, LOCAL_REPO_PATH)

                    # Construct the destination blob name within the specified GCS folder path
                    if GCS_FOLDER_PATH:
                        gcs_file_path = os.path.join(
                            GCS_FOLDER_PATH, relative_path)
                    else:
                        gcs_file_path = relative_path
                    gcs_file_path = gcs_file_path.replace("\\", "/")

                    future = executor.submit(
                        _upload_single_file,
                        local_file_path,
                        gcs_file_path,
                        gcs_bucket,
                    )
                    upload_futures.append(future)

        for future in as_completed(upload_futures):
            try:
                uploaded_uri = future.result()  # Now receives the GCS URI
                # Optional: print each successful upload
                print(f"  {uploaded_uri} uploaded successfully.")
                successful_uploads += 1
                gcs_files.append(uploaded_uri)

                # Print progress periodically
                if successful_uploads % 100 == 0:
                    print(f"  Uploaded {successful_uploads} files (skipped: {
                        skipped_uploads}...")
            except Exception as exc:
                print(f"  File upload generated an exception: {exc}")
                failed_uploads += 1

    begin_inx = 0
    times = -(-len(gcs_files)//rag_corpus_limit)  # round up the integer
    print("times", gcs_files, len(gcs_files), rag_corpus_limit, times)
    for c in range(times):
        end_index = begin_inx+rag_corpus_limit
        end_index = end_index if end_index < len(gcs_files) else len(gcs_files)
        gcs_files_to_ingest = gcs_files[begin_inx:end_index]
        begin_inx = end_index

        print(f"batch ingest_file, No.{c}, {times - 1 - c} left...")
        res = ingest_files_info_corpus_v2(rag_corpus, gcs_files_to_ingest)
        print(f" result for No.{c}: failed_rag_files_count: {res.failed_rag_files_count}, imported_rag_files_count:{
              res.imported_rag_files_count}, skipped_rag_files_count: {res.skipped_rag_files_count}")


def question(client, rag_retrieval_tool, q):
    response = client.models.generate_content(
        model=MODEL_ID,
        contents="What is the primary purpose or main functionality of this codebase?",
        config=GenerateContentConfig(tools=[rag_retrieval_tool]),
    )

    print(response.text)


if __name__ == "__main__":
    main()
    print("finished")
