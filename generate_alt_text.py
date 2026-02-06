import argparse
import json
import csv
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from io import BytesIO
from pathlib import Path
import time
from datetime import datetime
import logging
from google import genai
from google.genai import types
from google.oauth2 import service_account
from difflib import SequenceMatcher
import torch
import open_clip
from PIL import Image
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue


class AltTextGenerator:

    def __init__(
        self,
        is_bulk=False,
        bulk_data_path=None,
        gemini_credentials_file="",
        gemini_model="",
        piction_base_url="https://piction.clevelandart.org/cma/",
        piction_update_endpoint="",
        piction_query_endpoint="!soap.jsonget?n=image_query&surl=1710855618ZZBQFTLPLCCT&search=AGE:",
        piction_days_since_query="1",
        max_retries=3,
        min_cosine=0.85,
        min_inner_product=0.025,
        prompt_file="",
        rag_directory="",
        output_file=None,
        with_rag=False,
        store_metrics=False,
        max_workers=8,
    ):
        # Update appropriate variables
        self.BULK_UPDATE = is_bulk
        self.BULK_DATA_PATH = bulk_data_path
        self.PICTION_BASE_URL = piction_base_url
        self.PICTION_QUERY_DAYS_SINCE = piction_days_since_query
        self.PICTION_QUERY_ENDPOINT = f"{piction_base_url}{piction_query_endpoint}{piction_days_since_query}"
        self.PICTION_UPDATE_ENDPOINT = f"{piction_base_url}{piction_update_endpoint}"
        self.MAX_NUMBER_OF_RETRIES = max_retries
        self.MIN_COSINE_SIMILARITY = min_cosine
        self.MIN_INNER_PRODUCT_ARRAY = min_inner_product
        self.ALT_TEXT_PROMPT_FILE = prompt_file
        self.RAG_EXAMPLES = rag_directory
        self.WITH_RAG = with_rag
        self.STORE_METRICS = store_metrics
        self.MAX_WORKERS = max_workers

        self.gemini_credentials = service_account.Credentials.from_service_account_file(
            gemini_credentials_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self.GEMINI_MODEL = gemini_model
        self.GEMINI_LOCATION = (
            "global" if gemini_model == "gemini-3-pro-preview" else "us-central1"
        )
        self.gemini_client = genai.Client(
            vertexai=True,
            project="ai-alt-text-481516",
            location=self.GEMINI_LOCATION,
            credentials=self.gemini_credentials,
        )
        self.CO_API_ENDPOINT = (
            "https://oai-api-02.clevelandart.org/api/collection/artworks"
        )

        # Output file for incremental saves
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = output_file or f"alt_text_results_{timestamp}.csv"
        self.rag_output_file = (
            f"alt_text_results_rag_{timestamp}.csv" if with_rag else None
        )

        self.csv_queue = Queue()
        self.stop_writer = threading.Event()

        self.csv_initialized = False
        self.rag_csv_initialized = False

        self.clip_executor = ThreadPoolExecutor(max_workers=2)

        self.csv_writer_thread = threading.Thread(target=self._csv_writer_loop, daemon=True)
        self.csv_writer_thread.start()

        # Set up logger
        self.logger = logging.getLogger(__name__)

        # Load prompt from file
        self.base_prompt = self._load_prompt()
        self.output_format = "# Output format\nA single string of text using proper spelling, grammar, and punctuation."

        # Generator for RAG examples instead of loading all at once
        self.rag_directory = (
            self.RAG_EXAMPLES
            if self.RAG_EXAMPLES and os.path.exists(self.RAG_EXAMPLES)
            else None
        )
        if self.rag_directory:
            self.logger.info(f"RAG directory configured: {self.RAG_EXAMPLES}")
            self._cache_rag_examples()
        else:
            self.logger.warning("No RAG directory specified or directory not found")
            self.rag_cache = []

        self.cuda_disabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.clip_model, _, self.clip_preprocess = (
            open_clip.create_model_and_transforms(
                model_name="ViT-B-32", pretrained="openai"
            )
        )

        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_model.eval()
        self.clip_model.to(self.device)

    def _ensure_clip_device(self, device):
        current_device = next(self.clip_model.parameters()).device.type
        if current_device != device:
            self.logger.warning(
                f"Moving CLIP model from {current_device} → {device}"
            )
            self.clip_model.to(device)

    def _run_clip_safe(self, fn, *args, **kwargs):
        """
        Run CLIP inference safely.
        If CUDA fails once, permanently fall back to CPU.
        """
        if self.cuda_disabled:
            return fn(*args, device="cpu", **kwargs)

        try:
            return fn(*args, device=self.device, **kwargs)

        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                self.logger.critical(
                    "CUDA failure detected — disabling CUDA for remainder of run"
                )

                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                self.cuda_disabled = True
                self.device = "cpu"
                self.clip_model.to("cpu")
                torch.cuda.empty_cache()
                return fn(*args, device="cpu", **kwargs)

            raise

    def _cache_rag_examples(self):
        """Cache RAG examples for thread-safe access"""
        self.rag_cache = []
        if self.rag_directory:
            for file in Path(self.rag_directory).glob("*.txt"):
                with open(file, "r") as f:
                    self.rag_cache.append(f.read())
            self.logger.info(f"Cached {len(self.rag_cache)} RAG examples")

    def _backoff_sleep(self, attempt, base=2.0, cap=30.0):
        sleep = min(cap, base**attempt) + random.uniform(0.25, 1.0)
        self.logger.warning(f"Backing off for {sleep:.2f}s before retry")
        time.sleep(sleep)

    def _shutdown_concurrency(self):
        self.logger.info("Shutting down concurrency infrastructure")

        try:
            self.csv_queue.join()
        except Exception:
            pass

        self.stop_writer.set()
        try:
            self.csv_writer_thread.join(timeout=5)
        except Exception:
            pass

        try:
            self.clip_executor.shutdown(wait=True)
        except Exception:
            pass

    def _load_prompt(self):
        """Load the alt text generation prompt from file"""
        if self.ALT_TEXT_PROMPT_FILE and os.path.exists(self.ALT_TEXT_PROMPT_FILE):
            with open(self.ALT_TEXT_PROMPT_FILE, "r") as f:
                self.logger.info(f"Loaded prompt from {self.ALT_TEXT_PROMPT_FILE}")
                return f.read()
        self.logger.warning(
            "No prompt file specified or file not found, using default prompt"
        )
        return "Generate a descriptive alt text for this image."

    def _iterate_rag_examples(self):
        """Generator to iterate through cached RAG examples"""
        for example in self.rag_cache:
            yield example

    def _truncate_text_for_clip(self, text, max_words=50):
        """
        Truncate text to fit within CLIP's 77 token limit.
        Uses conservative word limit (~50 words ≈ 65-70 tokens with margin for special tokens)
        """
        import re

        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Split into words and truncate
        words = text.split()
        if len(words) > max_words:
            truncated = " ".join(words[:max_words])
            self.logger.debug(
                f"Truncated text from {len(words)} to {max_words} words for CLIP processing"
            )
            return truncated
        return text

    def _normalize_embedding(self, emb):
        emb = np.asarray(emb)
        return emb.reshape(-1)

    def _load_image(self, image_src):
        if image_src.startswith("http://") or image_src.startswith("https://"):
            response = requests.get(image_src, timeout=15)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:
            return Image.open(image_src).convert("RGB")

    def _load_piction_image(self, image_location):
        piction_url = f"{self.PICTION_BASE_URL}{image_location}"
        return self._load_image(piction_url)

    # Generate image embeddings tool using CLIP
    def generate_image_embeddings(self, image_src, device):
        self._ensure_clip_device(device)

        image = self._load_image(image_src)
        image_input = self.clip_preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()

    # Generate text embeddings tool using CLIP
    def generate_text_embeddings(self, text, device):
        self._ensure_clip_device(device)

        text_tokens = self.clip_tokenizer([text]).to(device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()

    # Compare image to text embeddings tool
    def _compare_embeddings_sync(self, image_src, generated_caption):
        img_embedding = self._run_clip_safe(self.generate_image_embeddings, image_src)
        caption_embedding = self._run_clip_safe(self.generate_text_embeddings, generated_caption)

        cosine_similarity_score = cosine_similarity(
            caption_embedding.reshape(1, -1), img_embedding.reshape(1, -1)
        )[0][0]

        # Calculate inner product for compatibility
        inner_product_score = np.inner(caption_embedding, img_embedding)

        self.logger.debug(
            f"CLIP cosine similarity: {cosine_similarity_score:.4f}, Inner product: {inner_product_score:.4f}"
        )

        metrics = {
            "inner_product": float(inner_product_score),
            "cosine_similarity": float(cosine_similarity_score),
        }

        if cosine_similarity_score >= self.MIN_COSINE_SIMILARITY:
            self.logger.info("Embeddings meet CLIP threshold")
            return True, metrics

        self.logger.warning("Embeddings do not meet CLIP threshold")
        return False, metrics

    # Get co-api artwork context
    def get_artwork_context(self, accession_number):
        artwork_url = f"{self.CO_API_ENDPOINT}/{accession_number}?fields=technique,type"
        try:
            response = requests.get(artwork_url).json()
            data = response.get("data")
            technique = data.get("technique")
            type_val = data.get("type")
            context = f"# Context\nTechnique: {technique}\nType: {type_val}"
            return context
        except Exception as e:
            self.logger.warning(f"Failed to get artwork context: {e}")
            return None

    # Find top 3 RAG examples by embedding similarity (memory efficient)
    def find_similar_rag_captions(self, generated_caption):
        if not self.rag_cache:
            return []

        try:
            cap_emb = self._normalize_embedding(
                self.generate_text_embeddings(generated_caption)
            )
            # Use a min-heap to keep only top 3, avoiding storing all similarities
            from heapq import nlargest

            sims = []
            for idx, ex in enumerate(self._iterate_rag_examples()):
                ex_emb = self._normalize_embedding(self.generate_text_embeddings(ex))
                score = cosine_similarity(
                    cap_emb.reshape(1, -1), ex_emb.reshape(1, -1)
                )[0][0]
                sims.append((score, idx, ex))

            # Get top 3
            top_n = nlargest(3, sims, key=lambda x: x[0])
            results = []
            for rank, (score, _, ex) in enumerate(top_n):
                results.append(f"Example {rank+1}:\n{ex}")
            return results
        except Exception as e:
            self.logger.warning(
                f"Embedding similarity failed, using string matching: {e}"
            )
            # Fallback to simple string similarity
            similar_rag_examples = []
            matcher = SequenceMatcher(None)
            matcher.set_seq2(generated_caption)
            for idx, ex in enumerate(self._iterate_rag_examples()):
                matcher.set_seq1(ex)
                ratio = round(matcher.ratio(), 2)
                similar_rag_examples.append((ratio, idx, ex))

            similar_rag_examples.sort(key=lambda x: x[0], reverse=True)
            top_fallback = similar_rag_examples[:3]
            return [
                f"Example {rank+1}:\n{ex}"
                for rank, (_, _, ex) in enumerate(top_fallback)
            ]

    # Generate alt text with RAG
    def generate_alt_text_with_rag(self, image_src, image_caption, image_prompt):
        """Send generated caption with RAG data to gemini endpoint"""
        self.logger.info("Generating alt text with RAG examples")

        # Construct prompt with RAG examples
        rag_context = "\n\n".join(self.find_similar_rag_captions(image_caption))
        prompt = f"The following alt-text correctly and accurately reflects the content of an image:\n{image_caption}\n\nChange the style and composition to more closely reflect that of the following examples:\n{rag_context}"

        generated_caption = image_caption  # Fallback value
        final_metrics = {}

        for attempt in range(1, self.MAX_NUMBER_OF_RETRIES + 1):
            self.logger.info(f"Attempt {attempt}/{self.MAX_NUMBER_OF_RETRIES}")
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.GEMINI_MODEL,
                    contents=[
                        types.Part.from_uri(file_uri=image_src, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                            ),
                        ]
                    ),
                )
                if not getattr(response, "text", None):
                    raise ValueError("Gemini returned no response text")

                generated_caption = response.text.strip()
                if not generated_caption:
                    raise ValueError("Gemini returned empty caption")

                self.logger.debug(f"Generated caption: {generated_caption[:100]}...")

                # Compare embeddings
                future = self.clip_executor.submit(
                    self._compare_embeddings_sync, image_src, generated_caption
                )
                meets_threshold, metrics = future.result()
                final_metrics = metrics

                if meets_threshold:
                    # Success - save with ALT_TEXT_MEETS_THRESHOLD = YES
                    self.logger.info(
                        f"Successfully generated alt text on attempt {attempt}"
                    )
                    result = {
                        "caption": generated_caption,
                        "ALT_TEXT_MEETS_THRESHOLD": "YES",
                        "attempts": attempt,
                    }
                    if self.STORE_METRICS:
                        result.update(final_metrics)
                    if self.device == "cuda" and not self.cuda_disabled:
                        torch.cuda.empty_cache()
                    return result
                else:
                    # Retry with regenerate prompt
                    prompt = f"{prompt}\nRevise the alt-text to more closely align with the following guidelines:\n{image_prompt}"
                    self.logger.warning(
                        f"Embeddings failed threshold, retrying with enhanced prompt"
                    )
            except Exception as e:
                self.logger.error(f"Attempt {attempt} failed: {e}")

            if attempt < self.MAX_NUMBER_OF_RETRIES:
                self._backoff_sleep(attempt)
            else:
                break

        # Maximum retries reached - save with ALT_TEXT_MEETS_THRESHOLD = NO
        self.logger.error(
            f"Failed to generate acceptable alt text after {self.MAX_NUMBER_OF_RETRIES} attempts"
        )
        result = {
            "caption": "",
            "ALT_TEXT_MEETS_THRESHOLD": "NO",
            "attempts": self.MAX_NUMBER_OF_RETRIES,
        }
        if self.STORE_METRICS:
            result.update({"inner_product": 0.0, "cosine_similarity": 0.0})
        return result

    # Generate alt text
    def generate_alt_text(self, image_src, accession_number):
        """Send image to gemini with prompt"""
        self.logger.info("Generating alt text")
        for attempt in range(1, self.MAX_NUMBER_OF_RETRIES + 1):
            self.logger.info(f"Attempt {attempt}/{self.MAX_NUMBER_OF_RETRIES}")
            try:
                image_context = self.get_artwork_context(accession_number)
                prompt = self.base_prompt
                if image_context is not None:
                    prompt = f"{prompt}\n{image_context}\n{self.output_format}"
                else:
                    prompt = f"{prompt}\n{self.output_format}"
                response = self.gemini_client.models.generate_content(
                    model=self.GEMINI_MODEL,
                    contents=[
                        types.Part.from_uri(file_uri=image_src, mime_type="image/jpeg"),
                        prompt,
                    ],
                    config=types.GenerateContentConfig(
                        safety_settings=[
                            types.SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="OFF",
                            ),
                            types.SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                            ),
                        ]
                    ),
                )
                if not getattr(response, "text", None):
                    raise ValueError("Gemini returned no response text")
                generated_caption = response.text.strip()
                if not generated_caption:
                    raise ValueError("Gemini returned empty caption")

                self.logger.debug(f"Generated caption: {generated_caption[:100]}...")

                # Compare embeddings
                future = self.clip_executor.submit(
                    self._compare_embeddings_sync, image_src, generated_caption
                )
                meets_threshold, metrics = future.result()

                if meets_threshold:
                    # Success - save with ALT_TEXT_MEETS_THRESHOLD = true
                    self.logger.info("Successfully generated alt text on first attempt")
                    result = {
                        "caption": generated_caption,
                        "ALT_TEXT_MEETS_THRESHOLD": "YES",
                        "attempts": 1,
                    }
                    if self.STORE_METRICS:
                        result.update(metrics)
                    if self.device == "cuda" and not self.cuda_disabled:
                        torch.cuda.empty_cache()
                    return result
                else:
                    # Use regenerate alt reprompt with RAG method
                    self.logger.warning(
                        "Initial generation failed threshold, switching to RAG method"
                    )
                    rag_result = self.generate_alt_text_with_rag(
                        image_src, generated_caption, prompt
                    )
                    # If STORE_METRICS and metrics not in rag_result, add initial metrics
                    if self.STORE_METRICS and "cosine_similarity" not in rag_result:
                        rag_result.update(metrics)
                    if self.device == "cuda" and not self.cuda_disabled:
                        torch.cuda.empty_cache()
                    return rag_result
            except Exception as e:
                self.logger.error(f"Error generating alt text: {e}")
                if attempt < self.MAX_NUMBER_OF_RETRIES:
                    self._backoff_sleep(attempt)
                else:
                    break

        result = {"caption": "", "ALT_TEXT_MEETS_THRESHOLD": "NO", "attempts": 1}
        if self.STORE_METRICS:
            result.update({"inner_product": 0.0, "cosine_similarity": 0.0})
        return result

    # Generate alt text with forced RAG (for --with-rag option)
    def generate_alt_text_with_forced_rag(self, image_src, accession_number):
        """Generate alt text using RAG regardless of initial threshold"""
        self.logger.info("Generating alt text with forced RAG method")

        # First generate baseline caption
        try:
            image_context = self.get_artwork_context(accession_number)
            prompt = self.base_prompt
            if image_context is not None:
                prompt = f"{prompt}\n{image_context}\n{self.output_format}"
            else:
                prompt = f"{prompt}\n{self.output_format}"

            response = self.gemini_client.models.generate_content(
                model=self.GEMINI_MODEL,
                contents=[
                    types.Part.from_uri(file_uri=image_src, mime_type="image/jpeg"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
                        ),
                    ]
                ),
            )

            if response.text:
                generated_caption = response.text
                self.logger.debug(
                    f"Generated initial caption for RAG: {generated_caption[:100]}..."
                )

                # Now apply RAG method regardless of threshold
                return self.generate_alt_text_with_rag(
                    image_src, generated_caption, prompt
                )
            else:
                self.logger.error(f"API error: No response text")
                result = {
                    "caption": "",
                    "ALT_TEXT_MEETS_THRESHOLD": "NO",
                    "attempts": 1,
                }
                if self.STORE_METRICS:
                    result.update({"inner_product": 0.0, "cosine_similarity": 0.0})
                return result
        except Exception as e:
            self.logger.error(f"Error generating alt text with forced RAG: {e}")
            result = {"caption": "", "ALT_TEXT_MEETS_THRESHOLD": "NO", "attempts": 1}
            if self.STORE_METRICS:
                result.update({"inner_product": 0.0, "cosine_similarity": 0.0})
            return result

    def _get_csv_fieldnames(self):
        """Get fieldnames for CSV based on configuration"""
        fieldnames = ["image_id", "caption", "ALT_TEXT_MEETS_THRESHOLD", "attempts"]
        if self.STORE_METRICS:
            fieldnames.extend(["inner_product", "cosine_similarity"])
        return fieldnames

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_initialized:
            with open(self.output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_csv_fieldnames())
                writer.writeheader()
            self.csv_initialized = True
            self.logger.info(f"Initialized output file: {self.output_file}")

    def _initialize_rag_csv(self):
        """Initialize RAG CSV file with headers"""
        if not self.rag_csv_initialized and self.WITH_RAG:
            with open(self.rag_output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_csv_fieldnames())
                writer.writeheader()
            self.rag_csv_initialized = True
            self.logger.info(f"Initialized RAG output file: {self.rag_output_file}")

    def _append_to_csv(self, result, is_rag=False):
        """Append a single result to CSV file immediately"""
        output_file = self.rag_output_file if is_rag else self.output_file
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._get_csv_fieldnames())
            writer.writerow(result)
        self.logger.debug(
            f"Saved result for image {result.get('image_id')} to {output_file}"
        )

    def _csv_writer_loop(self):
        while not self.stop_writer.is_set() or not self.csv_queue.empty():
            try:
                payload = self.csv_queue.get(timeout=0.5)
            except Exception:
                continue

            if payload is None:
                self.csv_queue.task_done()
                continue

            result, is_rag = payload

            if is_rag:
                if not self.rag_csv_initialized:
                    self._initialize_rag_csv()
                self._append_to_csv(result, is_rag=True)
            else:
                if not self.csv_initialized:
                    self._initialize_csv()
                self._append_to_csv(result, is_rag=False)

            self.csv_queue.task_done()

    # Save data tool - modified for incremental saving
    def save_data(self, result, image_id=None, is_rag=False):
        """Save individual result immediately"""
        if self.BULK_UPDATE:
            self.csv_queue.put((result, is_rag))
            return
        else:
            # Save to JSON file then post to piction update endpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_suffix = "_rag" if is_rag else ""
            json_file = f"alt_text_update_{image_id}{file_suffix}_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            self.logger.info(f"Data saved to {json_file}")

            # Post to piction update endpoint (only for non-RAG or if no standard version)
            if not is_rag:
                try:
                    response = requests.post(self.PICTION_UPDATE_ENDPOINT, json=result)
                    if response.status_code == 200:
                        self.logger.info(f"Successfully updated image {image_id}")
                    else:
                        self.logger.error(
                            f"Failed to update image {image_id}: {response.status_code}"
                        )
                except Exception as e:
                    self.logger.error(f"Error posting to endpoint: {e}")

    def _stream_bulk_data(self):
        """Generator to stream bulk data without loading entire file into memory"""
        with open(self.BULK_DATA_PATH, "r", encoding="utf-8") as f:
            if self.BULK_DATA_PATH.endswith(".csv"):
                reader = csv.DictReader(f)
                for row in reader:
                    yield row
            else:
                # For JSON, we need to load it, but we can process items one at a time
                data = json.load(f)
                for item in data:
                    yield item

    def _process_bulk_item(self, item, idx, total_count):
        image_src = item.get("image_src")
        image_id = item.get("UMO_ID")
        accession_number = item.get("accession_number")

        self.logger.info(
            f"[Worker] Processing image {idx}/{total_count}: {image_id}"
        )

        # Standard generation
        result = self.generate_alt_text(image_src, accession_number)
        result["image_id"] = image_id
        self.save_data(result, is_rag=False)

        # Optional RAG
        if self.WITH_RAG:
            rag_result = self.generate_alt_text_with_forced_rag(
                image_src, accession_number
            )
            rag_result["image_id"] = image_id
            self.save_data(rag_result, is_rag=True)

        return image_id

    def _safe_process_bulk_item(self, item, idx, total_count):
        try:
            return self._process_bulk_item(item, idx, total_count)
        except Exception as e:
            image_id = item.get("UMO_ID") or item.get("image_id")
            self.logger.error(
                f"[FATAL ITEM ERROR] image_id={image_id} idx={idx}: {e}",
                exc_info=True,
            )

            # Emit a failure row so the CSV stays complete
            failure = {
                "image_id": image_id,
                "caption": "",
                "ALT_TEXT_MEETS_THRESHOLD": "NO",
                "attempts": 0,
            }

            if self.STORE_METRICS:
                failure.update(
                    {"inner_product": 0.0, "cosine_similarity": 0.0}
                )

            self.csv_queue.put((failure, False))
            return image_id

    def _process_piction_updated_images(self, results):
        processed_results = []
        for item in enumerate(results):
            try:
                image_id = item.get('id')
                web_image_data = item.get('wq')
                web_image_info = web_image_data.get('1')
                web_image_src = web_image_info.get('u')
                web_image_url = f"{self.PICTION_BASE_URL}{web_image_src}"
                accession_number = web_image_info.get('f').split('.jpg')[0]
                processed_results.append({
                    'image_id': image_id,
                    'image_src': web_image_url,
                    'accession_number': accession_number
                })
            except Exception:
                self.logger.warning(f"Failed to process {item}")
                continue
        return processed_results

    def _query_piction_updated_images(self):
        data = requests.get(self.PICTION_QUERY_ENDPOINT)
        res_data = data.json()
        results = []
        if res_data.get('r'):
            results = res_data.get('r')
        return results

    # Generation tool - modified for streaming
    def run_generation(self):
        """Stream processing with immediate saves"""
        try:
            if self.BULK_UPDATE:
                self.logger.info(f"Starting bulk generation from {self.BULK_DATA_PATH}")

                # Initialize CSV for bulk output
                self._initialize_csv()
                if self.WITH_RAG:
                    self._initialize_rag_csv()
                    self.logger.info(
                        "--with-rag enabled: generating both standard and RAG versions"
                    )

                # Count total items for progress tracking (optional, requires one pass)
                total_count = sum(1 for _ in self._stream_bulk_data())
                self.logger.info(f"Processing {total_count} images")

                # Process and save each item immediately
                items = list(self._stream_bulk_data())
                total_count = len(items)
                max_workers = self.MAX_WORKERS
                self.logger.info(f"Processing {total_count} images using {max_workers} workers")

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(self._safe_process_bulk_item, item, idx, total_count): item
                        for idx, item in enumerate(items, 1)
                    }
                    for future in as_completed(futures):
                        try:
                            image_id = future.result()
                            self.logger.debug(f"[Worker completed] image_id={image_id}")
                        except Exception as e:
                            # This should NEVER fire now, but stays as a last-resort guard
                            self.logger.critical(
                                f"[UNEXPECTED FUTURE FAILURE]: {e}",
                                exc_info=True,
                            )
                self.logger.info(
                    f"Bulk generation completed. Results saved to {self.output_file}"
                )
                if self.WITH_RAG:
                    self.logger.info(f"RAG results saved to {self.rag_output_file}")
            else:
                self.logger.info(
                    f"Starting generation from Piction Query API"
                )
                # Loop through results from piction_query_endpoint
                try:
                    self.logger.info(f"Looking for piction uploads from last {self.PICTION_QUERY_DAYS_SINCE} days")
                    unprocessed_updates = self._query_piction_updated_images()
                    if len(unprocessed_updates) is 0:
                        self.logger.info(f"No recent piction uploads in {self.PICTION_QUERY_DAYS_SINCE} days")
                        return
                    processed_updates = self._process_piction_updated_images(unprocessed_updates)
                    for idx, item in enumerate(processed_updates):
                        image_id = item.get('image_id')
                        self.logger.info(f"Processing image {idx}/{len(processed_updates)}: {image_id}")
                        accession_number = item.get('accession_number')
                        image_src = item.get('image_src')
                        # If WITH_RAG, also generate RAG version
                        if self.WITH_RAG:
                            self.logger.info(
                                f"Generating RAG version for image {image_id}"
                            )
                            rag_result = self.generate_alt_text_with_forced_rag(
                                image_src, accession_number
                            )
                            rag_result["image_id"] = image_id
                            self.save_data(rag_result, image_id, is_rag=True)
                        else:
                            self.logger.info(
                                f"Generating alt text for image {image_id}"
                            )
                            result = self.generate_alt_text(image_src, accession_number)
                            result["image_id"] = image_id
                            self.save_data(result, image_id, is_rag=False)
                    self.logger.info("Generation of alt text using piction query / update endpoints complete")
                except Exception as e:
                    self.logger.error(f"Error updating individual images: {e}")

        except Exception:
            ### GLOBAL SAFETY NET
            self.logger.critical(
                "[RUN_GENERATION CRASH PREVENTED]",
                exc_info=True,
            )
        finally:
            self._shutdown_concurrency()

# Implement main function with arg parser including a help method
def main():
    parser = argparse.ArgumentParser(
        description="Generate alt text for images using AI with embedding validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process bulk data from CSV
  python generate_alt_text_with_agents.py --bulk --bulk-data-path data.csv --gemini-endpoint http://api.gemini.com/generate

  # Process from Piction API
  python generate_alt_text_with_agents.py --piction-query http://api.piction.com/query --piction-update http://api.piction.com/update
        """,
    )

    parser.add_argument(
        "--bulk", action="store_true", help="Enable bulk processing mode"
    )
    parser.add_argument(
        "--bulk-data-path",
        type=str,
        default=None,
        help="Path to bulk data file (CSV or JSON)",
    )
    parser.add_argument(
        "--gemini-credentials-file",
        type=str,
        required=True,
        help="Gemini API JSON credentials",
    )
    parser.add_argument(
        "--gemini-model", type=str, required=True, help="Gemini model to use"
    )
    parser.add_argument(
        "--piction-update", type=str, default="", help="Piction update endpoint"
    )
    parser.add_argument(
        "--piction-base-url",
        type=str,
        default="https://piction.clevelandart.org/cma/",
        help="Base URL for piction queries, update, and image endpoints"
    )
    parser.add_argument(
        "--piction-query",
        type=str,
        default="!soap.jsonget?n=image_query&surl=1710855618ZZBQFTLPLCCT&search=AGE:",
        help="Piction query endpoint",
    )
    parser.add_argument(
        "--piction-days-since-query",
        type=str,
        default="1",
        help="Piction days since query parameter"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum number of retry attempts (default: 5)",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.25,
        help="Minimum cosine similarity threshold (default: 0.25)",
    )
    parser.add_argument(
        "--min-inner-product",
        type=float,
        default=0.1,
        help="Minimum inner product threshold (default: 0.1)",
    )
    parser.add_argument(
        "--prompt-file", type=str, default="", help="Path to prompt template file"
    )
    parser.add_argument(
        "--rag-directory",
        type=str,
        default="",
        help="Directory containing RAG example files",
    )
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="Generate two outputs: one with standard prompt and one with RAG method applied regardless of threshold",
    )
    parser.add_argument(
        "--store-metrics",
        action="store_true",
        help="Store cosine similarity and inner product metrics with results",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output CSV file path (default: auto-generated with timestamp)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs to console only)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Concurrent workers"
    )

    args = parser.parse_args()

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler()]
    if args.log_file:
        log_filename = f"{args.log_file}_{timestamp}.log"
        handlers.append(logging.FileHandler(log_filename))

    logging.basicConfig(
        level=getattr(logging, args.log_level), format=log_format, handlers=handlers
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Alt Text Generator (Memory-Optimized Version)")
    logger.info(f"Log level: {args.log_level}")

    # Validate arguments
    if args.bulk and not args.bulk_data_path:
        parser.error("--bulk-data-path is required when --bulk is enabled")

    if not args.bulk and not args.piction_query:
        parser.error("--piction-query is required when not in bulk mode")

    if args.with_rag and not args.rag_directory:
        parser.error("--rag-directory is required when --with-rag is enabled")

    # Initialize generator
    logger.info("Initializing AltTextGenerator")
    generator = AltTextGenerator(
        is_bulk=args.bulk,
        bulk_data_path=args.bulk_data_path,
        gemini_credentials_file=args.gemini_credentials_file,
        gemini_model=args.gemini_model,
        piction_update_endpoint=args.piction_update,
        piction_query_endpoint=args.piction_query,
        max_retries=args.max_retries,
        min_cosine=args.min_cosine,
        min_inner_product=args.min_inner_product,
        prompt_file=args.prompt_file,
        rag_directory=args.rag_directory,
        output_file=args.output_file,
        with_rag=args.with_rag,
        store_metrics=args.store_metrics,
        max_workers=args.max_workers
    )

    # Run generation
    logger.info("Starting alt text generation...")
    generator.run_generation()
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()
