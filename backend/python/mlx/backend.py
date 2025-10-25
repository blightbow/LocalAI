#!/usr/bin/env python3
import asyncio
from concurrent import futures
import argparse
import signal
import sys
import os
from typing import List, Optional
import time

import backend_pb2
import backend_pb2_grpc

import grpc
from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
import mlx.core as mx
import base64
import io

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# If MAX_WORKERS are specified in the environment use it, otherwise default to 1
MAX_WORKERS = int(os.environ.get('PYTHON_GRPC_MAX_WORKERS', '1'))

def is_float(s):
    """Check if a string can be converted to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False
def is_int(s):
    """Check if a string can be converted to int."""
    try:
        int(s)
        return True
    except ValueError:
        return False

# Implement the BackendServicer class with the service methods
class BackendServicer(backend_pb2_grpc.BackendServicer):
    """
    A gRPC servicer that implements the Backend service defined in backend.proto.

    This servicer manages model state, tokenizer, and a global KV cache for efficient
    inference. The cache is shared across requests and uses prefix-based trimming to
    maintain correctness when prompts change.
    """

    def __init__(self):
        """Initialize the servicer with cache management state."""
        super().__init__()
        # Cache synchronization
        self._cache_lock = asyncio.Lock()

        # Track last prompt for prefix-based trimming
        self._last_prompt_tokens: Optional[List[int]] = None
        self._last_prompt_length: int = 0  # Track prompt length separately from cache size
        self._last_prompt_string: Optional[str] = None  # Cache prompt string for fast comparison

        # Detect append-only patterns for optimization
        self._consecutive_appends = 0

        # Cache performance metrics
        self._cache_hits = 0
        self._cache_trims = 0
        self._total_requests = 0
        self._tokenization_cache_hits = 0  # Track tokenization cache efficiency

    def _tokenize_with_cache(self, prompt_string: str) -> List[int]:
        """
        Tokenize a prompt with intelligent caching to avoid redundant tokenization.

        Optimization strategy:
        1. Identical prompt (retry/swipe): Reuse cached tokens (0 tokenization)
        2. Append-only prompt (chat): Tokenize only new part (partial tokenization)
        3. Different prompt: Full tokenization (unavoidable)

        Args:
            prompt_string: The prompt text to tokenize.

        Returns:
            List[int]: Token IDs.
        """
        # Optimization 1: Check for identical prompt (retry/swipe case)
        if prompt_string == self._last_prompt_string and self._last_prompt_tokens is not None:
            self._tokenization_cache_hits += 1
            print(
                f"cache: HIT - Reusing cached tokens ({len(self._last_prompt_tokens)} tokens)",
                file=sys.stderr
            )
            return self._last_prompt_tokens

        # Optimization 2: Check for append-only pattern (chat case)
        if (self._last_prompt_string is not None
            and self._last_prompt_tokens is not None
            and prompt_string.startswith(self._last_prompt_string)
            and len(prompt_string) > len(self._last_prompt_string)):

            # Tokenize only the NEW part
            new_part = prompt_string[len(self._last_prompt_string):]
            new_tokens = self.tokenizer.encode(new_part, add_special_tokens=False)

            # Combine with cached tokens
            full_tokens = self._last_prompt_tokens + new_tokens

            self._tokenization_cache_hits += 1
            print(
                f"cache: PARTIAL HIT - Tokenized {len(new_tokens)} new tokens "
                f"(reused {len(self._last_prompt_tokens)} cached tokens)",
                file=sys.stderr
            )
            return full_tokens

        # Optimization 3: Full tokenization (different prompt)
        tokens = self.tokenizer.encode(prompt_string, add_special_tokens=True)
        print(
            f"cache: MISS - Full tokenization ({len(tokens)} tokens)",
            file=sys.stderr
        )
        return tokens

    def _trim_cache_to_prefix(self, new_tokens: List[int]) -> int:
        """
        Trim the prompt cache to the longest common prefix with the new prompt.

        This implements prefix-based cache management.

        CRITICAL: The cache contains BOTH prompt tokens AND generated tokens.
        We must:
        1. First trim any previous generation (backtrack to prompt only)
        2. Then compare prompts and trim to common prefix
        3. Track the new prompt length for next request

        Args:
            new_tokens: Token IDs of the new prompt.

        Returns:
            int: Total number of tokens trimmed from the cache.
        """
        total_trimmed = 0

        # Step 1: Trim any previous generation tokens
        # The cache contains: [old_prompt_tokens | old_generation_tokens]
        # We need to backtrack to: [old_prompt_tokens] before comparing with new prompt
        if self._last_prompt_length > 0 and hasattr(self, 'prompt_cache') and self.prompt_cache:
            # Get current cache size (includes prompt + generation from previous request)
            cache_size = self.prompt_cache[0].offset if self.prompt_cache else 0
            generation_length = cache_size - self._last_prompt_length

            if generation_length > 0:
                # Trim the previous generation
                actual = trim_prompt_cache(self.prompt_cache, generation_length)
                total_trimmed += actual
                print(
                    f"cache: Backtracked {actual} tokens "
                    f"(removing previous generation, cache: {cache_size} → {cache_size - actual})",
                    file=sys.stderr
                )

        # Step 2: Handle first request
        if self._last_prompt_tokens is None:
            self._last_prompt_tokens = new_tokens
            self._last_prompt_length = len(new_tokens)
            self._consecutive_appends = 0
            print(
                f"cache: First request - initializing with {len(new_tokens)} tokens",
                file=sys.stderr
            )
            return total_trimmed

        # Step 3: Compare new prompt with old prompt
        old_tokens = self._last_prompt_tokens
        old_len = len(old_tokens)
        new_len = len(new_tokens)

        # Check for append-only pattern (optimization)
        if new_len > old_len and new_tokens[:old_len] == old_tokens:
            # Append-only: new prompt extends the old one
            # Example: "hello" -> "hello world"
            self._consecutive_appends += 1
            self._cache_hits += 1
            appended_tokens = new_len - old_len
            print(
                f"cache: Append-only pattern detected "
                f"(consecutive: {self._consecutive_appends}). "
                f"Cache fully reused, appending {appended_tokens} new tokens.",
                file=sys.stderr
            )
            self._last_prompt_tokens = new_tokens
            self._last_prompt_length = new_len
            return total_trimmed

        # Find longest common prefix
        common_len = 0
        min_len = min(old_len, new_len)
        for i in range(min_len):
            if old_tokens[i] == new_tokens[i]:
                common_len += 1
            else:
                break

        # Calculate tokens to trim from the prompt portion
        tokens_to_trim = old_len - common_len

        if tokens_to_trim > 0:
            # Reset append-only counter since we're diverging
            self._consecutive_appends = 0
            self._cache_trims += 1

            # Trim the cache to the common prefix
            actual_trimmed = trim_prompt_cache(self.prompt_cache, tokens_to_trim)
            total_trimmed += actual_trimmed

            print(
                f"cache: Trimmed {actual_trimmed} tokens "
                f"(common prefix: {common_len}/{old_len} tokens, diverged at token {common_len})",
                file=sys.stderr
            )
        elif old_len == new_len:
            # Prompts are identical
            self._cache_hits += 1
            print(
                f"cache: Identical prompt detected ({new_len} tokens), cache fully reused",
                file=sys.stderr
            )
        else:
            # New prompt is a prefix of old (rare edge case)
            # Example: "hello world" -> "hello"
            self._cache_trims += 1
            tokens_to_trim = old_len - new_len
            actual_trimmed = trim_prompt_cache(self.prompt_cache, tokens_to_trim)
            total_trimmed += actual_trimmed
            print(
                f"cache: New prompt is prefix of old, trimmed {actual_trimmed} tokens",
                file=sys.stderr
            )

        self._last_prompt_tokens = new_tokens
        self._last_prompt_length = new_len
        return total_trimmed

    def _log_cache_stats(self):
        """Log cache performance statistics."""
        if self._total_requests > 0:
            hit_rate = (self._cache_hits / self._total_requests) * 100
            tok_cache_rate = (self._tokenization_cache_hits / self._total_requests) * 100
            print(
                f"cache stats - Requests: {self._total_requests}, "
                f"KV cache hits: {self._cache_hits} ({hit_rate:.1f}%), "
                f"Tokenization cache hits: {self._tokenization_cache_hits} ({tok_cache_rate:.1f}%), "
                f"Trims: {self._cache_trims}, "
                f"Consecutive appends: {self._consecutive_appends}",
                file=sys.stderr
            )

    def Health(self, request, context):
        """
        Returns a health check message.

        Args:
            request: The health check request.
            context: The gRPC context.

        Returns:
            backend_pb2.Reply: The health check reply.
        """
        return backend_pb2.Reply(message=bytes("OK", 'utf-8'))

    async def LoadModel(self, request, context):
        """
        Loads a language model using MLX.

        Args:
            request: The load model request.
            context: The gRPC context.

        Returns:
            backend_pb2.Result: The load model result.
        """
        try:
            print(f"Loading MLX model: {request.Model}", file=sys.stderr)
            print(f"Request: {request}", file=sys.stderr)
            
            # Parse options like in the diffusers backend
            options = request.Options
            self.options = {}
            
            # The options are a list of strings in this form optname:optvalue
            # We store all the options in a dict for later use
            for opt in options:
                if ":" not in opt:
                    continue
                key, value = opt.split(":", 1)  # Split only on first colon to handle values with colons
                
                # Convert numeric values to appropriate types
                if is_float(value):
                    value = float(value)
                elif is_int(value):
                    value = int(value)
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                    
                self.options[key] = value
            
            print(f"Options: {self.options}", file=sys.stderr)
            
            # Build tokenizer config for MLX using options
            tokenizer_config = {}
            
            # Handle trust_remote_code from request or options
            if request.TrustRemoteCode or self.options.get("trust_remote_code", False):
                tokenizer_config["trust_remote_code"] = True
            
            # Handle EOS token from options
            if "eos_token" in self.options:
                tokenizer_config["eos_token"] = self.options["eos_token"]
            
            # Handle other tokenizer config options
            for key in ["pad_token", "bos_token", "unk_token", "sep_token", "cls_token", "mask_token"]:
                if key in self.options:
                    tokenizer_config[key] = self.options[key]
            
            # Load model and tokenizer using MLX
            if tokenizer_config:
                print(f"Loading with tokenizer_config: {tokenizer_config}", file=sys.stderr)
                self.model, self.tokenizer = load(request.Model, tokenizer_config=tokenizer_config)
            else:
                self.model, self.tokenizer = load(request.Model)
            
            # Initialize prompt cache for efficient generation
            max_kv_size = self.options.get("max_kv_size", None)
            self.prompt_cache = make_prompt_cache(self.model, max_kv_size)

            # Reset cache tracking state when new model is loaded
            self._last_prompt_tokens = None
            self._last_prompt_length = 0
            self._last_prompt_string = None
            self._consecutive_appends = 0
            self._cache_hits = 0
            self._cache_trims = 0
            self._total_requests = 0
            self._tokenization_cache_hits = 0

            print("cache: Initialized new prompt cache", file=sys.stderr)

        except Exception as err:
            print(f"Error loading MLX model {err=}, {type(err)=}", file=sys.stderr)
            return backend_pb2.Result(success=False, message=f"Error loading MLX model: {err}")

        print("MLX model loaded successfully", file=sys.stderr)
        return backend_pb2.Result(message="MLX model loaded successfully", success=True)

    async def Predict(self, request, context):
        """
        Generates text based on the given prompt and sampling parameters using MLX.

        This method uses a cache lock to ensure thread-safe access to the model,
        tokenizer, and prompt cache. The cache is trimmed to match the common
        prefix with the previous prompt before generation.

        Args:
            request: The predict request.
            context: The gRPC context.

        Returns:
            backend_pb2.Reply: The predict result.
        """
        try:
            # Acquire cache lock for thread-safe operation
            async with self._cache_lock:
                self._total_requests += 1

                # Prepare the prompt (apply chat template, etc.)
                prompt = self._prepare_prompt(request)

                # OPTIMIZATION: Use cached tokenization when possible
                # - Retry/swipe: Reuses tokens from identical prompt (0 tokenization)
                # - Append-only: Tokenizes only new part (partial tokenization)
                # - Different prompt: Full tokenization (unavoidable)
                prompt_tokens = self._tokenize_with_cache(prompt)

                # Trim cache to common prefix (with append-only optimization)
                self._trim_cache_to_prefix(prompt_tokens)

                # Update cached prompt string for next request
                self._last_prompt_string = prompt

                # Build generation parameters using request attributes and options
                max_tokens, sampler_params = self._build_generation_params(request)

                print(
                    f"Generating text with MLX - "
                    f"prompt_tokens: {len(prompt_tokens)}, "
                    f"max_tokens: {max_tokens}, "
                    f"sampler_params: {sampler_params}",
                    file=sys.stderr
                )

                # Create sampler with parameters
                sampler = make_sampler(**sampler_params)

                # OPTIMIZATION: Pass tokens directly to generate() to avoid re-tokenization
                # MLX generate() accepts: str | mx.array | List[int]
                # By passing List[int], we skip redundant tokenization inside generate()
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt_tokens,  # <-- Pass tokens directly!
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=self.prompt_cache,
                    verbose=False
                )

                # Log cache statistics periodically
                if self._total_requests % 10 == 0:
                    self._log_cache_stats()

                return backend_pb2.Reply(message=bytes(response, encoding='utf-8'))

        except Exception as e:
            print(f"Error in MLX Predict: {e}", file=sys.stderr)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Generation failed: {str(e)}")
            return backend_pb2.Reply(message=bytes("", encoding='utf-8'))

    def Embedding(self, request, context):
        """
        A gRPC method that calculates embeddings for a given sentence.
        
        Note: MLX-LM doesn't support embeddings directly. This method returns an error.

        Args:
            request: An EmbeddingRequest object that contains the request parameters.
            context: A grpc.ServicerContext object that provides information about the RPC.

        Returns:
            An EmbeddingResult object that contains the calculated embeddings.
        """
        print("Embeddings not supported in MLX backend", file=sys.stderr)
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Embeddings are not supported in the MLX backend.")
        return backend_pb2.EmbeddingResult()

    async def PredictStream(self, request, context):
        """
        Generates text based on the given prompt and sampling parameters, and streams the results using MLX.

        This method uses a cache lock to ensure thread-safe access to the model,
        tokenizer, and prompt cache. The cache is trimmed to match the common
        prefix with the previous prompt before generation.

        Args:
            request: The predict stream request.
            context: The gRPC context.

        Yields:
            backend_pb2.Reply: Streaming predict results.
        """
        try:
            # Acquire cache lock for thread-safe operation
            # Note: We hold the lock for the entire streaming duration to prevent
            # cache corruption from concurrent requests
            async with self._cache_lock:
                self._total_requests += 1

                # Prepare the prompt (apply chat template, etc.)
                prompt = self._prepare_prompt(request)

                # OPTIMIZATION: Use cached tokenization when possible
                prompt_tokens = self._tokenize_with_cache(prompt)

                # Trim cache to common prefix (with append-only optimization)
                self._trim_cache_to_prefix(prompt_tokens)

                # Update cached prompt string for next request
                self._last_prompt_string = prompt

                # Build generation parameters using request attributes and options
                max_tokens, sampler_params = self._build_generation_params(request, default_max_tokens=512)

                print(
                    f"Streaming text with MLX - "
                    f"prompt_tokens: {len(prompt_tokens)}, "
                    f"max_tokens: {max_tokens}, "
                    f"sampler_params: {sampler_params}",
                    file=sys.stderr
                )

                # Create sampler with parameters
                sampler = make_sampler(**sampler_params)

                # OPTIMIZATION: Pass tokens directly to stream_generate()
                # Stream text generation using MLX with trimmed cache
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt_tokens,  # <-- Pass tokens directly!
                    max_tokens=max_tokens,
                    sampler=sampler,
                    prompt_cache=self.prompt_cache,
                ):
                    yield backend_pb2.Reply(message=bytes(response.text, encoding='utf-8'))

                # Log cache statistics periodically
                if self._total_requests % 10 == 0:
                    self._log_cache_stats()

        except Exception as e:
            print(f"Error in MLX PredictStream: {e}", file=sys.stderr)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Streaming generation failed: {str(e)}")
            yield backend_pb2.Reply(message=bytes("", encoding='utf-8'))

    def _prepare_prompt(self, request):
        """
        Prepare the prompt for MLX generation, handling chat templates if needed.

        Args:
            request: The gRPC request containing prompt and message information.

        Returns:
            str: The prepared prompt.
        """
        # If tokenizer template is enabled and messages are provided instead of prompt, apply the tokenizer template
        if not request.Prompt and request.UseTokenizerTemplate and request.Messages:
            # Convert gRPC messages to the format expected by apply_chat_template
            messages = []
            for msg in request.Messages:
                messages.append({"role": msg.role, "content": msg.content})
            
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt
        else:
            return request.Prompt





    def _build_generation_params(self, request, default_max_tokens=200):
        """
        Build generation parameters from request attributes and options.

        Args:
            request: The gRPC request.
            default_max_tokens: Default max_tokens if not specified.

        Returns:
            tuple: (max_tokens, sampler_params dict)
        """
        # Extract max_tokens
        max_tokens = getattr(request, 'Tokens', default_max_tokens)
        if max_tokens == 0:
            max_tokens = default_max_tokens
        
        # Extract sampler parameters from request attributes
        temp = getattr(request, 'Temperature', 0.0)
        if temp == 0.0:
            temp = 0.6  # Default temperature
        
        top_p = getattr(request, 'TopP', 0.0)
        if top_p == 0.0:
            top_p = 1.0  # Default top_p
        
        # Initialize sampler parameters
        sampler_params = {
            'temp': temp,
            'top_p': top_p,
            'xtc_threshold': 0.0,
            'xtc_probability': 0.0,
        }
        
        # Add seed if specified
        seed = getattr(request, 'Seed', 0)
        if seed != 0:
            mx.random.seed(seed)
        
        # Override with options if available
        if hasattr(self, 'options'):
            # Max tokens from options
            if 'max_tokens' in self.options:
                max_tokens = self.options['max_tokens']
            
            # Sampler parameters from options
            sampler_option_mapping = {
                'temp': 'temp',
                'temperature': 'temp',  # alias
                'top_p': 'top_p', 
                'xtc_threshold': 'xtc_threshold',
                'xtc_probability': 'xtc_probability',
            }
            
            for option_key, param_key in sampler_option_mapping.items():
                if option_key in self.options:
                    sampler_params[param_key] = self.options[option_key]
            
            # Handle seed from options
            if 'seed' in self.options:
                mx.random.seed(self.options['seed'])
        
        # Special tokens for XTC sampling (if tokenizer has eos_token_ids)
        xtc_special_tokens = []
        if hasattr(self.tokenizer, 'eos_token_ids') and self.tokenizer.eos_token_ids:
            xtc_special_tokens = list(self.tokenizer.eos_token_ids)
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            xtc_special_tokens = [self.tokenizer.eos_token_id]
        
        # Add newline token if available
        try:
            newline_tokens = self.tokenizer.encode("\n")
            xtc_special_tokens.extend(newline_tokens)
        except:
            pass  # Skip if encoding fails
        
        sampler_params['xtc_special_tokens'] = xtc_special_tokens
        
        return max_tokens, sampler_params

async def serve(address):
    # Start asyncio gRPC server
    server = grpc.aio.server(migration_thread_pool=futures.ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ('grpc.max_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_send_message_length', 50 * 1024 * 1024),  # 50MB
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),  # 50MB
        ])
    # Add the servicer to the server
    backend_pb2_grpc.add_BackendServicer_to_server(BackendServicer(), server)
    # Bind the server to the address
    server.add_insecure_port(address)

    # Gracefully shutdown the server on SIGTERM or SIGINT
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.ensure_future(server.stop(5))
        )

    # Start the server
    await server.start()
    print("Server started. Listening on: " + address, file=sys.stderr)
    # Wait for the server to be terminated
    await server.wait_for_termination()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the gRPC server.")
    parser.add_argument(
        "--addr", default="localhost:50051", help="The address to bind the server to."
    )
    args = parser.parse_args()

    asyncio.run(serve(args.addr))
