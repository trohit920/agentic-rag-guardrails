"""FastAPI adapter layer exposing the RAG agent over HTTP.

Thin I/O boundary: request validation, API-key authentication, rate limiting,
timeout and error handling, and structured request logging. All business logic
lives in the src/ library; this package only adapts it to HTTP.
"""
