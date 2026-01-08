#!/usr/bin/env python3
"""
Simple test script to verify Bedrock integration.
"""

from ccrlm import RLM, RLMConfig

# Configure for Bedrock
config = RLMConfig(
    use_bedrock=True,
    bedrock_model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    bedrock_submodel="us.anthropic.claude-haiku-4-20250414-v1:0",
    max_turns=10,
    preserve_workspace=True,  # Keep workspace for inspection
)

print("Testing ccRLM with AWS Bedrock...")
print(f"  Root model: {config.bedrock_model}")
print(f"  Submodel: {config.bedrock_submodel}")
print()

rlm = RLM(config=config)

# Simple test prompt
result = rlm.invoke("What is 2 + 2? Please explain your reasoning.")

print("=" * 60)
print("Result:")
print("=" * 60)
print(f"Success: {result.success}")
print(f"Answer: {result.answer}")
print()
print("Metrics:")
for k, v in result.metrics.items():
    print(f"  {k}: {v}")

if result.errors:
    print("\nErrors:")
    for e in result.errors:
        print(f"  {e}")

if result.workspace_path:
    print(f"\nWorkspace preserved at: {result.workspace_path}")
